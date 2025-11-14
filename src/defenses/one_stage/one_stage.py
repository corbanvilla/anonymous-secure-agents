import time
from typing import Any, Dict, List, Tuple

# mypy: ignore-errors

import gymnasium as gym
from browsergym.core.spaces import AnyDict, Unicode
from openai import OpenAI

from src.defenses.abstractdefense import AbstractDefense
from src.environments.observations.dom_parser import (
    flatten_dom_to_str_get_dict,
    prune_dom_dict,
    prune_html,
)
from src.environments.observations.reconstruction import (
    flatten_axtree_to_str_and_dict,
    flatten_axtree_to_str_censored,
    flatten_dom_to_str_censored,
)

from .models import AnnotatedDom
from .parser import OneStageLLM


class OneStageDefense(AbstractDefense):
    """
    Single-stage defense that uses LLMs to analyze and filter DOM elements.
    This defense performs a single pass through the DOM to determine which elements
    should be allowed based on the user's query and element metadata.
    """

    # Defense-specific observation space
    DEFENSE_OBS_SPACE = {
        "dom_owners": AnyDict(),
        "dom_metadata": AnyDict(),
        "allowed_elements": AnyDict(),
        "annotated_dom": AnyDict(),
        "allowed_bids": gym.spaces.Sequence(Unicode()),
        "rejected_bids_html": gym.spaces.Sequence(Unicode()),
        "rejected_bids_axtree": gym.spaces.Sequence(Unicode()),
    }

    @staticmethod
    def get_defense_id() -> str:
        return "one_stage_defense_v1_lax"

    def __init__(
        self,
        openai_client: OpenAI = OpenAI(),
        sampling_params: Dict[str, Any] = {},
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.openai_client = self._wrap_openai_client(openai_client)
        self.sampling_params = sampling_params
        self.llm: OneStageLLM = OneStageLLM(self.openai_client, self.sampling_params)
        self.last_url = None
        self.last_dom_cache = None

    def _collect_a11y_bids(self, axtree_object: Dict[str, Any]) -> List[int]:
        """
        Collect the BIDs of the accessibility tree.
        """
        _, a11y_dict = flatten_axtree_to_str_and_dict(axtree_object)
        a11y_model = AnnotatedDom.from_a11y_tree(a11y_dict)
        return a11y_model.all_bids

    def _collect_annotated_dom(self, dom: Dict[str, Any]) -> Tuple[AnnotatedDom, Any, Any]:
        """
        Process the DOM and collect metadata to create an annotated DOM.

        Args:
            dom: The DOM object from the observation

        Returns:
            Tuple containing:
            - AnnotatedDom object
            - DOM owners
            - DOM metadata
        """
        # Process DOM
        html, dom_elements = flatten_dom_to_str_get_dict(dom)
        pruned_html = prune_html(html)
        pruned_dom = prune_dom_dict(dom_elements)
        annotated_dom = AnnotatedDom.from_a11y_tree(pruned_dom)

        # Stage 1.a: Collect DOM owners
        self.log.info("Fetching HTML metadata from gemini…")
        dom_owners_start = time.time()
        dom_owners = self.llm.get_html_owners(pruned_html)
        dom_owners_end = time.time()
        self.log.info(f"DOM owners returned in {dom_owners_end - dom_owners_start:.2f} seconds")

        # Stage 1.b: Collect DOM metadata
        self.log.info("Fetching HTML metadata from gemini…")
        dom_metadata_start = time.time()
        dom_metadata = self.llm.get_html_metadata(pruned_html)
        dom_metadata_end = time.time()
        self.log.info(f"DOM metadata returned in {dom_metadata_end - dom_metadata_start:.2f} seconds")

        # Stage 1.c: Aggregate data
        annotated_dom.apply_owners(dom_owners)
        annotated_dom.apply_metadata(dom_metadata)

        return annotated_dom, dom_owners, dom_metadata

    def run_defense(self, observation: Dict[str, Any], page) -> Dict[str, Any]:
        """
        Run the single-stage defense pipeline.

        Args:
            observation: The current environment observation
            page: The browser page object used for capturing censored screenshots

        Returns:
            Dict containing defense results including allowed/rejected elements
        """
        user_query = observation["goal"]
        ax_tree = observation["axtree_object"]
        dom = observation["dom_object"]
        current_url = observation["url"]

        # Check if we can use cached results
        if current_url == self.last_url and self.last_dom_cache is not None:
            annotated_dom, dom_owners, dom_metadata = self.last_dom_cache
            self.log.info(f"Using cached defense results for {current_url}")
        else:
            self.log.info(f"Collecting annotated DOM for {current_url}")
            # Process DOM and collect metadata
            annotated_dom, dom_owners, dom_metadata = self._collect_annotated_dom(dom)
            # Cache the results
            self.last_url = current_url
            self.last_dom_cache = (annotated_dom, dom_owners, dom_metadata)

        all_bids_html = set(annotated_dom.all_bids)
        all_bids_axtree = set(self._collect_a11y_bids(ax_tree))

        # Stage 2: Determine allowed elements
        self.log.info("Fetching allowed elements from gemini…")
        redacted_html = annotated_dom.export_html_censored()
        allowed_elements_start = time.time()
        allowed_elements = self.llm.get_html_allowed(redacted_html, user_query)
        allowed_elements_end = time.time()
        self.log.info(f"Allowed elements returned in {allowed_elements_end - allowed_elements_start:.2f} seconds")

        allowed_bids = set(allowed_elements.allowed_bids)
        required_parent_bids = set(annotated_dom.required_parent_bids(list(allowed_bids)))
        # Allowed bids included only explicitly allowed elements.
        # However, we include their parents in the trees that we generate (HTML+A11y)
        rejected_bids_html = all_bids_html - (allowed_bids | required_parent_bids)
        rejected_bids_axtree = all_bids_axtree - (allowed_bids | required_parent_bids)

        return {
            "screenshot_censored": self.capture_censored_screenshot(page, list(allowed_bids)),
            "html_censored": prune_html(flatten_dom_to_str_censored(dom, censor_bids=list(rejected_bids_html))),
            "a11y_censored": flatten_axtree_to_str_censored(ax_tree, censor_bids=list(rejected_bids_axtree)),
            "dom_owners": dom_owners.model_dump(),
            "dom_metadata": dom_metadata.model_dump(),
            "allowed_elements": allowed_elements.model_dump(),
            "annotated_dom": annotated_dom.model_dump(),
            "allowed_bids": list(allowed_bids),
            "rejected_bids_html": list(rejected_bids_html),
            "rejected_bids_axtree": list(rejected_bids_axtree),
        }
