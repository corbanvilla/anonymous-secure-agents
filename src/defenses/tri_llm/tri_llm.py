from typing import Any, Dict, List, cast

# mypy: ignore-errors
import gymnasium as gym
from browsergym.core.spaces import AnyDict, Unicode
from openai import AsyncOpenAI, OpenAI

from src.defenses.abstractdefense import AbstractDefense
from src.environments.observations.dom_parser import (
    prune_html,
)
from src.environments.observations.reconstruction import (
    flatten_axtree_to_str_censored,
    flatten_dom_to_str_censored,
)

from .models import AnnotatedElement
from .parser import PolicyAgent


class TriDefense(AbstractDefense):
    """
    Single-stage defense that uses LLMs to analyze and filter DOM elements.
    This defense performs a single pass through the DOM to determine which elements
    should be allowed based on the user's query and element metadata.
    """

    # Defense-specific observation space
    DEFENSE_OBS_SPACE = {
        "security_policy": AnyDict(),
        "allowed_elements": AnyDict(),
        "annotated_dom": AnyDict(),
        "allowed_bids": gym.spaces.Sequence(Unicode()),
        "rejected_bids_html": gym.spaces.Sequence(Unicode()),
        "rejected_bids_axtree": gym.spaces.Sequence(Unicode()),
        "annotated_dom_readable": Unicode(),
        "cap_set": AnyDict(),
    }

    @staticmethod
    def get_defense_id() -> str:
        return "tri_defense"

    def __init__(
        self,
        openai_client: OpenAI = OpenAI(),
        async_openai_client: AsyncOpenAI = AsyncOpenAI(),
        sampling_params: Dict[str, Any] = {},
        sampling_params_labeler: Dict[str, Any] = {},
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.openai_client = self._wrap_openai_client(openai_client)
        self.async_openai_client = self._wrap_async_openai_client(async_openai_client)
        self.sampling_params = sampling_params
        self.sampling_params_labeler = sampling_params_labeler
        self.llm = PolicyAgent(self.openai_client, self.sampling_params)
        self.last_url = None
        self.last_dom_cache = None

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

        security_policy = self.llm.get_security_policy(user_query, current_url)

        # Check if we can use cached results
        if current_url == self.last_url and self.last_dom_cache is not None:
            annotated_dom = self.last_dom_cache
            self.log.info(f"Using cached defense results for {current_url}")
        else:
            self.log.info(f"Collecting annotated DOM for {current_url}")
            # Process DOM and collect metadata
            annotated_dom = AnnotatedElement.from_dom(dom, ax_tree, prune_empty_branches=True)
            annotated_dom.populate_metadata(
                user_query, security_policy, self.async_openai_client, self.sampling_params_labeler
            )

            # Cache the results
            self.last_url = current_url
            self.last_dom_cache = annotated_dom

        annotated_dom_readable = annotated_dom.as_html_tree_top_full

        all_bids_html = set(annotated_dom.all_bids)
        assert annotated_dom.visible_bids is not None
        all_bids_axtree = set(annotated_dom.visible_bids)

        allowed_elements = self.llm.filter_by_security_policy_strict(annotated_dom, security_policy)

        allowed_bids_list: List[int] = cast(List[int], allowed_elements.allowed_bids)
        allowed_bids = set(allowed_bids_list)
        required_parent_bids = set(annotated_dom.required_parent_bids(list(allowed_bids)))
        print("allowed_bids:", allowed_bids)
        print("required_parent_bids:", required_parent_bids)
        # Allowed bids included only explicitly allowed elements.
        # However, we include their parents in the trees that we generate (HTML+A11y)
        rejected_bids_html = all_bids_html - (allowed_bids | required_parent_bids)
        rejected_bids_axtree = all_bids_axtree - (allowed_bids | required_parent_bids)

        return {
            "screenshot_censored": self.capture_censored_screenshot(page, [str(bid) for bid in allowed_bids]),
            "html_censored": prune_html(flatten_dom_to_str_censored(dom, censor_bids=list(rejected_bids_html))),
            "a11y_censored": flatten_axtree_to_str_censored(ax_tree, censor_bids=list(rejected_bids_axtree)),
            "security_policy": security_policy.model_dump(),
            "allowed_elements": allowed_elements.model_dump(),
            "annotated_dom": annotated_dom.model_dump(),
            "allowed_bids": list(allowed_bids),
            "rejected_bids_html": list(rejected_bids_html),
            "rejected_bids_axtree": list(rejected_bids_axtree),
            "annotated_dom_readable": annotated_dom_readable,
            "cap_set": annotated_dom.capability_map,
        }
