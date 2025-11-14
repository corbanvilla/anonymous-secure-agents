import logging
import time
from typing import Any, Dict, List, cast

from browsergym.core.spaces import AnyDict
from openai import OpenAI

from src.defenses.stricted_one_stage.stricted_one_stage import StrictedOneStageDefense

logger = logging.getLogger(__name__)


class StrictedDefense(StrictedOneStageDefense):
    """Stricter defense that also filters by content ownership."""

    DEFENSE_OBS_SPACE = StrictedOneStageDefense.DEFENSE_OBS_SPACE.copy()
    DEFENSE_OBS_SPACE["allowed_owners"] = AnyDict()

    @staticmethod
    def get_defense_id() -> str:
        return "stricted_defense"

    def __init__(self, openai_client: OpenAI = OpenAI(), sampling_params: Dict[str, Any] = {}, **kwargs) -> None:
        super().__init__(openai_client=openai_client, sampling_params=sampling_params, **kwargs)

    def run_defense(self, observation: Dict[str, Any], page) -> Dict[str, Any]:
        user_query = observation["goal"]
        ax_tree = observation["axtree_object"]
        dom = observation["dom_object"]
        current_url = observation["url"]

        if current_url == self.last_url and self.last_dom_cache is not None:
            annotated_dom, dom_owners, dom_metadata = self.last_dom_cache
            logger.info(f"Using cached defense results for {current_url}")
        else:
            logger.info(f"Collecting annotated DOM for {current_url}")
            annotated_dom, dom_owners, dom_metadata = self._collect_annotated_dom(dom)
            self.last_url = current_url
            self.last_dom_cache = (annotated_dom, dom_owners, dom_metadata)

        logger.info("Fetching allowed owners from gemini…")
        allowed_owners = self.llm.get_allowed_owners(user_query)
        annotated_dom.filter_by_owners(cast(List[str], allowed_owners.owners))

        all_bids_html = set(annotated_dom.all_bids())
        all_bids_axtree = set(self._collect_a11y_bids(ax_tree))

        logger.info("Fetching allowed elements from gemini…")
        redacted_html = annotated_dom.export_html_censored()
        allowed_elements_start = time.time()
        allowed_elements = self.llm.get_html_allowed(redacted_html, user_query)
        allowed_elements_end = time.time()
        logger.info(f"Allowed elements returned in {allowed_elements_end - allowed_elements_start:.2f} seconds")

        allowed_bids = set(allowed_elements.allowed_bids())
        required_parent_bids = set(annotated_dom.required_parent_bids(list(allowed_bids)))
        rejected_bids_html = all_bids_html - (allowed_bids | required_parent_bids)
        rejected_bids_axtree = all_bids_axtree - (allowed_bids | required_parent_bids)

        return {
            "screenshot_censored": self.capture_censored_screenshot(page, [str(bid) for bid in allowed_bids]),
            "html_censored": self._pruned_html(dom, rejected_bids_html),
            "a11y_censored": self._pruned_a11y(ax_tree, rejected_bids_axtree),
            "dom_owners": dom_owners.model_dump(),
            "dom_metadata": dom_metadata.model_dump(),
            "allowed_elements": allowed_elements.model_dump(),
            "annotated_dom": annotated_dom.model_dump(),
            "allowed_bids": list(allowed_bids),
            "rejected_bids_html": [str(b) for b in rejected_bids_html],
            "rejected_bids_axtree": [str(b) for b in rejected_bids_axtree],
            "allowed_owners": allowed_owners.model_dump(),
        }

    def _pruned_html(self, dom: Dict[str, Any], rejected: set[int]) -> str:
        from src.environments.observations.dom_parser import prune_html
        from src.environments.observations.reconstruction import (
            flatten_dom_to_str_censored,
        )

        return prune_html(flatten_dom_to_str_censored(dom, censor_bids=list(rejected)))

    def _pruned_a11y(self, ax_tree: Dict[str, Any], rejected: set[int]) -> str:
        from src.environments.observations.reconstruction import (
            flatten_axtree_to_str_censored,
        )

        return flatten_axtree_to_str_censored(ax_tree, censor_bids=list(rejected))
