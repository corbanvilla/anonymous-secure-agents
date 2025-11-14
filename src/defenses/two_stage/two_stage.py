import logging
from typing import Any, Dict

# mypy: ignore-errors

from src.defenses.one_stage.one_stage import OneStageDefense
from src.environments.observations.dom_parser import (
    flatten_dom_to_str_get_dict,
    prune_html,
)
from src.environments.observations.reconstruction import (
    flatten_axtree_to_str_censored,
    flatten_dom_to_str_censored,
)

logger = logging.getLogger(__name__)


class TwoStageDefense(OneStageDefense):
    """
    Two-stage defense that performs additional analysis on uncertain elements.
    This defense extends the one-stage defense by performing a second pass
    on elements that were not clearly allowed or rejected in the first pass.
    """

    @classmethod
    def get_defense_id(cls) -> str:
        return "two_stage_defense"

    def run_defense(self, observation: Dict[str, Any], page) -> Dict[str, Any]:
        """
        Run the two-stage defense pipeline.

        Args:
            observation: The current environment observation
            page: The browser page object used for capturing censored screenshots

        Returns:
            Dict containing defense results including allowed/rejected elements
        """
        user_query = observation["goal"]
        ax_tree = observation["axtree_object"]
        dom = observation["dom_object"]

        html, dom_elements = flatten_dom_to_str_get_dict(dom)
        pruned_html = prune_html(html)

        # Process DOM and collect metadata
        annotated_dom, dom_owners, dom_metadata = self._collect_annotated_dom(dom)
        all_bids = set(annotated_dom.all_bids)

        # Stage 2.a: First pass analysis
        logger.info("Running first pass analysis...")
        first = self.llm.get_html_allowed_first_pass(pruned_html, user_query)

        # Stage 2.b: Second pass for uncertain elements
        strict = None
        if first.unknown_bids:
            logger.info("Running second pass analysis on uncertain elements...")
            summaries = self.llm.get_bid_summaries(first.unknown_bids, dom_elements)
            annotated_dom.apply_summary(summaries)
            strict = self.llm.get_html_allowed_strict(pruned_html, user_query, summaries)

        # Stage 2.c: Combine results
        allowed_elements = self.llm.get_allowed_elements(first, strict)
        allowed_bids = set(allowed_elements.allowed_bids)
        rejected_bids = all_bids - allowed_bids
        logger.info(f"Allowed BIDs: {allowed_bids}")
        logger.info(f"Rejected BIDs: {rejected_bids}")

        return {
            "screenshot_censored": self.capture_censored_screenshot(page, [str(bid) for bid in allowed_bids]),
            "html_full": prune_html(flatten_dom_to_str_censored(dom)),
            "full_a11y": flatten_axtree_to_str_censored(ax_tree),
            "html_censored": prune_html(flatten_dom_to_str_censored(dom, censor_bids=list(rejected_bids))),
            "a11y_censored": flatten_axtree_to_str_censored(ax_tree, censor_bids=list(rejected_bids)),
            "dom_owners": dom_owners.model_dump(),
            "dom_metadata": dom_metadata.model_dump(),
            "allowed_elements": allowed_elements.model_dump(),
            "annotated_dom": annotated_dom.model_dump(),
            "allowed_bids": list(allowed_bids),
            "rejected_bids": list(rejected_bids),
        }
