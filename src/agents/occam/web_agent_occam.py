from __future__ import annotations

from src.agents.occam.prune import parse_tree_for_agent
from src.agents.rllm.web_agent import WebAgent as BaseWebAgent


class WebAgent(BaseWebAgent):
    """A variant of :class:`~src.agents.web_agent.WebAgent` that uses the
    Occam observation pruning strategy."""

    def _preproc_obs(self, obs: dict) -> dict:
        """Return a pruned observation for the agent."""
        assert "axtree_object" in obs, "axtree_object not in obs"
        pruned_axtree_agent = parse_tree_for_agent(obs["axtree_object"])
        return {
            "chat_messages": obs["chat_messages"],
            "goal_object": obs["goal_object"],
            "last_action": obs["last_action"],
            "last_action_error": obs["last_action_error"],
            "open_pages_urls": obs["open_pages_urls"],
            "open_pages_titles": obs["open_pages_titles"],
            "active_page_index": obs["active_page_index"],
            "screenshot": obs["screenshot"],
            "axtree_txt": pruned_axtree_agent,
            "pruned_html": obs["html"],
        }
