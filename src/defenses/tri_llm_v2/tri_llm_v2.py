import json
from typing import Any, Dict, List, cast

# mypy: ignore-errors
import gymnasium as gym
from browsergym.core.spaces import AnyDict, Unicode
from openai import OpenAI

from src.defenses.abstractdefense import AbstractDefense
from src.environments.observations.dom_parser import (
    prune_html,
)
from src.environments.observations.reconstruction import (
    flatten_axtree_to_str_censored,
    flatten_dom_to_str_censored,
)

# Default OpenAI client configuration
DEFAULT_OPENAI_CLIENT = OpenAI(
    api_key="sk-PCG3chp7epg_zuz5rng",
    base_url="http://localhost:4011/v1",
    max_retries=25,
)


from .models import AnnotatedElement, IntegrityLabel
from .parser import PolicyAgent

url_mapping = {
    "http://ec2-18-218-205-96.us-east-2.compute.amazonaws.com:8080": "https://www.reddit.com",
    "http://ec2-18-218-205-96.us-east-2.compute.amazonaws.com:9001": "https://gitlab.com",
    "http://ec2-18-218-205-96.us-east-2.compute.amazonaws.com:8081": "https://www.wikipedia.org",
    "http://ec2-18-218-205-96.us-east-2.compute.amazonaws.com:8090/admin": "https://www.shopping.com/admin",
    "http://ec2-18-218-205-96.us-east-2.compute.amazonaws.com:8091": "https://www.openstreetmap.org",
    "http://ec2-18-218-205-96.us-east-2.compute.amazonaws.com:8082": "https://www.shopping.com",
}

site_descriptions = {
    "http://ec2-18-218-205-96.us-east-2.compute.amazonaws.com:8080": "A Reddit-like social media forum where people share posts and engage in discussions across a wide range of topics. Users can browse communities, comment, and participate in ongoing conversations.",
    "http://ec2-18-218-205-96.us-east-2.compute.amazonaws.com:9001": "A GitLab-style development platform used for hosting and collaborating on code repositories. It supports creating projects, opening issues, and reviewing merge requests as part of software development.",
    "http://ec2-18-218-205-96.us-east-2.compute.amazonaws.com:8081": "An online encyclopedia offering articles on a wide variety of subjects. It is a resource for looking up facts, background information, and general knowledge. It is curated by an open source community of users who can edit and add new articles.",
    "http://ec2-18-218-205-96.us-east-2.compute.amazonaws.com:8090/admin": "An e-commerce store administration dashboard for managing products sold by third party sellers, customer information, and store settings. It provides tools for overseeing the operations of an online shop.",
    "http://ec2-18-218-205-96.us-east-2.compute.amazonaws.com:8091": "A map and navigation service that allows users to search for locations, explore points of interest, and plan routes. It can be used for finding directions and understanding geographic layouts.",
    "http://ec2-18-218-205-96.us-east-2.compute.amazonaws.com:8082": "An online shopping site where users can browse and purchase products sold by third party sellers. It includes a wide selection of items with prices, descriptions, images, and customer reviews.",
}


class TriDefenseV2(AbstractDefense):
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
        return "tri_defense_v2"

    def __init__(
        self,
        openai_client: OpenAI = DEFAULT_OPENAI_CLIENT,
        # async_openai_client: AsyncOpenAI = DEFAULT_OPENAI_CLIENT_ASYNC,
        sampling_params: Dict[str, Any] = {},
        sampling_params_labeler: Dict[str, Any] = {},
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.openai_client = self._wrap_openai_client(openai_client)
        self.sampling_params = sampling_params
        self.sampling_params_labeler = sampling_params_labeler
        self.llm = PolicyAgent(self.openai_client, self.sampling_params)
        self.last_url = None
        self.last_html = None
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

        site_description = "N/A"
        for url, description in site_descriptions.items():
            if current_url.startswith(url):
                new_domain = url_mapping[url]
                current_url = current_url.replace(current_url, new_domain)
                site_description = description
                break

        print(f"URL: {current_url}")
        print(f"Site description: {site_description}")

        security_policy = self.llm.get_security_policy(user_query, current_url, site_description)
        # Ensure developer content is always allowed
        if IntegrityLabel.DEVELOPER not in security_policy.integrity_levels.integrity_levels:
            security_policy.integrity_levels.levels.append(IntegrityLabel.DEVELOPER)

        # Check if we can use cached results

        html_now = observation.get("html_attack", None) or observation.get("html", None)
        if self.last_html is not None and self.last_html == html_now:
            annotated_dom = self.last_dom_cache
            self.log.info(f"Using cached defense results for {current_url}")
        else:
            self.log.info(f"Collecting annotated DOM for {current_url}")
            # Process DOM and collect metadata
            annotated_dom = AnnotatedElement.from_dom(dom, ax_tree, prune_empty_branches=True)
            site_details = json.dumps(
                {
                    "url": current_url,
                    "site_description": site_description,
                }
            )
            self._async_messages = annotated_dom.populate_metadata(
                user_query, security_policy, site_details, self.sampling_params_labeler
            )

            # Cache the results
            self.last_html = observation.get("html_attack", None) or observation.get("html", None)
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
        # print("allowed_bids:", allowed_bids)
        # print("required_parent_bids:", required_parent_bids)
        # Allowed bids included only explicitly allowed elements.
        # However, we include their parents in the trees that we generate (HTML+A11y)
        rejected_bids_html = all_bids_html - (allowed_bids | required_parent_bids)
        rejected_bids_axtree = all_bids_axtree - (allowed_bids | required_parent_bids)

        # with open(f"temp/visual_tree.{int(time.time())}.html", "w") as f:
        #     f.write(f"URL: {current_url}\n")
        #     f.write(f"Site description: {site_description}\n")
        #     f.write(annotated_dom.as_html_tree_top_full)

        # with open(f"temp_dom/annotated_dom.{int(time.time())}.json", "w") as f:
        #     json.dump(annotated_dom.model_dump(), f, indent=4)

        return {
            "screenshot_censored": self.capture_censored_screenshot(page, [str(bid) for bid in allowed_bids]),
            "html_censored": prune_html(flatten_dom_to_str_censored(dom, censor_bids=list(rejected_bids_html))),
            "a11y_censored": flatten_axtree_to_str_censored(
                ax_tree, censor_bids=list(rejected_bids_axtree), bid_capabilities=annotated_dom.capability_map
            ),
            "security_policy": security_policy.model_dump(),
            "allowed_elements": allowed_elements.model_dump(),
            "annotated_dom": annotated_dom.model_dump(),
            "allowed_bids": list(allowed_bids),
            "rejected_bids_html": list(rejected_bids_html),
            "rejected_bids_axtree": list(rejected_bids_axtree),
            "annotated_dom_readable": annotated_dom_readable,
            "cap_set": annotated_dom.capability_map,
        }
