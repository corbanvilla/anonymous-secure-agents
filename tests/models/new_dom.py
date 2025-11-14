import json
import logging
import time

from src.api_clients.async_logging import AsyncOpenAILoggingProxy
from src.db.helpers.trajectories import load_observation
from src.defenses.tri_llm_v2.models.dom import AnnotatedElement
from src.defenses.tri_llm_v2.models.enums import IntegrityLabel
from src.defenses.tri_llm_v2.models.policy import IntegritySet, SecurityPolicy
from src.defenses.tri_llm_v2.parser import PolicyAgent
from src.environments.observations.reconstruction import flatten_axtree_to_str_censored
from src.experiments.config.defaults import DEFAULT_OPENAI_CLIENT_ASYNC

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


def get_latest_observation():
    obs_id = 37745 - 12 * 75 - 3

    # Load the observation data with large files
    obs_data = load_observation(obs_id, load_screenshots=False, load_large_files=True)

    if obs_data is None:
        print(f"No observation found with ID: {obs_id}")
        return None

    # Get the axtree_object from the observation data
    assert "axtree_object" in obs_data
    axtree_object = obs_data["axtree_object"]

    # Print first 1000 chars of the axtree_object
    print("First 1000 characters of axtree_object:")
    print(str(axtree_object)[:1000])

    assert "dom_object" in obs_data
    dom_object = obs_data["dom_object"]
    print("First 1000 characters of dom:")
    print(str(dom_object)[:1000])

    from src.environments.observations.reconstruction import flatten_dom_to_str_censored

    html = flatten_dom_to_str_censored(dom_object)
    with open("html.html", "w") as f:
        f.write(html)

    axtree_object_tree = AnnotatedElement.from_axtree(axtree_object)
    dom_object_tree = AnnotatedElement.from_dom(dom_object)
    dom_object_pruned_tree = AnnotatedElement.from_dom(dom_object, axtree_object, prune_empty_branches=True)

    print(len(axtree_object_tree.all_bids))
    print(len(dom_object_tree.all_bids))
    print(len(dom_object_pruned_tree.all_bids))

    with open("axtree_object.json", "w") as f:
        json.dump(axtree_object_tree.model_dump(), f, indent=4)
    with open("dom_object.json", "w") as f:
        json.dump(dom_object_tree.model_dump(), f, indent=4)
    with open("dom_object_pruned.json", "w") as f:
        json.dump(dom_object_pruned_tree.model_dump(), f, indent=4)
    with open("html.html", "w") as f:
        f.write(obs_data["html"])

    request_logs = []
    openai_client = AsyncOpenAILoggingProxy(DEFAULT_OPENAI_CLIENT_ASYNC, request_logs)
    # samping_params = {"model": "gpt-oss-20B"}
    # samping_params = {"model": "gpt-5-nano"}
    # samping_params = {"model": "gemini-2.0-flash"}
    samping_params = {"model": "vertex_ai/gemini-2.0-flash-lite"}
    start = time.time()
    end = None

    user_query = obs_data["goal"]
    security_policy = SecurityPolicy(integrity_levels=IntegritySet(levels=[IntegrityLabel.DEVELOPER]))

    try:
        dom_object_pruned_tree.populate_metadata(user_query, security_policy, openai_client, samping_params)
        end = time.time()
    except KeyboardInterrupt:
        pass

    model_name = samping_params["model"].split("/")[-1]
    file = f"test_new_dom/request_logs_{len(request_logs)}_logs_{model_name}_{time.time()}.json"
    with open(file, "w") as f:
        f.write(json.dumps(request_logs, indent=4))
    print(f"Wrote {len(request_logs)} request logs to {file}")
    if end is not None:
        print(f"Time taken: {end - start} seconds")
    else:
        end = time.time()
        print(f"Keyboard interrupt after {end - start} seconds")

    with open("annotated_tree.json", "w") as f:
        json.dump(dom_object_pruned_tree.model_dump(), f, indent=4)

    with open("visual_tree.html", "w") as f:
        f.write(dom_object_pruned_tree.as_html_tree_top_full)

    policy_parser = PolicyAgent(openai_client, samping_params)  # type: ignore
    allowed_bids = set(
        policy_parser.filter_by_security_policy_strict(dom_object_pruned_tree, security_policy).allowed_bids
    )
    assert dom_object_pruned_tree.visible_bids is not None
    all_bids_axtree = set(dom_object_pruned_tree.visible_bids)
    censor_bids_axtree = all_bids_axtree - allowed_bids

    a11y = flatten_axtree_to_str_censored(axtree_object, censor_bids=list(censor_bids_axtree))
    with open("a11y.txt", "w") as f:
        f.write(a11y)

    cap_map = dom_object_pruned_tree.capability_map
    print("cap_map:", cap_map)
    assert cap_map is not None
    a11y_caps = flatten_axtree_to_str_censored(
        axtree_object, censor_bids=list(censor_bids_axtree), bid_capabilities=cap_map
    )
    with open("a11y_caps.txt", "w") as f:
        f.write(a11y_caps)

    return axtree_object_tree, dom_object_tree, dom_object_pruned_tree


if __name__ == "__main__":
    get_latest_observation()
