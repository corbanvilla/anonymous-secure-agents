import json
import os

import pytest

from src.defenses.one_stage.models.dom import AnnotatedDom, PolicyViolation
from src.defenses.one_stage.models.policy import ContentOwners
from src.environments.observations.reconstruction import (
    flatten_axtree_to_str_and_dict,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUTS_DIR = os.path.join(CURRENT_DIR, "inputs")


def load_input(name: str) -> dict:
    with open(os.path.join(INPUTS_DIR, name), "r", encoding="utf-8") as f:
        return json.load(f)


def build_dom_from_sample(name: str) -> AnnotatedDom:
    ax_tree = load_input(name)

    _, tree_dict = flatten_axtree_to_str_and_dict(ax_tree)

    return AnnotatedDom.from_a11y_tree(tree_dict)


def test_allowed_flow_policy():
    """
    Developer can flow to user/third_party and between user and third_party freely.
    """

    dom = build_dom_from_sample("sample1.json")

    # Assign ownership: root banner is developer, links and items are user/third_party

    owners = ContentOwners(developer=[222], user=[226, 227, 228, 229, 230, 232, 235], third_party=[237, 242])

    dom.apply_owners(owners)

    assert dom.enforce_content_ownership_flow() == []


def test_forbidden_flow_user_to_developer():
    """
    A user -> developer transition should be flagged.
    """

    dom = build_dom_from_sample("sample1.json")

    owners = ContentOwners(developer=[227], user=[226], third_party=[])

    dom.apply_owners(owners)

    with pytest.raises(PolicyViolation) as excinfo:
        dom.enforce_content_ownership_flow()

    assert "user -> developer" in str(excinfo.value)


def test_forbidden_flow_third_party_to_developer():
    """
    A third party -> developer transition should be flagged.
    """

    dom = build_dom_from_sample("sample1.json")

    owners = ContentOwners(developer=[226], user=[], third_party=[225])

    dom.apply_owners(owners)

    with pytest.raises(PolicyViolation) as excinfo:
        dom.enforce_content_ownership_flow()

    assert "third_party -> developer" in str(excinfo.value)
