import json

from browsergym.utils.obs import flatten_axtree_to_str

from src.agents.occam.processor import TextObervationProcessor
from src.agents.occam.prune import prune_tree


def load_sample_tree():
    with open("tests/occam/sample_axtree.json", "r") as f:
        data = json.load(f)
    return data["nodes"] if "nodes" in data else data


def test_prune_tree_reduces_size():
    axtree = load_sample_tree()
    original = flatten_axtree_to_str({"nodes": axtree})
    _, _, root = TextObervationProcessor.parse_accessibility_tree(axtree)
    pruned = prune_tree(root, mode="str")
    assert isinstance(pruned, str)
    assert len(pruned) < len(original)


def test_parse_tree_uses_browsergym_ids():
    axtree = load_sample_tree()
    tree_str, _, root = TextObervationProcessor.parse_accessibility_tree(axtree)
    assert "[148]" in tree_str
    assert len(tree_str.splitlines()) > 1
    assert root.search_node_by_id("148") is not None
