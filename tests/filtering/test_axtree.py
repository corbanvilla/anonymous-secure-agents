import json
import os
from typing import Any, Dict, List

from browsergym.utils.obs import flatten_axtree_to_str

from src.defenses.one_stage.models.dom import AnnotatedDom
from src.environments.observations.reconstruction import (
    flatten_axtree_to_str_and_dict,
    flatten_axtree_to_str_censored,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUTS_DIR = os.path.join(CURRENT_DIR, "inputs")
OUTPUTS_FILTERED_DIR = os.path.join(CURRENT_DIR, "outputs_filtered")
OUTPUTS_UNFILTERED_DIR = os.path.join(CURRENT_DIR, "outputs_unfiltered")

SAMPLES: List[Dict[str, Any]] = [
    {
        "name": "sample1",
        "outputs": [
            {
                "censor_bids": [401],
                "name": "sample1_censored_output_401",
            },
            {
                "censor_bids": [397],
                "name": "sample1_censored_branch",
            },
            {
                "censor_bids": [397, 1497, 1931, 1964, 1951],
                "name": "sample1_censored_branch_many",
            },
        ],
    }
]


def load_input(name):
    with open(name, "r") as f:
        return json.load(f)


def load_output(name):
    with open(name, "r") as f:
        return f.read()


def write_output(name, content):
    with open(name, "w") as f:
        f.write(content)


def test_censored_flatten_no_censoring():
    """
    Test that the censored flatten function maintains parity with browsergym's flatten function.
    """

    for sample in SAMPLES:
        input_file = os.path.join(INPUTS_DIR, f"{sample['name']}.json")
        output_uncensored_file = os.path.join(OUTPUTS_UNFILTERED_DIR, f"{sample['name']}.txt")
        input_axtree = load_input(input_file)
        expected_axtree_str = flatten_axtree_to_str(input_axtree)

        actual_axtree_str = flatten_axtree_to_str_censored(input_axtree, censor_bids=[])
        actual_axtree_str_and_dict = flatten_axtree_to_str_and_dict(input_axtree)
        assert actual_axtree_str == expected_axtree_str
        assert actual_axtree_str_and_dict[0] == expected_axtree_str

        # Write the file to unfiltered for reference
        write_output(output_uncensored_file, actual_axtree_str)


def test_annotated_element_dict():
    """
    Test that the annotated element dict is valid.
    """

    for sample in SAMPLES:
        input_file = os.path.join(INPUTS_DIR, f"{sample['name']}.json")
        output_dict_file = os.path.join(OUTPUTS_UNFILTERED_DIR, f"{sample['name']}_dict.json")
        output_model_file = os.path.join(OUTPUTS_UNFILTERED_DIR, f"{sample['name']}_model.json")
        input_axtree = load_input(input_file)

        # test dict
        _, output_dict = flatten_axtree_to_str_and_dict(input_axtree)
        write_output(output_dict_file, json.dumps(output_dict, indent=2))

        # load to model
        annotated_dom = AnnotatedDom.from_a11y_tree(output_dict)

        write_output(output_model_file, annotated_dom.model_dump_json(indent=2))


def test_censored_flatten_with_censoring():
    """
    Test that the censored flatten function maintains parity with browsergym's flatten function.
    """

    for sample in SAMPLES:
        input_file = os.path.join(INPUTS_DIR, f"{sample['name']}.json")
        input_axtree = load_input(input_file)

        for output in sample["outputs"]:
            censor_bids: List[int] = output["censor_bids"]
            expected_output_file = os.path.join(OUTPUTS_FILTERED_DIR, f"{output['name']}.txt")
            expected_axtree_str = load_output(expected_output_file)

            uncensored_axtree_str = flatten_axtree_to_str(input_axtree)
            actual_axtree_str = flatten_axtree_to_str_censored(input_axtree, censor_bids=censor_bids)

            assert actual_axtree_str == expected_axtree_str
            for bid in censor_bids:
                assert f"[{bid}]" in uncensored_axtree_str
                assert f"[{bid}]" not in actual_axtree_str


def test_required_parent_bids_simple():
    ax_tree = load_input(os.path.join(INPUTS_DIR, "sample1.json"))

    _, tree_dict = flatten_axtree_to_str_and_dict(ax_tree)

    dom = AnnotatedDom.from_a11y_tree(tree_dict)

    allowed = 229

    required = set(dom.discover_bid_path(allowed))
    expected = {229, 228, 225, 222}

    assert expected.issubset(required)


def test_required_parent_bids_multiple():
    ax_tree = load_input(os.path.join(INPUTS_DIR, "sample1.json"))

    _, tree_dict = flatten_axtree_to_str_and_dict(ax_tree)

    dom = AnnotatedDom.from_a11y_tree(tree_dict)

    allowed = [227, 242]

    required = set(dom.required_parent_bids(allowed))

    expected = {227, 226, 225, 222, 242}

    assert expected.issubset(required)
