# copied in from agentoccam
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, TypedDict, cast

from browsergym.core.constants import BROWSERGYM_ID_ATTRIBUTE as BID_ATTR
from browsergym.core.constants import BROWSERGYM_SETOFMARKS_ATTRIBUTE as SOM_ATTR
from browsergym.core.constants import BROWSERGYM_VISIBILITY_ATTRIBUTE as VIS_ATTR
from browsergym.utils.obs import _process_bid
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

IGNORED_AXTREE_ROLES = ["LineBreak"]

IGNORED_AXTREE_PROPERTIES = (
    "editable",
    "readonly",
    "level",
    "settable",
    "multiline",
    "invalid",
    "focusable",
)


class DomNode(TypedDict, total=False):
    """Typed representation of a DOM element used in observation helpers."""

    tag_name: str
    bid: Optional[str]
    attributes: Dict[str, str]
    children: List["DomNode"]


EMPTY_NODE: DomNode = {"tag_name": "", "bid": None, "attributes": {}, "children": []}


def flatten_dom_to_str_get_dict(
    dom_snapshot: Dict[str, Any],
    extra_properties: Optional[Dict[str, Any]] = None,
    with_visible: bool = False,
    with_clickable: bool = False,
    with_center_coords: bool = False,
    with_bounding_box_coords: bool = False,
    with_som: bool = False,
    filter_visible_only: bool = False,
    filter_with_bid_only: bool = False,
    filter_som_only: bool = False,
    coord_decimals: int = 0,
    hide_bid_if_invisible: int = False,
    hide_all_bids: bool = False,
) -> tuple[str, DomNode]:
    """Formats a DOM snapshot into a string text"""

    extra_properties = extra_properties or {}

    def to_string(idx):
        if idx == -1:
            return None
        else:
            return dom_snapshot["strings"][idx]

    def parse_document(document_idx: int) -> tuple[str, DomNode]:
        # adapted from [natbot](https://github.com/nat/natbot)

        nodes = dom_snapshot["documents"][document_idx]["nodes"]
        node_children = defaultdict(lambda: [])

        for node_idx in range(len(nodes["nodeName"])):
            parent_idx = nodes["parentIndex"][node_idx]
            if parent_idx != -1:
                node_children[parent_idx].append(node_idx)

        def dfs(node_idx: int, parent_node_skipped: bool) -> tuple[str, DomNode]:
            # https://developer.mozilla.org/en-US/docs/Web/API/Node/nodeType
            # https://developer.mozilla.org/en-US/docs/Web/API/Node/nodeName
            # https://developer.mozilla.org/en-US/docs/Web/API/Node/nodeValue

            node_type = nodes["nodeType"][node_idx]
            node_name = to_string(nodes["nodeName"][node_idx])
            node_value = to_string(nodes["nodeValue"][node_idx])
            html_before = ""
            html_after = ""
            skip_node = False
            node_dict = None

            # text nodes: print text content only if parent was not skipped
            if node_type == 3:  # node_name == "#text"
                if not parent_node_skipped and node_value is not None:
                    html_before += node_value
                    node_dict = {"element_text": node_value.strip()}

            # CData nodes: print content only if parent was not skipped
            elif node_type == 4:  # node_name == "#cdata-section":
                if not parent_node_skipped and node_value is not None:
                    html_before += f"<!CDATA[[{node_value}]]>"

            # processing instructions, comments, documents, doctypes, document fragments: don't print
            elif node_type in (7, 8, 9, 10, 11):
                skip_node = True
                # for document‐type nodes, collect children under a container
                node_dict = {"children": []}

            # now we should have an element node
            else:
                assert node_type == 1

                tag_name = node_name.lower().strip() if node_name else ""
                attributes = []  # to be printed as attributes with the tag
                bid = None

                # parse node attributes
                node_attr_idxs = nodes["attributes"][node_idx]
                for i in range(0, len(node_attr_idxs), 2):
                    attr_name = to_string(node_attr_idxs[i])
                    attr_value = to_string(node_attr_idxs[i + 1])

                    # extract and print bid
                    if attr_name == BID_ATTR:
                        bid = attr_value
                    # ignore browsergym attributes
                    elif attr_name in (VIS_ATTR, SOM_ATTR):
                        pass
                    # print other attributes
                    else:
                        if attr_value is None:
                            # attribute value missing
                            attributes.append(f"{attr_name}")
                        else:
                            # attribute value present
                            attributes.append(f'{attr_name}="{attr_value}"')

                skip_node, extra_attributes_to_print = _process_bid(
                    bid,
                    extra_properties=extra_properties,
                    with_visible=with_visible,
                    with_clickable=with_clickable,
                    with_center_coords=with_center_coords,
                    with_bounding_box_coords=with_bounding_box_coords,
                    with_som=with_som,
                    filter_visible_only=filter_visible_only,
                    filter_with_bid_only=filter_with_bid_only,
                    filter_som_only=filter_som_only,
                    coord_decimals=coord_decimals,
                )

                # insert extra attributes before regular attributes
                attributes = extra_attributes_to_print + attributes

                # insert bid as first attribute
                if not (
                    hide_all_bids
                    or bid is None
                    or (hide_bid_if_invisible and extra_properties.get(bid, {}).get("visibility", 0) < 0.5)
                ):
                    attributes.insert(0, f'bid="{bid}"')

                if not skip_node:
                    # print node opening tag, with its attributes
                    html_before += f"<{tag_name}" + " ".join([""] + attributes) + ">"
                    # print node closing tag
                    html_after += f"</{tag_name}>"

                    attributes_dict = {
                        k: v.strip('"')
                        for (k, v) in [
                            attr.split("=", maxsplit=1)
                            for attr in attributes
                            if "=" in attr and not attr.startswith("bid")
                        ]
                    }

                    # build this element node
                    node_dict = {
                        "tag_name": tag_name,
                        "bid": bid,
                        "attributes": attributes_dict,
                        "children": [],
                    }

            html = ""
            html += html_before

            # recursively print iframe nodes if any
            if node_idx in nodes["contentDocumentIndex"]["index"]:
                sub_document_idx = nodes["contentDocumentIndex"]["value"][
                    nodes["contentDocumentIndex"]["index"].index(node_idx)
                ]
                sub_html, sub_node = parse_document(document_idx=sub_document_idx)
                html += sub_html

            # recurse on children
            for child_idx in node_children[node_idx]:
                child_html, child_node = dfs(child_idx, skip_node)
                html += child_html
                # attach only if we have a container and a real child
                if node_dict is not None and child_node is not None:
                    node_dict["children"].append(child_node)

            html += html_after
            return html, node_dict or EMPTY_NODE

        html, root_node = dfs(node_idx=0, parent_node_skipped=False)
        if root_node is None:
            root_node = EMPTY_NODE

        # Format the HTML document with indentation
        soup = BeautifulSoup(html, "lxml")
        html = cast(str, soup.prettify())

        return html, root_node

    html, elements = parse_document(document_idx=0)

    return html, elements


def prune_html(html):
    html = re.sub(r"\n", " ", html)
    # remove html comments
    html = re.sub(r"<!--(.*?)-->", "", html, flags=re.MULTILINE)

    soup = BeautifulSoup(html, "lxml")
    for tag in reversed(soup.find_all()):
        # remove body and html tags (not their content)
        if tag.name in ("html", "body"):
            tag.unwrap()
        # remove useless tags
        elif tag.name in ("style", "link", "script", "br"):
            tag.decompose()
        # remove / unwrap structural tags
        elif tag.name in ("div", "span", "i", "p") and len(tag.attrs) == 1 and tag.has_attr("bid"):
            if not tag.contents:
                tag.decompose()
            else:
                tag.unwrap()

    html = soup.prettify()

    return html


def prune_dom_dict(node: DomNode, root: bool = True) -> DomNode | List[DomNode]:
    """
    Recursively prune a dict-based DOM:
    - unwrap <html>, <body>
    - decompose <style>, <link>, <script>, <br>
    - unwrap structural tags (<div>, <span>, <i>, <p>) if only 'bid' attr
    - drop nodes with no tag_name, no attrs, and empty children
    - censor nodes with tag_name starting with '::' (e.g., ::before, ::after)
    - if a kept node has children=[] after pruning, set children=None
    """
    tag = node.get("tag_name")
    attrs = node.get("attributes", {}) or {}
    children = node.get("children", []) or []
    bid = node.get("bid")
    element_text = node.get("element_text")

    # Allow text nodes to be kept
    if element_text:
        # drop empty text nodes
        if element_text.strip() == "":
            return []
        return node

    # Drop totally empty placeholders
    if not tag and not attrs and not children:
        return []

    # Censor pseudo-elements like ::before, ::after
    if tag and tag.startswith("::"):
        return []

    # Drop nodes with no bid
    if not bid and not root:
        return []

    # Helper to flatten results
    def _collect(pruned_list: List[DomNode], item: DomNode | List[DomNode] | None) -> None:
        if isinstance(item, list):
            pruned_list.extend(item)
        elif item:
            pruned_list.append(item)

    # 1) unwrap <html>, <body>
    if tag in ("html", "body"):
        pruned: List[DomNode] = []
        for c in children:
            _collect(pruned, prune_dom_dict(c, root=False))
        return pruned

    # 2) decompose these entirely
    if tag in ("style", "link", "script", "br"):
        return []

    # 3) unwrap structural tags with only 'bid' attr
    if tag in ("div", "span", "i", "p") and len(attrs) == 0:
        pruned = []
        for c in children:
            _collect(pruned, prune_dom_dict(c, root=False))
        return pruned

    # Default: keep this node, prune its children
    pruned_children: List[DomNode] = []
    for c in children:
        _collect(pruned_children, prune_dom_dict(c, root=False))

    # Build the new node (copy all keys except old children)
    new_node = cast(DomNode, {k: v for k, v in node.items() if k != "children"})
    # Attach children or None
    if pruned_children:
        new_node["children"] = pruned_children

    return new_node


def find_bids_dfs(data: DomNode | List[DomNode]) -> set[int]:
    """
    Performs a DFS to find all 'bid' values in a nested data structure and returns them as a sorted list.
    """
    bids = set()

    def dfs(node: Any) -> None:
        if isinstance(node, dict):
            if bid := node.get("bid"):
                bids.add(int(bid))
            for _, value in node.items():
                dfs(value)
        elif isinstance(node, list):
            for item in node:
                dfs(item)

    dfs(data)
    return bids
