# copied in from agentoccam
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union, cast

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


class A11yNode(TypedDict, total=False):
    """Typed representation of a node in an accessibility tree."""

    role: str
    name: Optional[str]
    value: Any
    bid: Optional[int]
    attributes: List[str]
    children: List["A11yNode"]


def flatten_dom_to_str_censored(
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
    censor_bids: Optional[List[int]] = None,
) -> str:
    """Formats a DOM snapshot into a string text"""

    extra_properties = extra_properties or {}

    if censor_bids is None:
        censor_bids = []

    def to_string(idx):
        if idx == -1:
            return None
        else:
            return dom_snapshot["strings"][idx]

    def parse_document(document_idx) -> str:
        # adapted from [natbot](https://github.com/nat/natbot)

        nodes = dom_snapshot["documents"][document_idx]["nodes"]
        node_children = defaultdict(lambda: [])

        for node_idx in range(len(nodes["nodeName"])):
            parent_idx = nodes["parentIndex"][node_idx]
            if parent_idx != -1:
                node_children[parent_idx].append(node_idx)

        def dfs(node_idx: int, parent_node_skipped: bool) -> str:
            # https://developer.mozilla.org/en-US/docs/Web/API/Node/nodeType
            # https://developer.mozilla.org/en-US/docs/Web/API/Node/nodeName
            # https://developer.mozilla.org/en-US/docs/Web/API/Node/nodeValue

            node_type = nodes["nodeType"][node_idx]
            node_name = to_string(nodes["nodeName"][node_idx])
            node_value = to_string(nodes["nodeValue"][node_idx])
            html_before = ""
            html_after = ""
            skip_node = False

            # text nodes: print text content only if parent was not skipped
            if node_type == 3:  # node_name == "#text"
                if not parent_node_skipped and node_value is not None:
                    html_before += node_value

            # CData nodes: print content only if parent was not skipped
            elif node_type == 4:  # node_name == "#cdata-section":
                if not parent_node_skipped and node_value is not None:
                    html_before += f"<!CDATA[[{node_value}]]>"

            # processing instructions, comments, documents, doctypes, document fragments: don't print
            elif node_type in (7, 8, 9, 10, 11):
                skip_node = True

            # skip pseudo-elements
            elif node_name.startswith("::before") or node_name.startswith("::after"):
                skip_node = True

            # now we should have an element node
            else:
                assert node_type == 1

                tag_name = node_name.lower().strip()
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

                if bid and (int(bid) in censor_bids):
                    skip_node = True

                if not skip_node:
                    # print node opening tag, with its attributes
                    html_before += f"<{tag_name}" + " ".join([""] + attributes) + ">"
                    # print node closing tag
                    html_after += f"</{tag_name}>"

            html = ""
            html += html_before

            # recursively print iframe nodes if any
            if node_idx in nodes["contentDocumentIndex"]["index"]:
                sub_document_idx = nodes["contentDocumentIndex"]["value"][
                    nodes["contentDocumentIndex"]["index"].index(node_idx)
                ]
                html += parse_document(document_idx=sub_document_idx)

            # recursively print children nodes if any
            for child_idx in node_children[node_idx]:
                html += dfs(node_idx=child_idx, parent_node_skipped=skip_node)

            html += html_after

            return html

        html = dfs(node_idx=0, parent_node_skipped=False)

        # Format the HTML document with indentation
        soup = BeautifulSoup(html, "lxml")
        html = cast(str, soup.prettify())

        return html

    html = parse_document(document_idx=0)

    return html


def flatten_axtree_to_str_censored(
    AX_tree: Dict[str, Any],
    censor_bids: Optional[List[int]] = None,
    extra_properties: Optional[Dict[str, Any]] = None,
    with_visible: bool = False,
    with_clickable: bool = False,
    with_center_coords: bool = False,
    with_bounding_box_coords: bool = False,
    with_som: bool = False,
    skip_generic: bool = True,
    filter_visible_only: bool = False,
    filter_with_bid_only: bool = False,
    filter_som_only: bool = False,
    coord_decimals: int = 0,
    remove_redundant_static_text: bool = True,
    hide_bid_if_invisible: bool = False,
    hide_all_children: bool = False,
    hide_all_bids: bool = False,
    bid_capabilities: Optional[Dict[int, str]] = None,
):
    if censor_bids is None:
        censor_bids = []
    # ensure dict for type checkers and safe .get usage
    extra_properties = extra_properties or {}
    node_id_to_idx = {node["nodeId"]: idx for idx, node in enumerate(AX_tree["nodes"])}

    def dfs(node_idx, depth, parent_filtered, parent_name):
        # stop recursion if parent was filtered
        if parent_filtered:
            return ""
        tree_str = ""
        node = AX_tree["nodes"][node_idx]
        indent = "\t" * depth
        skip_node = False
        filter_node = False

        role = node["role"]["value"]
        if role in IGNORED_AXTREE_ROLES or "name" not in node:
            skip_node = True
        else:
            name = node["name"]["value"]
            value = node.get("value", {}).get("value", None)
            bid = node.get("browsergym_id", None)

            # generic-role skipping
            props = node.get("properties", [])
            attributes = []
            for p in props:
                if "value" not in p or "value" not in p["value"]:
                    continue
                pn = p["name"]
                pv = p["value"]["value"]
                if pn in IGNORED_AXTREE_PROPERTIES:
                    continue
                elif pn in ("required", "focused", "atomic"):
                    if pv:
                        attributes.append(pn)
                else:
                    attributes.append(f"{pn}={repr(pv)}")
            if skip_generic and role == "generic" and not attributes:
                skip_node = True
            if hide_all_children and parent_filtered:
                skip_node = True

            # StaticText-specific skip
            if role == "StaticText":
                if parent_filtered or (remove_redundant_static_text and name in parent_name):
                    skip_node = True
            else:
                # apply user filters & censor flag
                filter_node, extra_attrs = _process_bid(
                    bid,
                    extra_properties,
                    with_visible,
                    with_clickable,
                    with_center_coords,
                    with_bounding_box_coords,
                    with_som,
                    filter_visible_only,
                    filter_with_bid_only,
                    filter_som_only,
                    coord_decimals,
                )
                if bid and int(bid) in censor_bids:
                    skip_node = True
                    filter_node = True
                skip_node = skip_node or filter_node
                attributes = extra_attrs + attributes

            # quick-fix: stop here if this node is filtered
            if filter_node:
                return ""

            # build output line
            if not skip_node:
                line = role if (role == "generic" and not name) else f"{role} {repr(name.strip())}"
                if not (
                    hide_all_bids
                    or bid is None
                    or (hide_bid_if_invisible and extra_properties.get(bid, {}).get("visibility", 0) < 0.5)
                ):
                    prefix = None
                    if bid_capabilities is not None:
                        lookup_bid = None
                        if isinstance(bid, int):
                            lookup_bid = bid
                        else:
                            try:
                                lookup_bid = int(bid)  # type: ignore[arg-type]
                            except (TypeError, ValueError):
                                lookup_bid = None
                        if lookup_bid is not None and lookup_bid in bid_capabilities:
                            cap_raw = bid_capabilities[lookup_bid]
                            cap_norm = str(cap_raw).lower()
                            if cap_norm not in ("interactable", "viewable"):
                                raise ValueError(
                                    f"Invalid capability '{cap_raw}' for bid {lookup_bid}. Expected one of: interactable, viewable."
                                )
                            if cap_norm == "viewable":
                                prefix = "[_]"
                            # elif cap_norm == "clickable":
                            # prefix = f"[{bid}*]"
                            else:  # interactable
                                prefix = f"[{bid}]"
                    if prefix is None:
                        prefix = f"[{bid}]"
                    line = f"{prefix} " + line
                if value is not None:
                    line += f" value={repr(value)}"
                if attributes:
                    line += ", ".join([""] + attributes)
                tree_str += f"{indent}{line}"

        # recurse
        for cid in node.get("childIds", []):
            if cid not in node_id_to_idx or cid == node["nodeId"]:
                continue
            next_depth = depth if skip_node else depth + 1
            sub = dfs(
                node_id_to_idx[cid],
                next_depth,
                filter_node,
                node.get("name", {}).get("value", ""),
            )
            if sub:
                tree_str += ("\n" if tree_str else "") + sub
        return tree_str

    return dfs(0, 0, False, "")


def flatten_axtree_to_str_and_dict(
    AX_tree: Dict[str, Any],
    extra_properties: Optional[Dict[str, Any]] = None,
    with_visible: bool = False,
    with_clickable: bool = False,
    with_center_coords: bool = False,
    with_bounding_box_coords: bool = False,
    with_som: bool = False,
    skip_generic: bool = True,
    filter_visible_only: bool = False,
    filter_with_bid_only: bool = False,
    filter_som_only: bool = False,
    coord_decimals: int = 0,
    ignored_roles=frozenset(IGNORED_AXTREE_ROLES),
    ignored_properties=frozenset(IGNORED_AXTREE_PROPERTIES),
    remove_redundant_static_text: bool = True,
    hide_bid_if_invisible: bool = False,
    hide_all_children: bool = False,
    hide_all_bids: bool = False,
) -> Tuple[str, A11yNode]:
    extra_properties = extra_properties or {}
    node_idx = {n["nodeId"]: i for i, n in enumerate(AX_tree["nodes"])}

    def _flatten_list(lst: List[Any]) -> List[A11yNode]:
        """Recursively flatten any nested lists into a single list of dicts."""
        out: List[A11yNode] = []
        for item in lst:
            if isinstance(item, list):
                out.extend(_flatten_list(item))
            else:
                out.append(item)
        return out

    def dfs(
        i: int, depth: int, parent_filtered: bool, parent_name: str
    ) -> Tuple[str, Optional[Union[A11yNode, List[A11yNode]]]]:
        n = AX_tree["nodes"][i]
        role = n["role"]["value"]
        name = n.get("name", {}).get("value", "").strip()
        bid = n.get("browsergym_id")
        val = n.get("value", {}).get("value", None)

        skip_node = False
        filter_node = False

        # collect native properties
        attrs: List[str] = []
        for p in n.get("properties", []):
            if "value" in p and "value" in p["value"]:
                v = p["value"]["value"]
                k = p["name"]
                if k in ignored_properties:
                    continue
                if k in ("required", "focused", "atomic"):
                    if v:
                        attrs.append(k)
                else:
                    attrs.append(f"{k}={repr(v)}")

        # skip logic
        if skip_generic and role == "generic" and not attrs:
            skip_node = True
        if role in ignored_roles or "name" not in n:
            skip_node = True
        if hide_all_children and parent_filtered:
            skip_node = True
        if role == "StaticText":
            if parent_filtered or (remove_redundant_static_text and name and name in parent_name):
                skip_node = True
        else:
            filter_node, extra = _process_bid(
                bid,
                extra_properties,
                with_visible,
                with_clickable,
                with_center_coords,
                with_bounding_box_coords,
                with_som,
                filter_visible_only,
                filter_with_bid_only,
                filter_som_only,
                coord_decimals,
            )
            attrs = extra + attrs
            skip_node = skip_node or filter_node

        # build string line
        indent = "\t" * depth
        line = ""
        if not skip_node:
            lbl = f"{role} {repr(name)}" if name or role != "generic" else role
            if (
                bid
                and not hide_all_bids
                and not (hide_bid_if_invisible and extra_properties.get(bid, {}).get("visibility", 0) < 0.5)
            ):
                lbl = f"[{bid}] " + lbl
            if val is not None:
                lbl += f" value={repr(val)}"
            if attrs:
                lbl += ", " + ", ".join(attrs)
            line = indent + lbl

        # recurse children
        lines: List[str] = []
        dicts_raw: List[Union[A11yNode, List[A11yNode]]] = []
        for cid in n.get("childIds", []):
            if cid == n["nodeId"] or cid not in node_idx:
                continue
            child_str, child_dict = dfs(node_idx[cid], depth if skip_node else depth + 1, filter_node, name)
            if child_str:
                lines.append(child_str)
            if child_dict is not None:
                dicts_raw.append(child_dict)

        # flatten any nested lists of children here
        dicts: List[A11yNode] = _flatten_list(dicts_raw)

        subtree_str = "\n".join(filter(None, [line] + lines)).rstrip()

        if skip_node:
            # bubble up only the flattened children
            return subtree_str, (dicts or None)

        # non-skipped: attach only flat list of children
        return subtree_str, {
            "role": role,
            "name": name or None,
            "value": val,
            "bid": bid,
            "attributes": attrs,
            "children": dicts,
        }

    raw_str, raw_tree = dfs(0, 0, False, "")
    tree_str = raw_str.strip()

    # if root was skipped, flatten & wrap
    if isinstance(raw_tree, list):
        flat = _flatten_list(raw_tree)
        tree_dict: A11yNode = {
            "role": "Root",
            "name": None,
            "value": None,
            "bid": -1,
            "attributes": [],
            "children": flat,
        }
    elif raw_tree is None:
        tree_dict = {
            "role": "Root",
            "name": None,
            "value": None,
            "bid": -1,
            "attributes": [],
            "children": [],
        }
    else:
        tree_dict = cast(A11yNode, raw_tree)

    return tree_str, tree_dict
