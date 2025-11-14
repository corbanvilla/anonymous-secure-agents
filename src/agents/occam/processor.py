from copy import deepcopy
from typing import Any, Dict, List, Tuple

IGNORED_ACTREE_PROPERTIES = (
    "focusable",
    "editable",
    "readonly",
    "level",
    "settable",
    "multiline",
    "invalid",
)


class TreeNode:
    def __init__(self, node_id: str, role: str, name: str, depth: int, **kwargs: Any) -> None:
        self.visible = True
        self.node_id = node_id
        self.role = role
        self.name = name
        self.depth = depth
        self.properties: Dict[str, Any] | None = kwargs.get("properties")
        self.children: List["TreeNode"] = []
        self.parent: "TreeNode | None" = None

    def add_child(self, child: "TreeNode") -> None:
        child.parent = self
        self.children.append(child)

    def copy(self) -> "TreeNode":
        """Return a shallow copy of the current node."""
        # deepcopying the entire tree is extremely expensive since each node
        # references its parent and children. We only need a shallow copy of the
        # current node when constructing a new DOM, so manually copy the
        # relevant attributes instead of recursively copying the whole tree.
        new_self = TreeNode(
            node_id=self.node_id,
            role=self.role,
            name=self.name,
            depth=self.depth,
            properties=deepcopy(self.properties) if self.properties else None,
        )
        new_self.visible = self.visible

        return new_self

    def visible_children(self) -> List["TreeNode"]:
        return [c for c in self.children if c.visible]

    def siblings(self) -> List["TreeNode"]:
        if not self.parent:
            return []
        return [n for n in self.parent.children if n.node_id != self.node_id]

    def visible_siblings(self) -> List["TreeNode"]:
        if not self.parent:
            return []
        return [n for n in self.parent.children if n.visible and n.node_id != self.node_id]

    def has_properties(self) -> bool:
        return bool(getattr(self, "properties", {}))

    def all_children_invisible(self) -> bool:
        return not any(child.visible for child in self.children)

    def search_node_by_id(self, target_id: str) -> "TreeNode | None":
        if self.node_id == target_id or (self.name and f"[{target_id}]" in self.name):
            return self
        for child in self.children:
            result = child.search_node_by_id(target_id)
            if result:
                return result
        return None

    def has_the_same_properties_as(self, another_node: "TreeNode") -> bool:
        node_a_has_properties = getattr(self, "properties", "")
        node_b_has_properties = getattr(another_node, "properties", "")
        if not node_a_has_properties and not node_b_has_properties:
            return True
        elif (node_a_has_properties and not node_b_has_properties) or (
            not node_a_has_properties and node_b_has_properties
        ):
            return False
        else:
            return self.properties == another_node.properties

    def is_identical_to(self, another_node: "TreeNode") -> bool:
        if another_node.children:
            return False
        return (
            self.role == another_node.role
            and self.name == another_node.name
            and self.has_the_same_properties_as(another_node=another_node)
        )

    def has_identical_siblings(self) -> bool:
        if not (self.parent and self.all_children_invisible()):
            return False
        return any(
            sibling.role == self.role and sibling.name == self.name
            for sibling in self.parent.children
            if sibling.node_id != self.node_id and sibling.all_children_invisible()
        )

    def has_identical_surrounding_siblings(self) -> bool:
        if not self.parent:
            return False
        siblings = self.parent.children
        idx = siblings.index(self)
        if idx > 0 and self.is_identical_to(siblings[idx - 1]):
            return True
        if idx + 1 < len(siblings) and self.is_identical_to(siblings[idx + 1]):
            return True
        return False

    def is_differentiable(self, strict: bool = False) -> bool:
        if self.parent and self.parent.role == "row":
            return True
        if not strict and self.has_identical_siblings():
            return False
        if self.has_identical_surrounding_siblings():
            return False
        return True


class TextObervationProcessor:
    @staticmethod
    def parse_accessibility_tree(accessibility_tree: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any], TreeNode]:
        """Parse the accessibility tree into a string text and structured node"""
        node_id_to_idx: Dict[str, int] = {}
        node_id_to_bid: Dict[str, str] = {}
        for idx, node in enumerate(accessibility_tree):
            node_id = str(node["nodeId"])
            node_id_to_idx[node_id] = idx
            node_id_to_bid[node_id] = str(node.get("browsergym_id", node_id))

        obs_nodes_info: Dict[str, Any] = {}

        def dfs(idx: int, obs_node_id: str, depth: int, active_node_dict: Dict[int, TreeNode | None]):
            tree_str = ""
            node = accessibility_tree[idx]
            indent = "\t" * depth
            valid_node = True
            try:
                role = node["role"]["value"]
                name = node["name"]["value"]
                node_str = f"[{obs_node_id}] {role} {repr(name)}"
                properties: List[str] = []
                structured_properties: Dict[str, Any] = {}
                for prop in node.get("properties", []):
                    try:
                        if prop["name"] in IGNORED_ACTREE_PROPERTIES:
                            continue
                        properties.append(f"{prop['name']}: {prop['value']['value']}")
                        structured_properties[prop["name"]] = prop["value"]["value"]
                    except KeyError:
                        pass

                if properties:
                    node_str += " " + " ".join(properties)

                if not node_str.strip():
                    valid_node = False

                if not name.strip():
                    if not properties and role in [
                        "generic",
                        "img",
                        "list",
                        "strong",
                        "paragraph",
                        "banner",
                        "navigation",
                        "Section",
                        "LabelText",
                        "Legend",
                        "listitem",
                    ]:
                        valid_node = False
                    elif role in ["listitem"]:
                        valid_node = False

                if valid_node:
                    tree_str += f"{indent}{node_str}"
                    obs_nodes_info[obs_node_id] = {
                        "backend_id": node.get("backendDOMNodeId"),
                        "union_bound": node.get("union_bound"),
                        "text": node_str,
                    }

            except Exception:
                valid_node = False

            structured_node = (
                TreeNode(
                    node_id=obs_node_id,
                    role=node["role"]["value"],
                    name=node["name"]["value"],
                    depth=depth,
                    properties=structured_properties,
                )
                if valid_node
                else None
            )
            active_node_dict[depth] = structured_node if valid_node else active_node_dict.get(depth)

            for child_node_id in node["childIds"]:
                child_id = str(child_node_id)
                if child_id not in node_id_to_idx:
                    continue
                child_depth = depth + 1 if valid_node else depth
                child_bid = node_id_to_bid[child_id]
                child_str, child_node = dfs(node_id_to_idx[child_id], child_bid, child_depth, active_node_dict)
                if child_str.strip():
                    if tree_str.strip():
                        tree_str += "\n"
                    tree_str += child_str
                if child_depth > 0 and child_node:
                    parent = active_node_dict.get(child_depth - 1)
                    if parent:
                        parent.add_child(child_node)

            return tree_str, structured_node

        root_bid = node_id_to_bid[str(accessibility_tree[0]["nodeId"])]
        tree_str, structured_node = dfs(0, root_bid, 0, active_node_dict={})
        return tree_str, obs_nodes_info, structured_node
