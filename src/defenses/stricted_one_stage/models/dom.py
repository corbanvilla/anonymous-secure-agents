import logging
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, computed_field

from .enums import ElementContext, ElementPurpose
from .policy import ContentOwners, DomMetadata

logger = logging.getLogger(__name__)


class PolicyViolation(Exception):
    pass


class AnnotatedElement(BaseModel):
    bid: int = Field(..., description="Unique block identifier")
    name: Optional[str] = Field(
        None,
        description="The HTML tag name, e.g. 'meta', 'title', etc.",
        alias="tag_name",
    )
    owner: Optional[str] = Field(None, description="Who owns or authored this node")
    owner_patched: Optional[bool] = Field(False, description="Was owner assumed from parent?")

    # injected metadata
    context: Optional[ElementContext] = Field(None, description="Where this node lives or applies")
    purpose: Optional[ElementPurpose] = Field(None, description="The intention or role of this node")
    metadata_group: Optional[int] = Field(None, description="The group ID of the metadata element this node belongs to")

    attributes: List[str] | Dict[str, str] = Field(
        default_factory=dict, description="HTML attributes for this node", exclude=True
    )
    children: List["AnnotatedElement"] = Field(default_factory=list, description="Nested child nodes")


class AnnotatedDom(BaseModel):
    children: List[AnnotatedElement] = Field(..., description="Top‐level nodes in the document")

    @classmethod
    def fill_missing_bids(cls, tree: dict, parent_bid: int = -1) -> dict:
        """
        Recursively patches any node whose 'bid' is None by assigning it the parent's bid.
        Uses parent_bid=-1 for the root if its bid is missing.

        Modifies the tree in place and returns it.
        """
        # Determine this node's bid: use its own if present, else inherit parent_bid
        node_bid = tree.get("bid")
        if node_bid is None:
            tree["bid"] = parent_bid
        else:
            parent_bid = node_bid  # update for children

        # Recurse into children
        for child in tree.get("children", []):
            logger.info(f"Filling missing bid for child: {child} with parent bid: {parent_bid}")
            cls.fill_missing_bids(child, parent_bid)

        return tree

    @classmethod
    def from_a11y_tree(cls, tree: dict) -> "AnnotatedDom":
        """
        Convert an accessibility tree to an annotated DOM.
        """
        tree = cls.fill_missing_bids(tree)
        return cls.model_validate({"children": [tree]})

    @computed_field
    def all_bids(self) -> List[int]:
        """
        Returns a list of all unique BIDs in the DOM.
        """
        all_bids = set()

        def _recurse(node: AnnotatedElement):
            all_bids.add(node.bid)
            for child in node.children:
                _recurse(child)

        for child in self.children:
            _recurse(child)

        return sorted(all_bids)

    def export_html_censored(self) -> str:
        from bs4 import BeautifulSoup

        def _node_to_html(n):
            if not isinstance(n, dict):
                return ""
            tag = n.get("tag_name") or "div"
            attrs = []
            for k in ("bid", "tag_name", "owner", "context", "purpose"):
                v = n.get(k)
                if v is not None:
                    attrs.append(f'{k}="{str(v)}"')
            attr_str = " " + " ".join(attrs) if attrs else ""
            children = n.get("children", [])
            if children:
                inner = "".join(_node_to_html(child) for child in children)
                return f"<{tag}{attr_str}>{inner}</{tag}>"
            else:
                return f"<{tag}{attr_str} />"

        data = self.export_dict(exclude=[])
        html = _node_to_html(data)
        html_pretty = BeautifulSoup(html, "html.parser").prettify()
        return str(html_pretty)

    def export_dict(self, exclude: List[str]) -> dict:
        def _recursive_remove(obj):
            if isinstance(obj, dict):
                for field in exclude:
                    obj.pop(field, None)
                for v in obj.values():
                    _recursive_remove(v)
            elif isinstance(obj, list):
                for item in obj:
                    _recursive_remove(item)

        data = self.model_dump(exclude_defaults=True)
        _recursive_remove(data)
        return data

    def apply_owners(self, owners: ContentOwners) -> list:
        """
        Recursively label each node's 'owner' field based on ContentOwners.
        If a node's bid is missing, inherit owner from nearest parent and log.

        Returns a list of missing bids, which were patched in by parents.
        """

        missing_bids: list[int] = []

        def get_owner(bid: int) -> Optional[str]:
            if bid in owners.developer:
                return "developer"
            elif bid in owners.user:
                return "user"
            elif bid in owners.third_party:
                return "third_party"
            return None

        # TODO - a better way to fill unknown nodes is to either requery, or assume the worst (lowest policy)
        def recurse(nodes: List[AnnotatedElement], parent_owner: Optional[str] = None):
            for node in nodes:
                node.owner = get_owner(node.bid)
                if not node.owner:
                    node.owner = parent_owner
                    node.owner_patched = True
                    logger.warning(f"Missing {node.bid}, inheriting from parent: {parent_owner}")
                if node.children:
                    recurse(node.children, node.owner)

        recurse(self.children)

        # TODO - apply cascading policy

        return missing_bids

    def apply_metadata(self, metadata: DomMetadata) -> list:
        """
        Recursively apply context and purpose from DomMetadata to nodes by bid.
        Returns a list of bids that were defined in multiple metadata elements.
        """
        bid_to_group = {}
        bid_to_meta = {}
        multiple_definitions = []

        # Create a mapping of bid to metadata element
        for group_id, element in enumerate(metadata.elements):
            for bid in element.bids:
                if bid in bid_to_meta:
                    logger.warning(f"Bid {bid} defined in multiple metadata elements; overwriting previous value.")
                    multiple_definitions.append(bid)
                bid_to_meta[bid] = element
                bid_to_group[bid] = group_id

        def recurse(nodes: List[AnnotatedElement]):
            for node in nodes:
                meta = bid_to_meta.get(node.bid)
                if meta:
                    node.context = meta.context
                    node.purpose = meta.purpose
                    node.metadata_group = bid_to_group.get(node.bid)
                else:
                    # No filling in for missing bid metadata for now
                    logger.warning("No metadata found for bid: %s", node.bid)
                if node.children:
                    recurse(node.children)

        recurse(self.children)

        # TODO - apply cascading policy

        return multiple_definitions

    def filter_by_owners(self, allowed: List[str]) -> List[int]:
        """Remove nodes whose ``owner`` is not in ``allowed``."""

        removed: List[int] = []

        def recurse(nodes: List[AnnotatedElement]) -> List[AnnotatedElement]:
            filtered: List[AnnotatedElement] = []
            for node in nodes:
                if node.owner in allowed:
                    if node.children:
                        node.children = recurse(node.children)
                    filtered.append(node)
                else:
                    removed.append(node.bid)
            return filtered

        self.children = recurse(self.children)
        return removed

    def enforce_content_ownership_flow(self) -> List[str]:
        """
        Recursively enforces no elevation of privelege:
        Forbidden flows: user -> developer, third_party -> developer.
        Returns a lift of violation messages or raises PolicyViolation.
        """

        violations = []

        def recurse(node: AnnotatedElement, parent_owner: Optional[str] = None):
            for child in node.children:
                p_owner = node.owner or parent_owner
                c_owner = child.owner
                if p_owner in ("user", "third_party") and c_owner == "developer":
                    violations.append(f"Forbidden flow: {p_owner} -> {c_owner} at bid {child.bid}")

                recurse(child, p_owner)

        for root in self.children:
            recurse(root, None)

        if violations:
            raise PolicyViolation("\n".join(violations))

        return violations

    def discover_bid_path(self, target_bid: int) -> List[int]:
        """
        Returns the list of bids from the root through each parent to the node
        matching target_bid (inclusive). Performs a depth-first search (DFS).
        If the target_bid is not found, returns an empty list.
        """

        def dfs(node: AnnotatedElement, path: List[int]) -> Optional[List[int]]:
            current_path = path + [node.bid]
            if node.bid == target_bid:
                return current_path
            for child in node.children:
                result = dfs(child, current_path)
                if result:
                    return result
            return None

        # Start DFS from each top-level child
        for child in self.children:
            result = dfs(child, [])
            if result:
                return result
        return []

    def required_parent_bids(self, target_bids: List[int]) -> List[int]:
        """
        Returns the list of bids from the root through each parent to the node
        matching target_bid (inclusive). Performs a depth-first search (DFS).
        If the target_bid is not found, returns an empty list.
        """

        required = set()
        for bid in target_bids:
            required.update(self.discover_bid_path(bid))
        return list(required)


# Self‐referencing model
AnnotatedElement.model_rebuild()
