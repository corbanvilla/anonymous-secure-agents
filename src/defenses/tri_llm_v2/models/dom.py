import asyncio
import logging
import threading
import time
from typing import Any, Dict, List, Literal, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, computed_field

from src.api_clients.async_logging import AsyncOpenAILoggingProxy
from src.environments.observations.dom_parser import flatten_dom_to_str_get_dict, prune_dom_dict
from src.environments.observations.reconstruction import flatten_axtree_to_str_and_dict

from .enums import CapabilityLabel, IntegrityLabel
from .policy import SecurityPolicy

logger = logging.getLogger(__name__)


# TODO - begin testing additional instruction from parser.py

PURPOSES = Literal[
    "site structure",
    "content",
    "product title",
    "navigation",
    "data table",
    "metadata",
    "search",
    "time display",
    "project information",
    "user link",
    "project name",
    "vote form",
    "commit details",
    "heading",
    "file name",
    "advertisement",
]

CONTEXTS = Literal[
    "site navigation",
    "sidebar navigation",
    "post content",
    "submission header",
    "search form",
    "project content",
    "project name",
    "order table",
    "comment",
    "user navigation",
    "file tree",
    "issues list",
    "product list",
    "external source",
]

METADATA_INSTRUCTION_PROMPT = """
Your task is to parse the HTML and categorize the LAST element in the DOM according to the following rules:

Owner labels:
 - developer: Content and elements created by the site's developers (core site functionality, layout, branding)
 - user: User-generated content (comments, posts, uploaded media, user profiles, usernames, etc)
 - third_party: External content (advertisements, social media embeds, analytics scripts, third-party widgets, products, product descriptions, etc)

Context and purpose labels:
 - context: summarize, in 2-3 words, in what context and where the element exists (e.g., product page, post, comment, navigation bar)
 - purpose: summarize, in 2-3 words, what role the element serves (e.g., site structure, navigation, content, etc.)
"""

CAP_INSTRUCTION_PROMPT = """
Your task is to consider the user's task and the provided abstracted HTML, then determine what capabilities the LAST element shown should have.

Consider the principle of least privilege: the element should have the minimum capabilities necessary to complete the user's described task. However, do be careful not to be too restrictive.

Capability labels:
 - viewable: The element can only be viewed by the user (read-only)
 - interactable: The element can be interacted with by the user (read-write)

For instance, if the user's task is to research a topic on Reddit, user content may be viewable, the usernames of authors should probably not be clickable. 

However, if the user needs to write a comment on a post, or interact with a product review, it makes sense to make the element comment box interactable. 

Third party elements, such as advertisements, should be viewable, but not interactable.
"""


class AnnotatedElementCapabilities(BaseModel):
    capability: CapabilityLabel = Field(..., description="Capability label for this node")


class AnnotatedElementMetadata(BaseModel):
    # context: ElementContext = Field(..., description="Where this node lives or applies")
    # purpose: ElementPurpose = Field(..., description="The intention or role of this node")
    # context: str = Field(..., description="Where this node lives or applies")
    # purpose: str = Field(..., description="The intention or role of this node")
    context: CONTEXTS = Field(..., description="Where this node lives or applies")
    purpose: PURPOSES = Field(..., description="The intention or role of this node")
    owner: IntegrityLabel = Field(..., description="Who owns or authored this node")


class AnnotatedElement(BaseModel):
    id: int = Field(description="Unique block identifier")
    bid: int = Field(description="BrowserGym identifier")
    name: Optional[str] = Field(
        None,
        description="The HTML tag name, e.g. 'meta', 'title', etc.",
        alias="tag_name",
    )

    # Metadata fields
    metadata: Optional[AnnotatedElementMetadata] = Field(None, description="Metadata for this node")
    skip_metadata: bool = Field(
        False, description="Whether to skip metadata for this node (e.g., if it is not allowed)"
    )
    capabilities: Optional[CapabilityLabel] = Field(None, description="Relevance label for this node")

    children: List["AnnotatedElement"] = Field(default_factory=list, description="Nested child nodes")

    # Relationships
    parent: Optional["AnnotatedElement"] = Field(
        default=None,
        description="Reference to parent element",
        exclude=True,  # Exclude from serialization to avoid circular references
    )

    # Debugging
    html_attributes: List[str] | Dict[str, str] = Field(
        default_factory=dict, description="HTML attributes for this node", exclude=True, alias="attributes"
    )
    element_text: Optional[str] = Field(None, description="Text content of this node")
    visible_bids: Optional[List[int]] = Field(None, description="AXTree bids")

    src_axtree: Optional[dict] = Field(None, description="AXTree", exclude=True)
    src_dom: Optional[dict] = Field(None, description="DOM", exclude=True)

    @computed_field
    def is_root(self) -> bool:
        """Whether this element is the root of the tree (has no parent)."""
        return self.parent is None

    @property
    def root(self) -> "AnnotatedElement":
        """The root element of the tree."""
        current = self
        while current.parent is not None:
            current = current.parent
        return current

    @computed_field
    def capability_map(self) -> Dict[int, Literal["viewable", "interactable"]]:
        """
        Returns a map of element IDs to their capabilities.
        """
        if self.visible_bids is None:
            return {}

        return {
            element.bid: element.capabilities.value
            for element in self.all_elements
            if element.capabilities is not None and not element.inherited_bid and element.bid in self.visible_bids
        }

    @property
    def __are_ancestors_metadata_resolved(self) -> bool:
        """Whether all ancestors of this element have been resolved."""
        # Recursively check all ancestors
        current = self.parent
        while current is not None:
            # Skipped metadata elements are considered resolved
            if current.skip_metadata:
                return True
            if current.metadata is None:
                return False
            if current.capabilities is None:
                return False
            current = current.parent

        return True

    @property
    def all_bids(self) -> List[int]:
        """
        Returns a list of all unique BIDs in the DOM.
        """
        all_bids = set()

        def _recurse(node: "AnnotatedElement"):
            if node.bid is not None:
                all_bids.add(node.bid)
            for child in node.children:
                _recurse(child)

        _recurse(self)
        return sorted(all_bids)

    @property
    def all_elements(self) -> List["AnnotatedElement"]:
        """
        Returns a list of all elements in the DOM tree.
        """
        elements = []

        def _recurse(node: "AnnotatedElement"):
            elements.append(node)
            for child in node.children:
                _recurse(child)

        _recurse(self)
        return elements

    @property
    def inherited_bid(self) -> bool:
        """
        Returns if the bid was inherited from a parent element.
        """
        if self.parent and self.bid == self.parent.bid:
            return True
        return False

    @property
    def direct_lineage_tree(self) -> "AnnotatedElement":
        """
        Creates a new tree containing this element and all its ancestors up to the root.
        Each node in the new tree is a copy of the original, with proper parent-child relationships.
        """
        # Create copy of current node as the leaf of new tree
        new_leaf = self.model_copy(update={"children": []})
        new_current = new_leaf

        # Walk up the parent chain all the way to root
        current = self.parent
        while current is not None:
            # Create copy of parent without its original children
            parent_copy = current.model_copy(update={"children": []})

            # Link the new nodes
            parent_copy.children = [new_current]
            new_current.parent = parent_copy

            # Move up the chain
            new_current = parent_copy
            current = current.parent

        return new_current.root

    @property
    def as_direct_lineage_html_tree_top(self) -> str:
        """
        Returns the DOM tree as a string of HTML.
        """
        html = ""
        element = self.direct_lineage_tree
        indent = ""
        while element is not None:
            metadata_str = ""
            # if element.metadata is not None:
            #     metadata_str = f' [owner="{element.metadata.owner.value}" context="{element.metadata.context}" purpose="{element.metadata.purpose}"]'
            if not element.metadata:
                metadata_str = ' [owner="???" context="???" purpose="???"]'
            if isinstance(element.html_attributes, dict):
                attrs_str = " ".join([f'{k}="{v}"' for k, v in element.html_attributes.items()])
            elif isinstance(element.html_attributes, list):
                attrs_str = " ".join(element.html_attributes)
            if len(attrs_str) > 0:
                attrs_str = f" {attrs_str}"
            html += f"{indent}<{element.name or ('html' if element.is_root else 'div')}{metadata_str}{attrs_str}>\n"
            indent += " " * 2
            element = element.children[0] if element.children else None

            # Check if next element is a text node, then show it (and break)
            if element and element.element_text:
                html += f"{indent}{element.element_text}\n"
                break

        return html

    @property
    def as_html_tree_top_full(self) -> str:
        """
        Enumerates the DOM tree as a string of HTML, including metadata
        """

        def _format_bracket(e: "AnnotatedElement") -> str:
            parts: list[str] = []
            if e.bid is not None:
                parts.append(f"bid={e.bid}")
            if e.metadata is not None:
                parts.append(f'owner="{e.metadata.owner.value}"')
                parts.append(f'context="{e.metadata.context}"')
                parts.append(f'purpose="{e.metadata.purpose}"')
            parts.append(f"capability={e.capabilities.value if e.capabilities else 'N/A'}")
            return f" [{' '.join(parts)}]" if parts else ""

        def _attrs_to_str(attrs: List[str] | Dict[str, str]) -> str:
            def _sanitize(text: str) -> str:
                # replace newlines with spaces, trim, then collapse multiple spaces
                text = text.replace("\n", " ").strip()
                return " ".join(text.split())

            if isinstance(attrs, dict):
                parts: list[str] = []
                for k, v in attrs.items():
                    sk = _sanitize(str(k))
                    sv = _sanitize(str(v))
                    parts.append(f'{sk}="{sv}"')
                s = " ".join(parts)
            else:
                s = " ".join(attrs)
            return f" {s}" if len(s) > 0 else ""

        lines: list[str] = []

        def _dfs(e: Optional["AnnotatedElement"], indent: str) -> None:
            if e is None:
                return
            if e.element_text:
                lines.append(f"{indent}{e.element_text}")
                return
            tag = e.name or ("html" if e.is_root else "div")
            bracket = _format_bracket(e)
            attrs_str = _attrs_to_str(e.html_attributes)
            lines.append(f"{indent}<{tag}{bracket}{attrs_str}>")
            for child in e.children:
                _dfs(child, indent + " " * 2)
            lines.append(f"{indent}</{tag}>")

        _dfs(self.root, "")
        return ("\n".join(lines) + "\n") if lines else ""

    @property
    def as_direct_lineage_relevance_tree_top(self) -> str:
        """
        Returns the DOM tree as a string of HTML, censored for capabilities only.
        """
        html = ""
        element = self.direct_lineage_tree
        indent = ""
        assert element.metadata is not None

        attributes = {
            "owner": element.metadata.owner.value,
            "context": element.metadata.context,
            "purpose": element.metadata.purpose,
        }
        while element is not None:
            attrs_str = " ".join([f'{k}="{v}"' for k, v in attributes.items()])
            html += f"{indent}<{element.name or ('html' if element.is_root else 'div')} {attrs_str}>\n"
            indent += " " * 2
            element = element.children[0] if element.children else None

        return html

    def __set_parent(self, parent: Optional["AnnotatedElement"] = None) -> None:
        """
        Sets the parent reference for this node and recursively for all children.

        Args:
            parent: The parent element to set, or None to clear the parent reference
        """
        self.parent = parent
        for child in self.children:
            child.__set_parent(self)

    def __add_child(self, child: "AnnotatedElement") -> None:
        """
        Adds a child element and sets up its parent reference properly.

        Args:
            child: The child element to add
        """
        self.children.append(child)
        child.__set_parent(self)

    @staticmethod
    def __fill_missing_bids(tree: Dict[str, Any], parent_bid: int = -1) -> Dict[str, Any]:
        """
        Recursively patches any node whose 'bid' is None by assigning it the parent's bid.
        Uses parent_bid=-1 for the root if its bid is missing.

        Modifies the tree in place and returns it.
        """
        # Determine this node's bid: use its own if present, else inherit parent_bid
        node_bid = tree.get("bid")
        if node_bid is None:
            tree["bid"] = str(parent_bid)  # Store as string for compatibility
        else:
            parent_bid = int(node_bid) if isinstance(node_bid, str) else node_bid  # update for children

        # Recurse into children
        for child in tree.get("children", []):
            AnnotatedElement.__fill_missing_bids(child, parent_bid)

        return tree

    @staticmethod
    def __set_ids(node: Dict[str, Any], counter: Optional[List[int]] = None) -> None:
        """
        Recursively sets auto-incrementing IDs for each element in the DOM tree.
        Uses a list for the counter to maintain state across recursive calls.
        """
        if counter is None:
            counter = [1]  # Start from 1

        # Add id as a custom field that will be used by AnnotatedElement
        node["id"] = counter[0]
        counter[0] += 1

        for child in node.get("children", []):
            AnnotatedElement.__set_ids(child, counter)

    @staticmethod
    def __build_tree(node_dict: Dict[str, Any]) -> "AnnotatedElement":
        """
        Recursively builds an AnnotatedElement tree from a dictionary,
        properly setting up parent references.
        """
        # Create a shallow copy of the dict and remove children to avoid recursion issues
        node_data = dict(node_dict)
        children_data = node_data.pop("children", [])

        # Convert bid to int if it's a string
        if "bid" in node_data and isinstance(node_data["bid"], str):
            node_data["bid"] = int(node_data["bid"])

        node = AnnotatedElement.model_validate(node_data)

        # Add children back and set up parent references
        for child_dict in children_data:
            child = AnnotatedElement.__build_tree(child_dict)
            node.__add_child(child)

        return node

    def __prune_dom_empty_paths(self, visible_bids: List[int]) -> None:
        """
        Prune branches of the DOM tree where no node in the subtree has a BID
        that appears in ``visible_bids``. This modifies the tree in place.

        Args:
            visible_bids: Collection of BIDs considered visible (e.g., from AX tree).
        """

        visible_set = set(visible_bids)

        def subtree_contains_visible_bid(node: "AnnotatedElement") -> bool:
            if node.bid is not None and node.bid in visible_set:
                return True
            for child in node.children:
                if subtree_contains_visible_bid(child):
                    return True
            return False

        def prune(node: "AnnotatedElement") -> None:
            # Keep only children whose subtree contains at least one visible bid
            kept_children: List[AnnotatedElement] = []
            for child in node.children:
                if subtree_contains_visible_bid(child):
                    prune(child)
                    kept_children.append(child)
            node.children = kept_children

        prune(self)

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

    @staticmethod
    def _run_async_blocking(coro):
        """
        Run an async coroutine to completion by executing it in a dedicated
        background thread with its own event loop. Always uses a thread to
        avoid interfering with any existing event loop in the current thread.
        """
        result_holder: Dict[str, Optional[BaseException] | Any] = {"result": None, "error": None}

        def _runner():
            try:
                result_holder["result"] = asyncio.run(coro)
            except BaseException as exc:  # noqa: BLE001 - bubble up original exception
                result_holder["error"] = exc

        thread = threading.Thread(target=_runner, name="AnnotatedElementAsyncRunner", daemon=True)
        thread.start()
        thread.join()

        error_obj: Optional[BaseException] = result_holder.get("error")
        if error_obj is not None:
            raise error_obj

        return result_holder["result"]

    @classmethod
    def from_axtree(cls, tree: dict) -> "AnnotatedElement":
        """
        Convert an accessibility tree to an annotated element tree.
        """
        _, axtree_dict = flatten_axtree_to_str_and_dict(tree)
        cls.__set_ids(axtree_dict)
        cls.__fill_missing_bids(axtree_dict)

        # Build the tree with proper parent references
        root = cls.__build_tree(axtree_dict)
        return root

    @classmethod
    def from_dom(
        cls, dom: dict, axtree: Optional[dict] = None, prune_empty_branches: bool = False
    ) -> "AnnotatedElement":
        """
        Convert a DOM to an annotated element tree.
        """

        if prune_empty_branches and axtree is None:
            raise ValueError("Cannot prune DOM without axtree to determine visible bids")

        _, dom_dict = flatten_dom_to_str_get_dict(dom)
        dom_dict = prune_dom_dict(dom_dict)

        cls.__set_ids(dom_dict)
        cls.__fill_missing_bids(dom_dict)

        # Build the tree with proper parent references
        root = cls.__build_tree(dom_dict)

        root.src_dom = dom_dict
        root.src_axtree = axtree

        if prune_empty_branches:
            root.visible_bids = AnnotatedElement.from_axtree(axtree).all_bids
            root.__prune_dom_empty_paths(root.visible_bids)

        return root

    def populate_metadata(
        self,
        user_query: str,
        security_policy: SecurityPolicy,
        site_details: str,
        sampling_params: dict,
        max_concurrent_requests: int = 20,
    ) -> list[dict]:
        """
        Build metadata for the DOM tree synchronously. Internally spawns a dedicated
        event loop in a background thread and waits for completion.
        """

        async def _run_all():
            semaphore = asyncio.Semaphore(max_concurrent_requests)
            unlock_cond = asyncio.Condition()
            messages = []

            async with AsyncOpenAI(
                api_key="sk-PCG3chp7epg_zuz5rng",
                base_url="http://localhost:4011/v1",
                max_retries=25,
            ) as _underlying_client:
                client = AsyncOpenAILoggingProxy(_underlying_client, messages)

                async def populate_metadata_for_element(element: "AnnotatedElement") -> None:
                    """
                    Populate metadata for a single element.
                    """
                    try:
                        # wait for parents to be resolved
                        async with unlock_cond:
                            await unlock_cond.wait_for(lambda: element.__are_ancestors_metadata_resolved)

                        if element.id % 100 == 0:
                            import os

                            proc_id = os.getpid()
                            logger.info(f"[{proc_id=}]: Populating metadata for element {element.id}")

                        # don't request metadata for elements with parents that are not allowed
                        if element.parent:
                            if element.parent.skip_metadata:
                                element.skip_metadata = True
                                # logger.info(
                                #     f"Skipping metadata for element {element.id} ({element.name}, {element.html_attributes}) because parent is not allowed"
                                # )
                                return

                            assert element.parent.metadata is not None
                            if element.parent.metadata.owner not in security_policy.integrity_levels.integrity_levels:
                                # logger.info(
                                #     f"Skipping metadata for element {element.id} ({element.name}, {element.html_attributes}) because parent is not allowed"
                                # )
                                element.skip_metadata = True
                                return

                        # copy from parent if it's a text element
                        if element.element_text:
                            assert element.parent is not None
                            assert element.parent.metadata is not None
                            element.metadata = element.parent.metadata
                            element.capabilities = element.parent.capabilities
                            # logger.info(
                            #     f"Copied metadata from parent for text element {element.id} ({element.element_text})"
                            # )
                            return

                        # # html root is dev content
                        # if element.id == 1:
                        #     element.metadata = AnnotatedElementMetadata(
                        #         owner=IntegrityLabel.DEVELOPER,
                        #         context="root element",
                        #         purpose="site structure",
                        #     )
                        #     return

                        # start resolving current node
                        async with semaphore:
                            # logger.info(
                            #     f"Populating metadata for element {element.id} ({element.name}, {element.html_attributes})"
                            # )

                            resp = await client.beta.chat.completions.with_raw_response.parse(
                                messages=[
                                    {
                                        "role": "system",
                                        "content": "You are a helpful assistant that annotates HTML elements.",
                                    },
                                    {"role": "user", "content": METADATA_INSTRUCTION_PROMPT},
                                    {
                                        "role": "user",
                                        "content": f"Here are some contextual details about the site: {site_details}",
                                    },
                                    {"role": "user", "content": element.as_direct_lineage_html_tree_top},
                                ],
                                response_format=AnnotatedElementMetadata,
                                extra_body={"cache": {"use-cache": True}},  # Allow caching owner label requests
                                **sampling_params,
                            )
                            # print(f"Requested metadata: \n\n{element.as_direct_lineage_html_tree_top}")
                            # print(f"Response: \n\n{resp.choices[0].message.parsed}")
                            message = resp.choices[0].message
                            assert message.parsed
                            element.metadata = message.parsed
                            assert element.metadata is not None

                            # Override owner with parent if parent is untrusted
                            if (
                                element.parent
                                and element.parent.metadata
                                and (parent_owner := element.parent.metadata.owner) != IntegrityLabel.DEVELOPER
                            ):
                                element.metadata.owner = parent_owner

                            # No need to request capabilities for content that is not allowed
                            if element.metadata.owner not in security_policy.integrity_levels.integrity_levels:
                                # logger.info(
                                #     f"Skipping capabilities for element {element.id} ({element.name}, {element.html_attributes}) because owner is not allowed"
                                # )
                                element.skip_metadata = True
                                return

                            # # request capabilities -- only for visible elements
                            # assert element.root.visible_bids is not None
                            # if element.bid not in element.root.visible_bids:
                            #     # logger.info(
                            #     #     f"Skipping capabilities for element {element.id} ({element.name}, {element.html_attributes}) because it is not visible"
                            #     # )
                            #     return

                            # Testing Mode - Dev elements should have all capabilities
                            if element.metadata.owner == IntegrityLabel.DEVELOPER:
                                element.capabilities = CapabilityLabel.INTERACTABLE
                                return

                            if element.parent and element.parent.capabilities == CapabilityLabel.VIEWABLE:
                                element.capabilities = CapabilityLabel.VIEWABLE
                                return

                            resp = await client.beta.chat.completions.with_raw_response.parse(
                                messages=[
                                    {
                                        "role": "system",
                                        "content": "You are a helpful security assistant that makes decisions about what HTML elements have what capabilities.",
                                    },
                                    {"role": "user", "content": CAP_INSTRUCTION_PROMPT},
                                    {
                                        "role": "user",
                                        "content": f"Here are some contextual details about the site: {site_details}",
                                    },
                                    {"role": "user", "content": f"Here is the user query: {user_query}"},
                                    {
                                        "role": "user",
                                        "content": f"HTML: {element.as_direct_lineage_relevance_tree_top}",
                                    },
                                ],
                                response_format=AnnotatedElementCapabilities,
                                extra_body={"cache": {"use-cache": True}},  # Allow caching capabilities requests
                                **sampling_params,
                            )
                            message = resp.choices[0].message
                            assert message.parsed
                            element.capabilities = message.parsed.capability
                            # print(f"Requested capabilities: \n\n{element.as_direct_lineage_relevance_tree_top}")
                            # print(f"Response: \n\n{message.parsed}")

                            # Test - set all to high
                            # element.relevance = RelevanceLabel.HIGH
                    finally:
                        # notify waiters that a parent is resolved (metadata set or skipped)
                        async with unlock_cond:
                            unlock_cond.notify_all()

                logger.info(f"Populating metadata for {len(self.all_elements)} elements")
                await asyncio.gather(*[populate_metadata_for_element(child) for child in self.all_elements])
                logger.info(f"Populated metadata for {len(self.all_elements)} elements")

            return messages

        start = time.time()
        messages = self._run_async_blocking(_run_all())
        end = time.time()

        logger.info(f"Resolved metadata for {len(self.all_elements)} elements in {end - start:.2f} seconds")

        return messages


# Self‐referencing model
AnnotatedElement.model_rebuild()
