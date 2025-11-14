from __future__ import annotations

import json
from typing import Dict, List, Optional, cast
import re

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from src.db.helpers.timing import record_timing

from .models import (
    AllowedBids,
    AnnotatedDom,
    AnnotatedElement,
    ContentOwners,
    Decision,
    DomMetadata,
    RelevanceLabel,
    RelevanceLabels,
    SecurityPolicy,
    BidRelevanceResponse,
)


class PolicyAgent:
    """Wrapper around the OpenAI client for policy agent LLM calls."""

    def __init__(self, client: OpenAI, sampling_params: Optional[Dict] = None) -> None:
        self.client = client
        self.sampling_params = sampling_params or {}

    @record_timing()
    def get_security_policy(self, user_query: str, url: str) -> SecurityPolicy:
        system_prompt = """You are a security policy agent controlling data access for a web browsing agent handling everyday user tasks. Analyze user requests and set security policies that balance safety with functionality.
        INTEGRITY LEVELS (Content Types)
        Select which content types to allow:
        'developer' - Site developer's content (official pages, documentation, product info, navigation)
        'user' - User generated content (reviews, comments, forum posts, social media)
        'third-party' - Other content (advertisements, sponsored links, affiliate content)
        Output a subset of these levels to be allowed.
        RELEVANCE LEVELS (Information Scope)
        Select threshold for information breadth:
        'high' - Only directly relevant information allowed
        'medium' - Direct + supporting information allowed
        'low' - All information types allowed
        Higher selections include lower levels (low > medium > high).
        DECISION GUIDELINES
        Integrity Selection:
        Include 'developer' for most tasks requiring official site functionality:

        Account management, navigation, procedures
        Product specifications, official documentation
        Standard site operations

        Add 'user' when task specifically needs user-generated content:

        Reading reviews/ratings for decision-making
        Social media sentiment or discussions
        Community troubleshooting experiences
        User opinions/experiences

        Add 'third-party' only when external content may be necessary:

        Comparison shopping that may include sponsored content
        Tasks where ads might contain relevant information

        Relevance Selection:
        'high' - Specific, well-defined tasks with clear objectives
        'medium' - Tasks with some ambiguity or needing context
        'low' - Complex exploration tasks (use sparingly)
        EXAMPLES
        "Buy me the best noise-canceling headphones under $200"
        → Integrity: ['developer', 'user'], Relevance: 'medium'
        (Need product pages + user reviews; somewhat open-ended criteria)
        "Change my Netflix password"
        → Integrity: ['developer'], Relevance: 'high'
        (Official help docs sufficient; very specific task)
        "Find a good Italian restaurant for tonight"
        → Integrity: ['developer', 'user'], Relevance: 'medium'
        (Need restaurant info + user reviews; subjective criteria)
        "How do I cancel my subscription?"
        "Compare the time for walking and driving to Oracle Park"
        → Integrity: ['developer'], Relevance: 'high'
        (No user content needed; official procedures sufficient)
        "Change my Netflix password"
        → Integrity: ['developer'], Relevance: 'high'
        (Official support docs; specific procedure)
        OUTPUT FORMAT
        {
        "reasoning": "Brief explanation of content types needed",
        "integrity_levels": ["developer", "user", "third-party"],
        "relevance_level": "high|medium|low"
        }"""

        payload = {
            "TASK": user_query,
            "URL": url,
        }

        messages: List[ChatCompletionMessageParam] = [
            cast(ChatCompletionSystemMessageParam, {"role": "system", "content": system_prompt}),
            cast(ChatCompletionUserMessageParam, {"role": "user", "content": json.dumps(payload)}),
        ]

        resp = self.client.beta.chat.completions.parse(
            messages=messages,
            response_format=SecurityPolicy,
            **self.sampling_params,
        )

        message = resp.choices[0].message
        assert message.parsed
        return SecurityPolicy.model_validate(message.parsed)

    @record_timing()
    def get_html_metadata(self, html: str) -> DomMetadata:
        system = (
            "You are a quarantined parser. I will give you an HTML document. "
            "Your task is to parse the HTML and categorize elements with `bid` attributes. "
            "The output should group elements into metadata entries as follows:\n"
            " - bids: list of numeric bid values for elements that share the same properties\n"
            " - context: select the most appropriate context from these options: "
            "head, header, navigation, main, sidebar, footer, form, advertisement, modal, iframe, user_content, third_party_widget, other\n"
            " - purpose: select the most appropriate purpose from these options: "
            "content, link, style, script, media, form_control, structure, advertising, tracking, authentication, comment, embed, other\n"
            "Combine elements with the same context and purpose into single entries with multiple bids. "
            "Each element should be categorized by where it appears on the page (context) and what role it serves (purpose)."
        )

        messages: List[ChatCompletionMessageParam] = [
            cast(ChatCompletionSystemMessageParam, {"role": "system", "content": system}),
            cast(ChatCompletionUserMessageParam, {"role": "user", "content": html}),
        ]

        resp = self.client.beta.chat.completions.parse(
            messages=messages,
            response_format=DomMetadata,
            **self.sampling_params,
        )

        message = resp.choices[0].message
        assert message.parsed
        return DomMetadata.model_validate(message.parsed)

    @record_timing()
    def get_html_owners(self, html: str) -> ContentOwners:
        system = (
            "You are a quarantined parser. I will give you an HTML document. "
            "Your task is to parse the HTML and categorize element bid values by owner type. "
            "Extract numeric bid attributes from elements and classify them into these categories:\n"
            " - developer: Content and elements created by the site's developers (core site functionality, layout, branding)\n"
            " - user: User-generated content (comments, posts, uploaded media, user profiles)\n"
            " - third_party: External content (advertisements, social media embeds, analytics scripts, third-party widgets)\n"
            "Return lists of numeric bid values for each owner category. "
            "Each bid should appear in only one category based on the element's primary ownership."
        )

        messages: List[ChatCompletionMessageParam] = [
            cast(ChatCompletionSystemMessageParam, {"role": "system", "content": system}),
            cast(ChatCompletionUserMessageParam, {"role": "user", "content": html}),
        ]

        resp = self.client.beta.chat.completions.parse(
            messages=messages,
            response_format=ContentOwners,
            **self.sampling_params,
        )

        message = resp.choices[0].message
        assert message.parsed
        return ContentOwners.model_validate(message.parsed)
    
    def find_largest_bid_number_regex(self, html: str) -> Optional[int]:
        """
        Alternative approach using regex to find bid numbers.
        More flexible - finds bid="number" or data-bid="number" patterns.
        """
        # Pattern to match bid="123" or data-bid="123"
        pattern = r'(?:data-)?bid="(\d+)"'
        matches = re.findall(pattern, html)
        
        if matches:
            bid_numbers = [int(match) for match in matches]
            return max(bid_numbers)
        
        return None

    @record_timing()
    def get_html_relevance(self, html: str, user_task: str) -> RelevanceLabels:
        system = (
    "You are a quarantined parser analyzing HTML elements for task relevance. "
    "I will provide an HTML document and a user task. Parse the HTML and extract numeric bid attributes, "
    "then classify them into relevance categories based on how they relate to the user task.\n\n"
    "Be generous and lenient when assigning relevance levels:\n\n"
    "- high: Elements that could be directly useful for the task, including buttons, forms, links, "
    "interactive elements, main content, navigation, search boxes, or any element that might help accomplish the task. "
    "Cast a wide net - if an element could reasonably be involved in completing the task, classify it as high.\n\n"
    "- medium: Elements providing context, labels, headers, secondary content, or structural elements "
    "that support understanding of the page or task.\n\n"
    "- low: Irrelevant elements like unrelated advertisements or purely decorative content.\n\n"
    "IMPORTANT: You must populate the high category with at least a few elements."
    "Classify elements generously. When in doubt, always choose the higher relevance category.\n\n"
    "Return your response as a JSON object with this exact structure:\n"
    "{\n"
    '  "mapping": {\n'
    '    "1": "high",\n'
    '    "2": "medium",\n'
    '    "3": "low"\n'
    '  },\n'
    "}\n\n"
    "- You CANNOT return an empty mapping\n"
    "- Use bid numbers as string keys in the mapping\n"
    "- Only use 'high', 'medium', or 'low' as values\n"
    "- Include all found bid numbers in the mapping\n"
)

        messages: List[ChatCompletionMessageParam] = [
            cast(ChatCompletionSystemMessageParam, {"role": "system", "content": system}),
            cast(
                ChatCompletionUserMessageParam, {"role": "user", "content": f"User Task: {user_task}\n\nHTML:\n{html}"}
            ),
        ]
        # largest_bid_number = self.find_largest_bid_number_regex(html)
        # ExplicitBidRelevanceMap = create_explicit_bid_model(
        #     max_bid_number=largest_bid_number if largest_bid_number is not None else 2000,
        #     default_level=RelevanceLabel.HIGH
        # )
        # schema = ExplicitBidRelevanceMap.model_json_schema()
        resp = self.client.beta.chat.completions.parse(
            messages=messages,
            response_format=BidRelevanceResponse,
            **self.sampling_params,
        )
        message = resp.choices[0].message
        assert message.parsed
        mapping = BidRelevanceResponse.model_validate(message.parsed)
        print('resp', resp)
        print('message', message)
        print("Relevance mapping:", mapping.mapping)
        return mapping.to_labels()

    def filter_by_security_policy_strict(
        self, annotated_dom: AnnotatedDom, security_policy: SecurityPolicy
    ) -> AllowedBids:
        """
        Returns elements that directly satisfy policy.
        """

        # Define relevance hierarchy (higher values include lower levels)
        relevance_hierarchy = {
            RelevanceLabel.HIGH: {RelevanceLabel.HIGH},
            RelevanceLabel.MEDIUM: {RelevanceLabel.HIGH, RelevanceLabel.MEDIUM},
            RelevanceLabel.LOW: {RelevanceLabel.HIGH, RelevanceLabel.MEDIUM, RelevanceLabel.LOW},
        }

        # Get allowed integrity levels from policy
        print('security policy', security_policy)
        print('security_policy.integrity_levels:', security_policy.integrity_levels)
        allowed_integrity_levels = set(security_policy.integrity_levels.integrity_levels)

        # Get allowed relevance levels based on minimum threshold
        allowed_relevance_levels = relevance_hierarchy[security_policy.relevance_level]

        # Initialize allowed bids structure
        allowed_bids = AllowedBids()

        def evaluate_element(element: AnnotatedElement) -> tuple[bool, str]:
            """Check if element satisfies security policy constraints and return rationale."""

            # Check integrity level (owner)
            if element.owner not in allowed_integrity_levels:
                return False, f"Owner '{element.owner}' not in allowed integrity levels"

            # Check relevance level
            if element.relevance:
                try:
                    element_relevance = RelevanceLabel(element.relevance)
                    if element_relevance not in allowed_relevance_levels:
                        return False, f"Relevance '{element.relevance}' below threshold"
                except ValueError:
                    return False, f"Invalid relevance value: '{element.relevance}'"
            else:
                return False, "No relevance label set"

            return True, f"Allowed: {element.owner} owner, {element.relevance} relevance"

        def collect_allowed_bids(nodes: List[AnnotatedElement]):
            """Recursively collect bids from elements that satisfy policy."""
            for node in nodes:
                is_allowed, rationale = evaluate_element(node)
                if is_allowed:
                    # Create Decision object and add to allowed_bids
                    decision = Decision(bid=node.bid, rationale=rationale)
                    allowed_bids.bids.append(decision)

                # Always check children
                if node.children:
                    collect_allowed_bids(node.children)

        # Start collection from root nodes
        collect_allowed_bids(annotated_dom.children)

        # Sort the decisions by bid for consistency
        allowed_bids.bids.sort(key=lambda d: d.bid)

        return allowed_bids
