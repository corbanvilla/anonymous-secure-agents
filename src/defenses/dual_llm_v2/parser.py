from __future__ import annotations

import json
from typing import Dict, List, Optional, cast

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from src.db.helpers.timing import record_timing

from .models import (
    AllowedBids,
    AnnotatedElement,
    Decision,
    RelevanceLabel,
    SecurityPolicy,
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

    def filter_by_security_policy_strict(
        self, annotated_dom: AnnotatedElement, security_policy: SecurityPolicy
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
        print("security policy", security_policy)
        print("security_policy.integrity_levels:", security_policy.integrity_levels)
        allowed_integrity_levels = set(security_policy.integrity_levels.integrity_levels)

        # Get allowed relevance levels based on minimum threshold
        allowed_relevance_levels = relevance_hierarchy[security_policy.relevance_level]

        # Initialize allowed bids structure
        allowed_bids = AllowedBids()

        def evaluate_element(element: AnnotatedElement) -> tuple[bool, str]:
            """Check if element satisfies security policy constraints and return rationale."""

            # Check integrity level (owner)
            assert element.metadata is not None
            if element.metadata.owner not in allowed_integrity_levels:
                return False, f"Owner '{element.metadata.owner}' not in allowed integrity levels"

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

            return True, f"Allowed: {element.metadata.owner} owner, {element.relevance} relevance"

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
