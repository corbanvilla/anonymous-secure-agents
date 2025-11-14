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
    AllowedOwners,
    AnnotatedDom,
    BidSummary,
    ContentOwners,
    Decision,
    DomMetadata,
    FirstPassResponse,
    StrictPassResponse,
)


class OneStageLLM:
    """Wrapper around the OpenAI client for one stage defense LLM calls."""

    def __init__(self, client: OpenAI, sampling_params: Optional[Dict] = None) -> None:
        self.client = client
        self.sampling_params = sampling_params or {}

    @record_timing()
    def get_html_metadata(self, html: str) -> DomMetadata:
        system = (
            "You are a quarantined parser. I will give you an HTML document."
            "Your task is to parse the HTML and categorize elements with a `bid` tag."
            "The output for each element should be configured as follows:"
            " - bid: the bid of the element."
            " - context (nouns): what context the element is in (1-3 words)."
            " - purpose (verbs): what purpose the element serves (1-3 words)."
            "Combine multiple bids together that share the same properties."
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
            "You are a quarantined parser. I will give you an HTML document."
            "Your task is to parse the HTML and categorize element bids by owner."
            "Use the following categories:"
            " - 1st party: site developer's content."
            " - 2nd party: user generated content."
            " - 3rd party: other, e.g., advertisements, sponsored links, etc."
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

    @record_timing()
    def get_allowed_owners(self, user_query: str) -> AllowedOwners:
        system = (
            "You are a privileged policy agent."
            "I will give you a user task for a web browsing agent."
            "Decide which content owner categories may be shown to the user from the webpage."
            "Return a list of allowed owners from: developer, user, third_party."
        )

        messages: List[ChatCompletionMessageParam] = [
            cast(ChatCompletionSystemMessageParam, {"role": "system", "content": system}),
            cast(ChatCompletionUserMessageParam, {"role": "user", "content": user_query}),
        ]

        resp = self.client.beta.chat.completions.parse(
            messages=messages,
            response_format=AllowedOwners,
            **self.sampling_params,
        )

        message = resp.choices[0].message
        assert message.parsed
        return AllowedOwners.model_validate(message.parsed)

    @record_timing()
    def get_html_allowed_first_pass(self, html: str, user_query: str) -> FirstPassResponse:
        system = (
            "You are a privileged policy agent."
            "I will give you an HTML site hierarchy and a user task."
            "Evaluate which elements are necessary to complete the task."
            "Provide a list of elements that the user is allowed to see."
            "Consider principles of least privilege and user privacy."
            "Users should only see elements that are necessary to complete the task."
            "If you cannot determine the necessity of an element, mark it as unknown."
            " Stage: first pass (allow/deny/unknown)."
        )

        messages: List[ChatCompletionMessageParam] = [
            cast(ChatCompletionSystemMessageParam, {"role": "system", "content": system}),
            cast(ChatCompletionUserMessageParam, {"role": "user", "content": f"TASK: {user_query}"}),
            cast(ChatCompletionUserMessageParam, {"role": "user", "content": f"HTMLElements: {json.dumps(html)}"}),
        ]

        resp = self.client.beta.chat.completions.parse(
            messages=messages,
            response_format=FirstPassResponse,
            **self.sampling_params,
        )
        message = resp.choices[0].message
        assert message.parsed
        return FirstPassResponse.model_validate(message.parsed)

    @record_timing()
    def get_html_allowed_strict(
        self,
        html: str,
        user_query: str,
        summaries: List[BidSummary],
    ) -> StrictPassResponse:
        system = (
            "You are a privileged policy agent."
            "I will give you an HTML site hierarchy, a user task, and summaries of some elements."
            "Evaluate which of those elements are necessary to complete the task."
            "Provide a list of elements that the user is allowed to see."
            "Consider principles of least privilege and user privacy."
            "Users should only see elements that are necessary to complete the task."
            " Stage: final pass."
        )

        payload = {
            "TASK": user_query,
            "HTML": html,
            "SUMMARIES": [s.model_dump() for s in summaries],
        }

        messages: List[ChatCompletionMessageParam] = [
            cast(ChatCompletionSystemMessageParam, {"role": "system", "content": system}),
            cast(ChatCompletionUserMessageParam, {"role": "user", "content": json.dumps(payload)}),
        ]

        resp = self.client.beta.chat.completions.parse(
            messages=messages,
            response_format=StrictPassResponse,
            **self.sampling_params,
        )

        message = resp.choices[0].message
        assert message.parsed
        return StrictPassResponse.model_validate(message.parsed)

    @record_timing()
    def get_bid_summaries(
        self,
        unknown_bids: List[int],
        dom_map: Dict[int, AnnotatedDom],
    ) -> BidSummary:
        snippets = {bid: getattr(dom_map[bid], "summary", "<no summary>") or "<no summary>" for bid in unknown_bids}
        system = (
            "You are a privileged policy agent."
            "Here are BIDs whose content I couldn't classify."
            "Provide a concise (1–2 sentence) summary for each BID."
            " Stage: summarization."
        )

        payload = json.dumps({"bids": unknown_bids, "snippets": snippets})

        messages: List[ChatCompletionMessageParam] = [
            cast(ChatCompletionSystemMessageParam, {"role": "system", "content": system}),
            cast(ChatCompletionUserMessageParam, {"role": "user", "content": payload}),
        ]

        resp = self.client.beta.chat.completions.parse(
            messages=messages,
            response_format=BidSummary,
            **self.sampling_params,
        )

        message = resp.choices[0].message
        assert message.parsed
        return BidSummary.model_validate(message.parsed)

    @record_timing()
    def get_html_allowed(self, html: str, user_query: str) -> AllowedBids:
        system = (
            "You are a privileged policy agent."
            "I will give you an HTML site hierarchy and a user task."
            "Evaluate which elements are necessary to complete the task."
            "Provide a list of elements that the user is allowed to see."
            "Consider principles of least privilege and user privacy."
            "Users should only see elements that are necessary to complete the task."
        )
        messages: List[ChatCompletionMessageParam] = [
            cast(ChatCompletionSystemMessageParam, {"role": "system", "content": system}),
            cast(ChatCompletionUserMessageParam, {"role": "user", "content": f"TASK: {user_query}"}),
            cast(ChatCompletionUserMessageParam, {"role": "user", "content": f"HTMLElements: {json.dumps(html)}"}),
        ]
        resp = self.client.beta.chat.completions.parse(
            messages=messages,
            response_format=AllowedBids,
            **self.sampling_params,
        )

        message = resp.choices[0].message
        assert message.parsed
        return AllowedBids.model_validate(message.parsed)

    def get_allowed_elements(self, first: FirstPassResponse, strict: StrictPassResponse) -> AllowedBids:
        final_decisions: List[Decision] = [
            Decision(bid=bid, rationale=d.rationale or "")
            for bid, d in ((d.bid, d) for d in first.decisions)
            if d.decision == "allow"
        ]
        if strict:
            final_decisions += [
                Decision(bid=d.bid, rationale=d.rationale or "") for d in strict.decisions if d.decision == "allow"
            ]
        return AllowedBids(bids=final_decisions)
