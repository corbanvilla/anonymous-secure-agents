from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import cast

import openai
from beartype import beartype
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from webarena.evaluation_harness.evaluators import (
    Evaluator,
    EvaluatorComb,
    HTMLContentEvaluator,
    StringEvaluator,
    URLEvaluator,
)

client = openai.OpenAI()


class EvaluationResult(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIALLY_CORRECT = "partially correct"


class UnachievableResult(str, Enum):
    SAME = "same"
    DIFFERENT = "different"


class FuzzyMatchResponse(BaseModel):
    explanation: str
    result: EvaluationResult


class UnachievableMatchResponse(BaseModel):
    explanation: str
    result: UnachievableResult


class StringEvaluatorUpdated(StringEvaluator):
    @staticmethod
    @beartype
    def fuzzy_match(ref: str, pred: str, intent: str) -> float:
        return llm_fuzzy_match(pred, ref, intent)

    @staticmethod
    @beartype
    def ua_match(ref: str, pred: str, intent: str) -> float:
        return llm_ua_match(pred, ref, intent)


@beartype
def evaluator_router(config_file: Path | str) -> EvaluatorComb:
    """Router to get the evaluator class"""
    with open(config_file, "r") as f:
        configs = json.load(f)

    eval_types = configs["eval"]["eval_types"]
    evaluators: list[Evaluator] = []
    for eval_type in eval_types:
        match eval_type:
            case "string_match":
                evaluators.append(StringEvaluatorUpdated())
            case "url_match":
                evaluators.append(URLEvaluator())
            case "program_html":
                evaluators.append(HTMLContentEvaluator())
            case _:
                raise ValueError(f"eval_type {eval_type} is not supported")

    return EvaluatorComb(evaluators)


def llm_fuzzy_match(pred: str, reference: str, question: str) -> float:
    """Check whether the prediction matches the reference with GPT4-turbo"""
    messages: list[ChatCompletionMessageParam] = []
    # construct the question to ask
    message = "Help a teacher to grade the answer of a student given a question. Keep in mind that the student may use different phrasing or wording to answer the question. The goal is to evaluate whether the answer is semantically equivalent to the reference answer.\n"
    message += f"question: {question}\n"
    message += f"reference answer: {reference}\n"
    message += "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
    message += f"student answer: {pred}\n"
    message += "Provide your evaluation with a detailed explanation of why the answer is correct, incorrect, or partially correct."
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": message},
    ]

    parsed = client.beta.chat.completions.parse(
        model="gpt-5-mini",
        messages=messages,
        # temperature=0,
        max_completion_tokens=768,
        top_p=1.0,
        response_format=FuzzyMatchResponse,
    )
    response = cast(FuzzyMatchResponse, parsed.choices[0].message.parsed)

    match response.result:
        case EvaluationResult.CORRECT:
            return 1.0
        case _:
            return 0.0


def llm_ua_match(pred: str, reference: str, question: str) -> float:
    """Check whether the prediction matches the reference with GPT-turbo"""
    messages: list[ChatCompletionMessageParam] = []
    # construct the question to ask
    message = ""
    message += f"task: {question}\n"
    message += f"actual unachievable reason: {reference}\n"
    message += f"reported unachievable reason: {pred}\n"
    message += (
        "The task described above is inherently unachievable due to the reason specified under 'actual unachievable reason'. "
        "An individual previously attempted this task and was unable to complete it. They provided a reason for their failure, "
        "which is listed under 'reported unachievable reason'. Your role is to review both the actual and reported reasons. "
        "Determine if the reported reason aligns with the actual reason, even if implicitly. "
        "Provide your evaluation with a detailed explanation of why the reasons are the same or different."
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": message},
    ]

    parsed = client.beta.chat.completions.parse(
        model="gpt-5-mini",
        messages=messages,
        # temperature=0,
        max_completion_tokens=768,
        top_p=1.0,
        response_format=UnachievableMatchResponse,
    )
    response = cast(UnachievableMatchResponse, parsed.choices[0].message.parsed)

    match response.result:
        case UnachievableResult.SAME:
            return 1.0
        case _:
            return 0.0
