from __future__ import annotations

from typing import List

from openai import AsyncOpenAI, OpenAI

from src.experiments.config.models import (
    AsyncEngineArgs,
    LanguageModel,
    ModelSamplingParams,
)

# Default OpenAI client configuration
DEFAULT_OPENAI_CLIENT = OpenAI(
    api_key="sk-PCG3chp7epg_zuz5rng",
    base_url="http://localhost:4011/v1",
    max_retries=25,
)

DEFAULT_OPENAI_CLIENT_ASYNC = AsyncOpenAI(
    api_key="sk-PCG3chp7epg_zuz5rng",
    base_url="http://localhost:4011/v1",
    max_retries=25,
)

DEFAULT_SAMPLING_PARAMS = ModelSamplingParams(
    model=LanguageModel.GPT4_MINI,
    temperature=1.0,
)

DEFAULT_ENGINE_OPTIONS = AsyncEngineArgs(
    n_parallel_agents=2,
    engine_name="openai",
    max_steps=12,
    max_response_length=32768,
    max_prompt_length=32768 * 20,
    sampling_params=DEFAULT_SAMPLING_PARAMS,
)

DEFAULT_IGNORE_KEYS_DEFENSE_HARNESS: List[str] = [
    "defense_kwargs.openai_client",
    "defense_kwargs.async_openai_client",
    "defense_kwargs.sampling_params.max_retries",
    "n_parallel_agents",
]

DEFAULT_IGNORE_KEYS: List[str] = [
    "engine_options.n_parallel_agents",
    "engine_options.max_prompt_length",
    "engine_options.api_client",
    "env_args.attack_kwargs.openai_client",
    "env_args.defense_kwargs.openai_client",
    "env_args.defense_kwargs.async_openai_client",
    "engine_options.sampling_params.max_retries",
]
