from src.experiments.config import (
    DefenseMode,
    LanguageModel,
    ModelSamplingParams,
)
from src.experiments.config.defaults import DEFAULT_OPENAI_CLIENT

TESTS = True

DEFENSE_MODE = DefenseMode.ONE_STAGE  # ONE_STAGE, STRICTED_ONE_STAGE, STRICTED_OWNERS, DUAL_DEFENSE
N_PARALLEL_AGENTS = 3

BASENAME = "defense-eval-gemini-2.5-flash-v3"
EXPERIMENT_DESCRIPTION = "Evaluate defense harness"
REFERENCE_EXPERIMENT_NAME = "WA-EASY-20-a11y-gpt-4.1-mini"

TIMESTAMP_EXP_NAME = False
if TESTS:
    BASENAME = "test-new-harness-5"
    EXPERIMENT_DESCRIPTION = "Test evaluation"
    TIMESTAMP_EXP_NAME = True


DEFENSE_KWARGS: dict = {
    "openai_client": DEFAULT_OPENAI_CLIENT,
    "sampling_params": ModelSamplingParams(
        model=LanguageModel.GEMINI_25_FLASH,
    ).model_dump(),
}
