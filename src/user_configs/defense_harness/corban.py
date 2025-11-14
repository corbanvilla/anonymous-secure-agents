from src.experiments.config import (
    DefenseMode,
    LanguageModel,
    ModelSamplingParams,
)
from src.experiments.config.defaults import DEFAULT_OPENAI_CLIENT, DEFAULT_OPENAI_CLIENT_ASYNC

TESTS = False

DEFENSE_MODE = DefenseMode.TRI_DEFENSE_V2  # ONE_STAGE, STRICTED_ONE_STAGE, STRICTED_OWNERS, DUAL_DEFENSE
N_PARALLEL_AGENTS = 10

BASENAME = "defense-eval-t25"
EXPERIMENT_DESCRIPTION = "First run of tri-defense on whole easy dataset - refactor cap prompt"
# REFERENCE_EXPERIMENT_NAME = "WA-EASY-BASELINE-V3-a11y-none-none-gpt-5-mini"
REFERENCE_EXPERIMENT_NAME = "WA-EASY-100-BASELINE-V3-a11y-none-none-gpt-5-mini"
# REFERENCE_EXPERIMENT_NAME = "WA-EASY-20-a11y-gpt-4.1-mini"

TIMESTAMP_EXP_NAME = False
if TESTS:
    BASENAME = "test-new-harness-14"
    EXPERIMENT_DESCRIPTION = "Test evaluation"
    TIMESTAMP_EXP_NAME = False


DEFENSE_KWARGS: dict = {
    # "openai_client": DEFAULT_OPENAI_CLIENT,
    # "async_openai_client": DEFAULT_OPENAI_CLIENT_ASYNC,
    "sampling_params": ModelSamplingParams(
        model=LanguageModel.GEMINI_25_FLASH_LITE,
        temperature=0.0,
        # reasoning_effort="none",
    ).model_dump(),
    "sampling_params_labeler": ModelSamplingParams(
        model=LanguageModel.GEMINI_20_FLASH_LITE,
        temperature=0.0,
        # reasoning_effort="none",
    ).model_dump(),
}
