from typing import List

from src.experiments.config import (
    AgentSrcType,
    AttackMode,
    DefenseMode,
    LanguageModel,
    ModelSamplingParams,
)
from src.experiments.config.defaults import DEFAULT_OPENAI_CLIENT, DEFAULT_OPENAI_CLIENT_ASYNC
from src.tasks import WA_EASY_20

TESTS = False

ENGINE_MODEL = LanguageModel.GPT5_MINI
DEFENSE_MODE = DefenseMode.TRI_DEFENSE_V2  # NONE, ONE_STAGE, STRICTED_ONE_STAGE, STRICTED_OWNERS, DUAL_DEFENSE
ATTACK_MODE = AttackMode.INFEASIBLE  # NONE, EDA, EIA, EIA_V2, POPUP
N_PARALLEL_AGENTS = 20

BASENAME = "WA-EASY-20-PRELIM-V3"
EXPERIMENT_DESCRIPTION = "Test of TRI defense update, with Gemini 20 flash"
TASKS = WA_EASY_20

WEBAGENT_SRC_COMBINATIONS: List[list[AgentSrcType]] = [[AgentSrcType.A11Y]]
MAX_STEPS = 12

TIMESTAMP_EXP_NAME = False
RESTART_VWA_SERVER = False
if TESTS:
    BASENAME = "test-new-defense"
    EXPERIMENT_DESCRIPTION = "Test evaluation"
    TASKS = ["browsergym/webarena.safe.187"]
    MAX_STEPS = 2
    TIMESTAMP_EXP_NAME = True
    RESTART_VWA_SERVER = False
    N_PARALLEL_AGENTS = 1

ATTACK_KWARGS = {
    # "openai_client": DEFAULT_OPENAI_CLIENT,
    "sampling_params": ModelSamplingParams(
        model=LanguageModel.GEMINI_20_FLASH,
    ).model_dump(),
}

DEFENSE_KWARGS = {
    # "openai_client": DEFAULT_OPENAI_CLIENT,
    "sampling_params": ModelSamplingParams(
        model=LanguageModel.GEMINI_25_FLASH,
    ).model_dump(),
    # "async_openai_client": DEFAULT_OPENAI_CLIENT_ASYNC,
    "sampling_params_labeler": ModelSamplingParams(
        model=LanguageModel.GEMINI_20_FLASH,
    ).model_dump(),
}
