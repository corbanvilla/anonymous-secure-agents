from typing import List

from src.experiments.config import (
    AgentSrcType,
    AttackMode,
    DefenseMode,
    LanguageModel,
    ModelSamplingParams,
)
from src.experiments.config.defaults import DEFAULT_OPENAI_CLIENT
from src.tasks import WA_EASY_20, WA_EASY_ALL, WA_EASY_100

TESTS = True

ENGINE_MODEL = LanguageModel.GPT5_MINI
DEFENSE_MODE = DefenseMode.NONE # NONE, ONE_STAGE, STRICTED_ONE_STAGE, STRICTED_OWNERS, DUAL_DEFENSE
ATTACK_MODE = AttackMode.DISTRACTING_BANNER_AD # NONE, EDA, EIA, EIA_VISIBLE, POPUP, EIA_INPUT_FIELD, FAKE_COMP, IGNORE_INSTRUCTION, INFEASIBLE, DISTRACTING_BANNER_AD
N_PARALLEL_AGENTS = 10

BASENAME = "WA_EASY_ALL_MIRROR_BASELINE"
EXPERIMENT_DESCRIPTION = "MIRROR baseline no defense"
TASKS = WA_EASY_20

WEBAGENT_SRC_COMBINATIONS: List[list[AgentSrcType]] = [[AgentSrcType.A11Y]]
MAX_STEPS = 15

TIMESTAMP_EXP_NAME = False
RESTART_VWA_SERVER = True
if TESTS:
    BASENAME = "wa-easy-all-mirror-baseline-test"
    EXPERIMENT_DESCRIPTION = "mirror testing"
    # TASKS = WA_EASY_ALL
    TASKS = TASKS
    MAX_STEPS = 12
    TIMESTAMP_EXP_NAME = True
    RESTART_VWA_SERVER = False
    N_PARALLEL_AGENTS = 20

ATTACK_KWARGS = {
    "openai_client": DEFAULT_OPENAI_CLIENT,
    "sampling_params": ModelSamplingParams(
        model=LanguageModel.GPT5_MINI,
    ).model_dump(),
}

DEFENSE_KWARGS = {
    "openai_client": DEFAULT_OPENAI_CLIENT,
    "sampling_params": ModelSamplingParams(
        model=LanguageModel.GEMINI_25_FLASH_LITE,
    ).model_dump(),
}
