from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from src.attacks.banner_ad import DistractingBannerAd
from src.attacks.eda_new import EDAAttackV2
# from src.attacks.eia import EIAAttackv2
from src.attacks.eia_input_field import EIAInputField
from src.attacks.eia_visible import EIAAttackv2
from src.attacks.fake_comp import FakeCompletion
from src.attacks.ignore_instr import IgnoreInstruction
from src.attacks.infeasible import Infeasible
from src.attacks.popup import PopupAttack
from src.defenses.dual_llm.dual_llm import DualDefense
from src.defenses.dual_llm_v2.dual_llm_v2 import DualDefenseV2
from src.defenses.one_stage.one_stage import OneStageDefense
from src.defenses.stricted_one_stage.stricted_one_stage import StrictedOneStageDefense
from src.defenses.stricted_owners.stricted_owners import StrictedDefense
from src.defenses.tri_llm.tri_llm import TriDefense
from src.defenses.tri_llm_v2.tri_llm_v2 import TriDefenseV2
from src.attacks.eia_try import MirrorInjectionAttack


class AgentSrcType(str, Enum):
    HTML = "html"
    A11Y = "a11y"
    SCREENSHOT = "screenshot"


class DefenseMode(Enum):
    NONE = None
    ONE_STAGE = OneStageDefense
    STRICTED_ONE_STAGE = StrictedOneStageDefense
    STRICTED_OWNERS = StrictedDefense
    DUAL_DEFENSE = DualDefense
    DUAL_DEFENSE_V2 = DualDefenseV2
    TRI_DEFENSE = TriDefense
    TRI_DEFENSE_V2 = TriDefenseV2

    def __str__(self) -> str:
        if self.value is None:
            return "none"
        try:
            return self.value.get_defense_id()
        except AttributeError:
            return str(self.value)

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def from_str(cls, value: str) -> "DefenseMode":
        for mode in cls:
            s = str(mode)
            if s == value or s.startswith(value):
                return mode
        raise ValueError(f"Invalid defense mode: {value}")


class AttackMode(Enum):
    NONE = None
    EDA_V2 = EDAAttackV2
    EIA = EIAAttackv2
    POPUP = PopupAttack
    EIA_VISIBLE = EIAAttackv2
    EIA_INPUT_FIELD = EIAInputField
    FAKE_COMP = FakeCompletion
    IGNORE_INSTRUCTION = IgnoreInstruction
    INFEASIBLE = Infeasible
    DISTRACTING_BANNER_AD = DistractingBannerAd
    MIRROR = MirrorInjectionAttack

    def __str__(self) -> str:
        if self.value is None:
            return "none"
        try:
            return self.value.get_attack_id()
        except AttributeError:
            return str(self.value)

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def from_str(cls, value: str) -> "AttackMode":
        for mode in cls:
            if str(mode) == value:
                return mode
        raise ValueError(f"Invalid attack mode: {value}")


class LanguageModel(str, Enum):
    """Available language models for use in the system."""

    GPT4_MINI = "gpt-4.1-mini"
    GPT5_MINI = "gpt-5-mini"
    GPT5 = "gpt-5"
    GEMINI_20_FLASH = "vertex_ai/gemini-2.0-flash"
    GEMINI_20_FLASH_LITE = "vertex_ai/gemini-2.0-flash-lite"
    GEMINI_25_FLASH = "vertex_ai/gemini-2.5-flash"
    GEMINI_25_FLASH_LITE = "vertex_ai/gemini-2.5-flash-lite"
    GEMINI_25_PRO = "vertex_ai/gemini-2.5-pro"

    @classmethod
    def from_str(cls, value: str) -> "LanguageModel":
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"Invalid language model: {value}")


class ModelSamplingParams(BaseModel):
    model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None

    def model_dump(self, **kwargs):
        if "exclude_defaults" not in kwargs:
            kwargs["exclude_defaults"] = True
        return super().model_dump(**kwargs)


class WebAgentEnvArgs(BaseModel):
    chat_mode: bool = False
    use_html: bool = False
    use_axtree: bool = False
    use_screenshot: bool = False


class AsyncEngineArgs(BaseModel):
    n_parallel_agents: int = Field(default=3)
    engine_name: str
    max_steps: int
    max_response_length: int
    max_prompt_length: int
    sampling_params: ModelSamplingParams
    rollout_engine: Optional[Any] = None
    config: Dict[str, Any] = Field(default_factory=dict)


class BrowserEnvArgs(BaseModel):
    defense: DefenseMode
    defense_kwargs: Dict[str, Any] = Field(default_factory=dict)
    attack: AttackMode
    attack_kwargs: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("attack", mode="before")
    @classmethod
    def _parse_attack(cls, v: Any) -> Any:
        if isinstance(v, str):
            return AttackMode.from_str(v)
        if isinstance(v, dict):
            aid = v.get("attack_id") or v.get("type")
            if isinstance(aid, str):
                return AttackMode.from_str(aid)
        return v

    @field_validator("defense", mode="before")
    @classmethod
    def _parse_defense(cls, v: Any) -> Any:
        if isinstance(v, str):
            return DefenseMode.from_str(v)
        if isinstance(v, dict):
            did = v.get("defense_id") or v.get("type")
            if isinstance(did, str):
                return DefenseMode.from_str(did)
        return v


class ExperimentConfig(BaseModel):
    experiment_name: str
    experiment_description: str
    tasks: List[str]
    engine_options: AsyncEngineArgs
    env_args: BrowserEnvArgs = Field(
        default_factory=lambda: BrowserEnvArgs(defense=DefenseMode.NONE, attack=AttackMode.NONE)
    )
    agent_env_args: WebAgentEnvArgs = Field(default_factory=WebAgentEnvArgs)
    tokenizer_model_path: str = Field(default="Qwen/Qwen2.5-7B-Instruct-1M")


class DefenseHarnessExperimentConfig(BaseModel):
    experiment_name: str
    experiment_description: str
    defense: DefenseMode
    defense_kwargs: Dict[str, Any] = Field(default_factory=dict)
    reference_experiment_name: str
    n_parallel_agents: int

    @field_validator("defense", mode="before")
    @classmethod
    def _parse_defense(cls, v: Any) -> Any:
        if isinstance(v, str):
            return DefenseMode.from_str(v)
        if isinstance(v, dict):
            did = v.get("defense_id") or v.get("type")
            if isinstance(did, str):
                return DefenseMode.from_str(did)
        return v
