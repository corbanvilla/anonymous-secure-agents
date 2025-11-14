import argparse
import difflib
import importlib.util
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, cast

import numpy as np
from PIL import Image

import gradio as gr

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
from src.db.config import SCREENSHOT_DIR
from src.db.helpers.datasets import (
    get_all_datasets,
    get_dataset_name_for_task_ids,
)
from src.db.helpers.experiments import (
    filter_experiment_names,
    get_all_attacks,
    get_all_defenses,
    load_recent_experiments,
)
from src.db.helpers.trajectories import (
    load_experiment_trajectories,
    load_trajectory,
)
from src.experiments.config import ExperimentConfig
from src.experiments.config.models import AttackMode, DefenseMode

MAX_REQUEST_SLOTS = 10
MAX_STEPS = 15
FIELD_DEFS = [
    ("html_full", "HTML Full"),
    ("full_a11y", "AXTree Full"),
    ("html_attack", "HTML Attack"),
    ("a11y_attack", "AXTree Attack"),
    ("html_censored", "HTML Censored"),
    ("a11y_censored", "AXTree Censored"),
    ("dom_owners", "DOM Owners"),
    ("dom_metadata", "DOM Metadata"),
    ("security_policy", "Security Policy"),
    ("relevance_labels", "Relevance Labels"),
    ("allowed_elements", "Allowed Elements"),
    ("annotated_dom", "Annotated DOM"),
    ("annotated_dom_readable", "Annotated Dom Readable"),
    ("allowed_bids", "Allowed Bids"),
    ("rejected_bids_html", "Rejected Bids HTML"),
    ("rejected_bids_axtree", "Rejected Bids AXTree"),
    ("html_diff", "HTML Diff"),
    ("a11y_diff", "AXTree Diff"),
]

FIELD_NAMES = [name for name, _ in FIELD_DEFS]


def _load_task_configs(package: str, filename: str) -> dict[str, dict]:
    """Load task configs from a package without importing it."""
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        return {}
    path = next(iter(spec.submodule_search_locations)) + f"/{filename}"
    with open(path, "r") as f:
        return {str(cfg["task_id"]): cfg for cfg in json.load(f)}


_WA_CONFIGS = _load_task_configs("webarena", "test.raw.json")
_VWA_CONFIGS = _load_task_configs("visualwebarena", "test_raw.json")

SUCCESS_CRITERIA_MAP: dict[str, dict] = {}
for tid, cfg in _WA_CONFIGS.items():
    SUCCESS_CRITERIA_MAP[f"browsergym/webarena.safe.{tid}"] = cfg.get("eval", {})
for tid, cfg in _VWA_CONFIGS.items():
    SUCCESS_CRITERIA_MAP[f"browsergym/visualwebarena.safe.{tid}"] = cfg.get("eval", {})


def _pretty_json(val):
    try:
        if isinstance(val, (dict, list)):
            obj = val
        else:
            obj = json.loads(val)

        def compact_lists(o):
            if isinstance(o, list):
                if all(isinstance(x, (int, float)) for x in o):
                    return json.dumps(o)
                else:
                    return [compact_lists(x) for x in o]
            elif isinstance(o, dict):
                return {k: compact_lists(v) for k, v in o.items()}
            else:
                return o

        compacted = compact_lists(obj)
        if isinstance(compacted, str):
            return compacted
        else:
            return json.dumps(compacted, indent=2)
    except Exception:
        return str(val)


def _compute_diff(a, b):
    try:
        a_lines = a.splitlines() if isinstance(a, str) else str(a).splitlines()
        b_lines = b.splitlines() if isinstance(b, str) else str(b).splitlines()
        diff = difflib.unified_diff(
            a_lines,
            b_lines,
            fromfile="Original",
            tofile="Censored",
            lineterm="",
        )
        return "\n".join(diff) or "No diff"
    except Exception as e:
        return f"Diff error: {str(e)}"


def _data_to_image(data: Any) -> Optional[Image.Image]:
    """Convert screenshot data to a PIL image if possible."""
    if isinstance(data, np.ndarray):
        return Image.fromarray(data).convert("RGB")
    if isinstance(data, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            return None
    if isinstance(data, str):
        path = Path(data)
        if path.exists() and path.is_file():
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                return None
        # If the path doesn't exist, try loading from SCREENSHOT_DIR using just
        # the filename. This supports cases where only the hashed filename was
        # stored in the database.
        fallback = SCREENSHOT_DIR / path.name
        if fallback.exists() and fallback.is_file():
            try:
                return Image.open(fallback).convert("RGB")
            except Exception:
                return None
    return None


def get_step_images(step) -> tuple[Optional[Image.Image], Optional[Image.Image], Optional[Image.Image]]:
    """Return screenshot, censored screenshot, and attack screenshot images for a step."""
    screenshot_data = step.observation.get("screenshot")
    censored_data = step.observation.get("screenshot_censored")
    attack_data = step.observation.get("screenshot_attack")

    img = _data_to_image(screenshot_data)
    censored_img = _data_to_image(censored_data)
    attack_img = _data_to_image(attack_data)
    return img, censored_img, attack_img


@dataclass
class RequestComponents:
    """Container for a single request's UI components."""

    title: Any
    meta: Any
    messages: Any
    response: Any


@dataclass
class StepComponents:
    tab: Any
    action: Any
    thought: Any
    attack_acc: Any
    attack_requests: list[RequestComponents]
    defense_acc: Any
    defense_requests: list[RequestComponents]
    logs_acc: Any
    attack_logs: Any
    defense_logs: Any
    environment_logs: Any
    screenshot: Any
    screenshot_censored: Any
    screenshot_attack: Any
    data_fields: dict[str, Any]
    has_censored_image: bool = False
    has_attack_image: bool = False
    num_attack_requests: int = 0
    num_defense_requests: int = 0

    def reset_updates(self) -> tuple[Any, ...]:
        """Return gr.update() calls to clear this step's state."""
        self.has_censored_image = False
        self.num_attack_requests = 0
        self.num_defense_requests = 0
        updates = [
            gr.update(visible=False),  # tab
            gr.update(visible=False, value=""),  # action
            gr.update(visible=False, value=""),  # thought
            gr.update(visible=False),  # attack accordion
        ]
        for req in self.attack_requests:
            updates.extend(
                [
                    gr.update(visible=False),
                    gr.update(visible=False, value=""),
                    gr.update(visible=False, value=""),
                    gr.update(visible=False, value=""),
                ]
            )
        updates.append(gr.update(visible=False))  # defense accordion
        for req in self.defense_requests:
            updates.extend(
                [
                    gr.update(visible=False),
                    gr.update(visible=False, value=""),
                    gr.update(visible=False, value=""),
                    gr.update(visible=False, value=""),
                ]
            )
        updates.extend(
            [
                gr.update(visible=False),  # logs accordion
                gr.update(visible=False, value=""),
                gr.update(visible=False, value=""),
                gr.update(visible=False, value=""),
            ]
        )
        updates.extend(gr.update(visible=False, value="") for _ in FIELD_NAMES)
        updates.extend(
            [
                gr.update(visible=False, value=None),
                gr.update(visible=False, value=None),
                gr.update(visible=False, value=None),
            ]
        )
        return tuple(updates)


# Helper to navigate between trajectories


def navigate_trajectory(
    exp,
    current_task,
    direction,
    filter_status="All",
    attack_filter_status="All",
    defense_filter_status="All",
):
    """
    exp: experiment object with attribute valid_task_ids
    current_task: current task ID (string or int)
    direction: +1 for next, -1 for previous
    Returns the next or previous task ID, wrapping around.
    """
    if filter_status == "Successful":
        ids = exp.success_task_ids
    elif filter_status == "Failed":
        ids = exp.failed_task_ids
    else:
        ids = exp.valid_task_ids

    if attack_filter_status == "Successful":
        ids = [tid for tid in ids if tid in exp.attack_success_task_ids]
    elif attack_filter_status == "Failed":
        ids = [tid for tid in ids if tid in exp.attack_failure_task_ids]

    if defense_filter_status == "Successful":
        ids = [tid for tid in ids if tid in exp.defense_success_task_ids]
    elif defense_filter_status == "Failed":
        ids = [tid for tid in ids if tid in exp.defense_failure_task_ids]

    if not ids:
        return None
    try:
        idx = ids.index(current_task)
    except ValueError:
        return ids[0]
    new_idx = (idx + direction) % len(ids)
    return ids[new_idx]


def update_task_dropdown(exp, filter_status, attack_filter_status="All", defense_filter_status="All"):
    """Return gradio update for task dropdown based on filter."""
    if exp is None:
        return gr.update(choices=[], value=None, visible=False)

    if filter_status == "Successful":
        ids = exp.success_task_ids
        prefix = "✅"
    elif filter_status == "Failed":
        ids = exp.failed_task_ids
        prefix = "❌"
    else:
        ids = exp.valid_task_ids

    if attack_filter_status == "Successful":
        ids = [tid for tid in ids if tid in exp.attack_success_task_ids]
    elif attack_filter_status == "Failed":
        ids = [tid for tid in ids if tid in exp.attack_failure_task_ids]

    if defense_filter_status == "Successful":
        ids = [tid for tid in ids if tid in exp.defense_success_task_ids]
    elif defense_filter_status == "Failed":
        ids = [tid for tid in ids if tid in exp.defense_failure_task_ids]

    if filter_status == "All":
        choices = [(f"✅  {tid}" if tid in exp.success_task_ids else f"❌  {tid}", tid) for tid in ids]
    else:
        choices = [(f"{prefix} {tid}", tid) for tid in ids]

    first = ids[0] if ids else None
    return gr.update(choices=choices, value=first, visible=True)


def filter_experiments(
    show_tests,
    selected_user="All",
    selected_model="All",
    selected_attack="Any",
    selected_defense="Any",
    selected_dataset="All",
    show_hidden=False,
    favorites_only=False,
    prefix: str | None = None,
):
    """Return experiment names filtered directly in the database."""

    return filter_experiment_names(
        show_tests=show_tests,
        selected_user=selected_user,
        selected_model=selected_model,
        selected_attack=selected_attack,
        selected_defense=selected_defense,
        selected_dataset=selected_dataset,
        show_hidden=show_hidden,
        favorites_only=favorites_only,
        prefix=prefix,
    )


def update_experiment_dropdown(
    prefix,
    show_tests,
    user,
    model,
    attack,
    defense,
    dataset,
    show_hidden,
    favorites_only,
):
    """Update experiment dropdown choices based on filters and prefix."""
    choices = filter_experiments(
        show_tests,
        user,
        model,
        attack,
        defense,
        dataset,
        show_hidden,
        favorites_only,
        prefix,
    )
    first = choices[0] if choices else None
    return gr.update(choices=choices, value=first)


def handle_load_group(experiment_name):
    """Load experiment and prepare metadata for the UI."""
    (
        exp,
        valid_task_ids,
        incomplete_task_ids,
        success_task_ids,
        failed_task_ids,
        success_rate,
    ) = load_experiment_trajectories(experiment_name)

    if exp is None:
        return (
            None,
            *clear_experiment_state(),
        )

    exp_any = cast(Any, exp)
    exp_any.valid_task_ids = valid_task_ids
    exp_any.success_task_ids = success_task_ids
    exp_any.failed_task_ids = failed_task_ids

    # Determine the first task index to auto-select if available
    first_task = valid_task_ids[0] if valid_task_ids else None
    successful_tasks = len(success_task_ids)
    total_tasks = len(valid_task_ids)
    attack_successes = 0
    attack_success_task_ids = []
    attack_failure_task_ids = []
    for traj in exp.trajectories:
        steps = traj.trajectory.get("steps", [])
        if steps and steps[0].get("info", {}).get("attack_success", False):
            attack_successes += 1
            attack_success_task_ids.append(traj.task_id)
        else:
            attack_failure_task_ids.append(traj.task_id)
    exp_any.attack_success_task_ids = attack_success_task_ids
    exp_any.attack_failure_task_ids = attack_failure_task_ids
    attack_success_rate = attack_successes / total_tasks if total_tasks else 0.0

    defense_successes = 0
    defense_success_task_ids = []
    defense_failure_task_ids = []
    for traj in exp.trajectories:
        steps = traj.trajectory.get("steps", [])
        if steps and steps[0].get("info", {}).get("defense_success", False):
            defense_successes += 1
            defense_success_task_ids.append(traj.task_id)
        else:
            defense_failure_task_ids.append(traj.task_id)
    exp_any.defense_success_task_ids = defense_success_task_ids
    exp_any.defense_failure_task_ids = defense_failure_task_ids
    defense_success_rate = defense_successes / total_tasks if total_tasks else 0.0

    # Format creation timestamp with ordinal day
    created = exp.created_at
    day = created.day
    suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    created_at_str = f"{created.strftime('%B')} {day}{suffix}, {created.year} - {created.hour:02d}:{created.minute:02d}"

    # Set experiment status with color
    status_text = "🟠 Running" if bool(exp.is_running) else "🟢 Complete"

    # Load and validate experiment config with Pydantic
    config = ExperimentConfig.model_validate(exp.config)
    config_json = _pretty_json(exp.config)

    # Create metadata table with additional fields
    attack = cast(AttackMode, config.env_args.attack)
    defense = cast(DefenseMode, config.env_args.defense)
    attack_id = str(attack)
    defense_id = str(defense)
    attack_id_display = "None" if attack_id.lower() == "none" else attack_id
    defense_id_display = "None" if defense_id.lower() == "none" else defense_id
    dataset_name = get_dataset_name_for_task_ids(cast(list[str], exp.task_ids))
    if dataset_name is None:
        dataset_name = "Unknown"

    metadata = [
        ["User", str(exp.username)],
        ["Created At", created_at_str],
        ["Engine Model", config.engine_options.sampling_params.model],
        ["Dataset", dataset_name],
        ["Use HTML", str(config.agent_env_args.use_html)],
        ["Use A11y Tree", str(config.agent_env_args.use_axtree)],
        ["Use Screenshot", str(config.agent_env_args.use_screenshot)],
        ["Defense ID", defense_id_display],
        ["Attack ID", attack_id_display],
    ]

    all_choices = [(f"✅ {tid}" if tid in success_task_ids else f"❌ {tid}", tid) for tid in valid_task_ids]

    # Show N/A for attack/defense success rate if not enabled
    if attack_id.lower() == "none":
        attack_success_rate_str = "Attack Success Rate: N/A"
    else:
        attack_success_rate_str = f"Attack Success Rate: {attack_success_rate:.2%} ({attack_successes}/{total_tasks})"
    if defense_id.lower() == "none":
        defense_success_rate_str = "Defense Success Rate: N/A"
    else:
        defense_success_rate_str = (
            f"Defense Success Rate: {defense_success_rate:.2%} ({defense_successes}/{total_tasks})"
        )

    return (
        exp,
        gr.update(value="All", visible=True),
        gr.update(value="All", visible=True),
        gr.update(value="All", visible=True),
        gr.update(choices=all_choices, value=first_task, visible=True),
        gr.update(value=status_text, visible=True),
        gr.update(value=str(exp.description) if exp.description is not None else "", visible=True),
        gr.update(value=metadata, visible=True),
        gr.update(value=f"Task Success Rate: {success_rate:.2%} ({successful_tasks}/{total_tasks})", visible=True),
        gr.update(value=attack_success_rate_str, visible=True),
        gr.update(value=defense_success_rate_str, visible=True),
        gr.update(
            label=f"Missing or Incomplete Trajectories ({len(incomplete_task_ids)})",
            value=", ".join(incomplete_task_ids) if incomplete_task_ids else "None",
            visible=True,
        ),
        gr.update(value=config_json, visible=True),
    )


def clear_experiment_state():
    """Clear all experiment-related state."""
    return (
        gr.update(value="All", visible=False),  # clear task filter
        gr.update(value="All", visible=False),  # clear attack success filter
        gr.update(value="All", visible=False),  # clear defense success filter
        gr.update(choices=[], value=None, visible=False),  # clear task index dropdown
        gr.update(value="", visible=False),  # clear experiment status
        gr.update(value="", visible=False),  # clear experiment description
        gr.update(value=None, visible=False),  # clear metadata table
        gr.update(value="", visible=False),  # clear success rate
        gr.update(value="", visible=False),  # clear attack success rate
        gr.update(value="", visible=False),  # clear defense success rate
        gr.update(value="", visible=False),  # clear missing/incomplete text box
        gr.update(value="", visible=False),  # clear experiment config
    )


class TrajectoryVisualizer:
    def __init__(self):
        self.step_components: list[StepComponents] = []
        self.current_step_count = 0
        self.step_tabs: Optional[Any] = None

    def step_output_components(self) -> list[Any]:
        """Return components corresponding to each step in output order."""
        return [
            comp
            for sc in self.step_components
            for comp in (
                sc.tab,
                sc.action,
                sc.thought,
                sc.attack_acc,
                *[
                    c
                    for req in sc.attack_requests
                    for c in (
                        req.title,
                        req.meta,
                        req.messages,
                        req.response,
                    )
                ],
                sc.defense_acc,
                *[
                    c
                    for req in sc.defense_requests
                    for c in (
                        req.title,
                        req.meta,
                        req.messages,
                        req.response,
                    )
                ],
                sc.logs_acc,
                sc.attack_logs,
                sc.defense_logs,
                sc.environment_logs,
                *[sc.data_fields[fname] for fname in FIELD_NAMES],
                sc.screenshot,
                sc.screenshot_censored,
                sc.screenshot_attack,
            )
        ]

    def step_detail_components(self) -> list[Any]:
        """Components for each step excluding the Tab objects."""
        return [
            comp
            for sc in self.step_components
            for comp in (
                sc.action,
                sc.thought,
                sc.attack_acc,
                *[
                    c
                    for req in sc.attack_requests
                    for c in (
                        req.title,
                        req.meta,
                        req.messages,
                        req.response,
                    )
                ],
                sc.defense_acc,
                *[
                    c
                    for req in sc.defense_requests
                    for c in (
                        req.title,
                        req.meta,
                        req.messages,
                        req.response,
                    )
                ],
                sc.logs_acc,
                sc.attack_logs,
                sc.defense_logs,
                sc.environment_logs,
                *[sc.data_fields[fname] for fname in FIELD_NAMES],
                sc.screenshot,
                sc.screenshot_censored,
                sc.screenshot_attack,
            )
        ]

    @staticmethod
    def _extract_step_data(obs) -> dict[str, str]:
        """Return formatted data fields for a single observation."""
        html_full = obs.get("html", "No HTML data")
        html_censored = obs.get("html_censored", "No HTML censored data")
        a11y_full = obs.get("a11y", "No AXTree data")
        a11y_censored = obs.get("a11y_censored", "No AXTree censored data")

        return {
            "full_a11y": a11y_full,
            "html_full": html_full,
            "html_censored": html_censored,
            "a11y_censored": a11y_censored,
            "html_attack": obs.get("html_attack", "No HTML attack data"),
            "a11y_attack": obs.get("a11y_attack", "No AXTree attack data"),
            "dom_owners": _pretty_json(obs.get("dom_owners", "No dom_owners data")),
            "dom_metadata": _pretty_json(obs.get("dom_metadata", "No dom_metadata data")),
            "security_policy": _pretty_json(obs.get("security_policy", "No security_policy data")),
            "relevance_labels": _pretty_json(obs.get("relevance_labels", "No relevance_labels data")),
            "allowed_elements": _pretty_json(obs.get("allowed_elements", "No allowed_elements data")),
            "annotated_dom": _pretty_json(obs.get("annotated_dom", "No annotated_dom data")),
            "annotated_dom_readable": _pretty_json(obs.get("annotated_dom_readable", "No annotated_dom_readable data")),
            "allowed_bids": _pretty_json(obs.get("allowed_bids", "No allowed_bids data")),
            "rejected_bids_html": _pretty_json(obs.get("rejected_bids_html", "No rejected_bids_html data")),
            "rejected_bids_axtree": _pretty_json(obs.get("rejected_bids_axtree", "No rejected_bids_axtree data")),
            "html_diff": _compute_diff(html_full, html_censored),
            "a11y_diff": _compute_diff(a11y_full, a11y_censored),
        }

    def clear_main_column(self):
        loading = "Loading..."
        updates = [
            gr.update(selected=0) if self.step_tabs is not None else gr.update(),
            gr.update(value=loading),  # url
            gr.update(value=loading),  # task
            gr.update(value=loading),  # success criteria
            gr.update(value=loading),  # chat log
            gr.update(value=loading),  # task success
            gr.update(value=loading),  # reward
            gr.update(value=loading),  # attack success
            gr.update(value=loading),  # defense success
            gr.update(value=loading),  # termination reason
        ]
        self.current_step_count = 0
        # Hide all step tabs
        for comp in self.step_components[:MAX_STEPS]:
            updates.extend(comp.reset_updates())
        return tuple(updates)

    def update_step_tabs(
        self,
        traj_steps,
        field_vis: dict[str, bool],
        show_actions: bool = True,
        show_thoughts: bool = True,
        show_screenshot: bool = True,
        show_screenshot_censored: bool = True,
        show_screenshot_attack: bool = True,
    ):
        """Update step tabs with trajectory data."""
        self.current_step_count = len(traj_steps)
        updates = [gr.update(selected=0) if self.step_tabs is not None else gr.update()]
        for i, comp in enumerate(self.step_components[:MAX_STEPS]):
            if i < self.current_step_count:
                step = traj_steps[i]
                obs = step.observation
                data_vals = self._extract_step_data(obs)
                screenshot, screenshot_censored, screenshot_attack = get_step_images(step)
                comp.has_censored_image = screenshot_censored is not None
                comp.has_attack_image = screenshot_attack is not None
                attack_requests = step.info.get("attack_requests", []) or []
                defense_requests = step.info.get("defense_requests", []) or []
                comp.num_attack_requests = len(attack_requests)
                comp.num_defense_requests = len(defense_requests)
                updates.extend(
                    [
                        gr.update(visible=True),
                        gr.update(value=step.action or "", visible=show_actions),
                        gr.update(value=step.thought or "", visible=show_thoughts),
                        gr.update(visible=bool(attack_requests) and show_thoughts),
                    ]
                )
                for req_idx, req in enumerate(comp.attack_requests):
                    if req_idx < len(attack_requests):
                        r = attack_requests[req_idx]
                        msgs = []
                        for m in r.get("messages", []):
                            mc = m.copy()
                            if isinstance(mc.get("content"), str) and len(mc["content"]) > 1000:
                                mc["content"] = mc["content"][:1000] + "..."
                            msgs.append(mc)
                        meta = {
                            "model": r.get("model"),
                            "callee": r.get("callee"),
                        }
                        usage = r.get("response", {}).get("usage", {})
                        meta["input_tokens"] = usage.get("prompt_tokens")
                        meta["output_tokens"] = usage.get("completion_tokens")
                        dur = r.get("request_duration")
                        if dur is None:
                            start = r.get("request_start")
                            end = r.get("request_end")
                            if start is not None and end is not None:
                                dur = end - start
                        meta["duration_seconds"] = round(dur, 2) if dur is not None else None
                        updates.extend(
                            [
                                gr.update(visible=True),
                                gr.update(value=_pretty_json(meta), visible=True),
                                gr.update(value=_pretty_json(msgs), visible=True),
                                gr.update(
                                    value=_pretty_json(
                                        (
                                            r.get("response", {})
                                            .get("choices", [{}])[0]
                                            .get("message", {})
                                            .get("content", "")
                                        )
                                    ),
                                    visible=True,
                                ),
                            ]
                        )
                    else:
                        updates.extend(
                            [
                                gr.update(visible=False),
                                gr.update(visible=False, value=""),
                                gr.update(visible=False, value=""),
                                gr.update(visible=False, value=""),
                            ]
                        )
                updates.append(gr.update(visible=bool(defense_requests) and show_thoughts))
                for req_idx, req in enumerate(comp.defense_requests):
                    if req_idx < len(defense_requests):
                        r = defense_requests[req_idx]
                        msgs = []
                        for m in r.get("messages", []):
                            mc = m.copy()
                            if isinstance(mc.get("content"), str) and len(mc["content"]) > 1000:
                                mc["content"] = mc["content"][:1000] + "..."
                            msgs.append(mc)
                        meta = {
                            "model": r.get("model"),
                            "callee": r.get("callee"),
                        }
                        usage = r.get("response", {}).get("usage", {})
                        meta["input_tokens"] = usage.get("prompt_tokens")
                        meta["output_tokens"] = usage.get("completion_tokens")
                        dur = r.get("request_duration")
                        if dur is None:
                            start = r.get("request_start")
                            end = r.get("request_end")
                            if start is not None and end is not None:
                                dur = end - start
                        meta["duration_seconds"] = round(dur, 2) if dur is not None else None
                        updates.extend(
                            [
                                gr.update(visible=True),
                                gr.update(value=_pretty_json(meta), visible=True),
                                gr.update(value=_pretty_json(msgs), visible=True),
                                gr.update(
                                    value=_pretty_json(
                                        (
                                            r.get("response", {})
                                            .get("choices", [{}])[0]
                                            .get("message", {})
                                            .get("content", "")
                                        )
                                    ),
                                    visible=True,
                                ),
                            ]
                        )
                    else:
                        updates.extend(
                            [
                                gr.update(visible=False),
                                gr.update(visible=False, value=""),
                                gr.update(visible=False, value=""),
                                gr.update(visible=False, value=""),
                            ]
                        )
                updates.extend(
                    [
                        gr.update(visible=show_thoughts),
                        gr.update(
                            value=_pretty_json(step.info.get("attack_logs", "")),
                            visible=show_thoughts,
                        ),
                        gr.update(
                            value=_pretty_json(step.info.get("defense_logs", "")),
                            visible=show_thoughts,
                        ),
                        gr.update(
                            value=_pretty_json(step.info.get("environment_logs", "")),
                            visible=show_thoughts,
                        ),
                    ]
                )
                for fname in FIELD_NAMES:
                    updates.append(
                        gr.update(
                            value=data_vals[fname],
                            visible=field_vis.get(fname, False),
                        )
                    )
                updates.extend(
                    [
                        gr.update(value=screenshot, visible=show_screenshot),
                        gr.update(
                            value=screenshot_censored,
                            visible=show_screenshot_censored and screenshot_censored is not None,
                        ),
                        gr.update(
                            value=screenshot_attack,
                            visible=show_screenshot_attack and screenshot_attack is not None,
                        ),
                    ]
                )
            else:
                updates.extend(comp.reset_updates())
        return tuple(updates)

    def apply_view_config(
        self,
        show_actions: bool,
        show_thoughts: bool,
        show_screenshot: bool,
        show_screenshot_censored: bool,
        show_screenshot_attack: bool,
        *field_vis_flags: bool,
    ):
        """Return visibility updates for current step components."""
        updates: list[Any] = []
        vis_map = dict(zip(FIELD_NAMES, field_vis_flags))
        for i, comp in enumerate(self.step_components[:MAX_STEPS]):
            visible = i < self.current_step_count
            updates.extend(
                [
                    gr.update(visible=visible and show_actions),
                    gr.update(visible=visible and show_thoughts),
                    gr.update(visible=visible and show_thoughts and comp.num_attack_requests > 0),
                ]
            )
            for req_idx, _ in enumerate(comp.attack_requests):
                req_vis = visible and show_thoughts and req_idx < comp.num_attack_requests
                updates.extend(
                    [
                        gr.update(visible=req_vis),
                        gr.update(visible=req_vis),
                        gr.update(visible=req_vis),
                        gr.update(visible=req_vis),
                    ]
                )
            updates.append(gr.update(visible=visible and show_thoughts and comp.num_defense_requests > 0))
            for req_idx, _ in enumerate(comp.defense_requests):
                req_vis = visible and show_thoughts and req_idx < comp.num_defense_requests
                updates.extend(
                    [
                        gr.update(visible=req_vis),
                        gr.update(visible=req_vis),
                        gr.update(visible=req_vis),
                        gr.update(visible=req_vis),
                    ]
                )
            updates.extend(
                [
                    gr.update(visible=visible and show_thoughts),
                    gr.update(visible=visible and show_thoughts),
                    gr.update(visible=visible and show_thoughts),
                    gr.update(visible=visible and show_thoughts),
                ]
            )
            for fname in FIELD_NAMES:
                updates.append(gr.update(visible=visible and vis_map.get(fname, False)))
            updates.extend(
                [
                    gr.update(visible=visible and show_screenshot),
                    gr.update(
                        visible=visible and show_screenshot_censored and comp.has_censored_image,
                    ),
                    gr.update(
                        visible=visible and show_screenshot_attack and comp.has_attack_image,
                    ),
                ]
            )

        expected_len = len(self.step_detail_components())
        if len(updates) > expected_len:
            updates = updates[:expected_len]
        elif len(updates) < expected_len:
            updates.extend(gr.update() for _ in range(expected_len - len(updates)))
        return tuple(updates)

    def extract_labels_from_trajectory(
        self,
        exp: Optional[Any],
        task_id: Optional[str],
        show_actions: bool = True,
        show_thoughts: bool = True,
        show_screenshot: bool = True,
        show_screenshot_censored: bool = True,
        show_screenshot_attack: bool = True,
        *field_vis_flags: bool,
    ) -> Tuple[Any, ...]:
        """Extract labels from a trajectory for visualization.

        Args:
            exp: The experiment object or None
            task_id: The task ID string or None

        Returns:
            A tuple containing the visualization updates
        """
        if not exp or not task_id:
            return self.clear_main_column()

        traj, chat_log = load_trajectory(exp.name, task_id)
        if traj is None:
            return self.clear_main_column()

        url = traj.steps[0].observation["url"]
        task = traj.steps[0].observation["goal"]

        reward = str(traj.steps[-1].reward)
        termination_reason = traj.termination_reason or ""  # Convert None to empty string

        # Task success: reward == 1
        task_success_val = str(traj.steps[-1].reward == 1)

        attack_defense_step = traj.steps[0]
        # Attack success: info["attack_success"] if present
        attack_success_val = str(attack_defense_step.info.get("attack_success", False))
        # Defense success: use info value if present, otherwise inverse of attack success
        defense_success_val = str(attack_defense_step.info.get("defense_success", False))

        field_vis = dict(zip(FIELD_NAMES, field_vis_flags))

        step_updates = self.update_step_tabs(
            traj.steps,
            field_vis,
            show_actions=show_actions,
            show_thoughts=show_thoughts,
            show_screenshot=show_screenshot,
            show_screenshot_censored=show_screenshot_censored,
            show_screenshot_attack=show_screenshot_attack,
        )

        # The update tuple must match the output order expected by the event
        # handlers: the first element corresponds to the step tab component,
        # followed by task related text boxes. Returning updates in this order
        # prevents mismatches that previously caused invalid keyword arguments
        # like ``selected`` to be sent to non-tab components.
        success_cfg = SUCCESS_CRITERIA_MAP.get(task_id)
        success_str = json.dumps(success_cfg, indent=2) if success_cfg else ""
        return (
            step_updates[0],
            gr.update(value=url),
            gr.update(value=task),
            gr.update(value=success_str),
            gr.update(value=chat_log or ""),
            gr.update(value=task_success_val),
            gr.update(value=reward),
            gr.update(value=attack_success_val),
            gr.update(value=defense_success_val),
            gr.update(value=termination_reason),
            *step_updates[1:],
        )


def main(share: bool = True, server_port: int = 9781):
    """
    Load trajectory data from the given save path and launch a Gradio interface
    for visualizing different trajectories.

    Args:
        share (bool): Whether to share the interface publicly
        server_port (int): Port to run the server on
    """
    trajs: list[dict] = []
    visualizer = TrajectoryVisualizer()

    with gr.Blocks() as demo:
        # Add custom CSS for code box max height
        gr.HTML(
            """
        <style>
        .step-code-box .wrap {
            max-height: 500px !important;
            overflow-y: auto !important;
        }
        .chat-code-box .wrap {
            max-height: 600px !important;
            overflow-y: auto !important;
        }
        .chat-code-box-half .wrap {
            max-height: 300px !important;
            overflow-y: auto !important;
        }
        .request-code-box .wrap {
            max-height: 300px !important;
            overflow-y: auto !important;
        }
        </style>
        """
        )
        trajs_state = gr.State(trajs)
        show_tests_state = gr.State(True)
        selected_user_state = gr.State("All")
        selected_model_state = gr.State("All")
        selected_attack_state = gr.State("Any")
        selected_defense_state = gr.State("Any")
        selected_dataset_state = gr.State("All")
        show_hidden_state = gr.State(False)
        favorites_only_state = gr.State(False)

        # Load recent experiment group IDs and users for dropdowns
        all_experiments = load_recent_experiments()
        # Build user list with "All" default
        user_list = sorted({cast(Any, exp).username for exp in all_experiments})
        user_choices = ["All"] + user_list
        # Build engine model list with "All" default, skipping invalid configs
        engine_model_set = set()
        for exp in all_experiments:
            try:
                config = ExperimentConfig.model_validate(exp.config)
                model_name = config.engine_options.sampling_params.model
            except Exception:
                # Skip experiments with invalid config
                continue
            engine_model_set.add(model_name)
        engine_models = sorted(engine_model_set)
        engine_choices = ["All"] + engine_models

        attack_choices_db = get_all_attacks()
        defense_choices_db = get_all_defenses()
        dataset_choices_db = get_all_datasets()
        attack_choices = ["Any", "None"] + attack_choices_db
        defense_choices = ["Any", "None"] + defense_choices_db
        dataset_choices = ["All", "Unknown"] + dataset_choices_db

        filtered_experiments = filter_experiments(
            show_tests_state.value,
            selected_user_state.value,
            selected_model_state.value,
            selected_attack_state.value,
            selected_defense_state.value,
            selected_dataset_state.value,
            show_hidden_state.value,
            favorites_only_state.value,
            "",
        )
        default_choice = filtered_experiments[0] if filtered_experiments else None

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Experiments")
                refresh_button = gr.Button("Refresh Latest Experiment", variant="secondary")
                gr.Markdown("### Filter Experiments")

                user_filter = gr.Dropdown(label="User", choices=user_choices, value="All", interactive=True)
                model_filter = gr.Dropdown(label="Model", choices=engine_choices, value="All", interactive=True)
                attack_filter_exp = gr.Dropdown(label="Attack", choices=attack_choices, value="Any", interactive=True)
                defense_filter_exp = gr.Dropdown(
                    label="Defense", choices=defense_choices, value="Any", interactive=True
                )
                dataset_filter = gr.Dropdown(label="Dataset", choices=dataset_choices, value="All", interactive=True)
                show_tests = gr.Checkbox(label="Include Tests", value=True)
                show_hidden_cb = gr.Checkbox(label="Show Hidden", value=False)
                favorites_only_cb = gr.Checkbox(label="Favorites Only", value=False)

                gr.Markdown("### Select an Experiment")
                experiment_prefix = gr.Textbox(
                    label=None,
                    placeholder="Search experiments",
                    lines=1,
                    interactive=True,
                    show_label=False,
                )
                experiments_dropdown = gr.Dropdown(
                    label="Experiment Name",
                    choices=filtered_experiments,
                    value=default_choice,
                    interactive=True,
                )

                gr.Markdown("### Experiment Details")
                experiment_status = gr.Textbox(label="Experiment Status", lines=1, visible=False)
                experiment_description = gr.Textbox(label="Experiment Description", max_lines=2, visible=False)
                gr.Markdown("### Performance")
                success_rate = gr.Textbox(show_label=False, lines=1, visible=False)
                attack_success_rate = gr.Textbox(show_label=False, lines=1, visible=False)
                defense_success_rate = gr.Textbox(show_label=False, lines=1, visible=False)

                gr.Markdown("### Missing or Incomplete Trajectories")
                failed_trajectories = gr.Textbox(show_label=False, max_lines=4, visible=False)

                metadata_table = gr.Dataframe(
                    datatype=["str", "str"],
                    value=None,
                    headers=["Key", "Value"],
                    interactive=False,
                    label="Experiment Metadata",
                    visible=False,
                )
                experiment_config = gr.Code(
                    label="Experiment Config",
                    language=None,
                    visible=False,
                    elem_classes=["chat-code-box-half"],
                )

            with gr.Column(scale=3):
                gr.Markdown("## Tasks")
                gr.Markdown("### Navigation")
                with gr.Row():
                    gr.Button("Back", variant="primary", scale=1)
                    gr.Button("Next", variant="primary", scale=1)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Filter Tasks")
                        task_filter = gr.Dropdown(
                            label="Task Filter",
                            choices=["All", "Successful", "Failed"],
                            value="All",
                            interactive=True,
                            visible=False,
                        )
                        attack_success_filter = gr.Dropdown(
                            label="Attack Success Filter",
                            choices=["All", "Successful", "Failed"],
                            value="All",
                            interactive=True,
                            visible=False,
                        )
                        defense_success_filter = gr.Dropdown(
                            label="Defense Success Filter",
                            choices=["All", "Successful", "Failed"],
                            value="All",
                            interactive=True,
                            visible=False,
                        )

                        gr.Markdown("### Select a Task")
                        task_idx = gr.Dropdown(label=None, choices=[], value=None, visible=False, show_label=False)

                    with gr.Column(scale=1):
                        gr.Markdown("### Task Success")
                        task_success = gr.Textbox(label="Task Success", lines=1, interactive=False)
                        reward = gr.Textbox(label="Reward", lines=1, interactive=False)
                        attack_success = gr.Textbox(label="Attack Success", lines=1, interactive=False)
                        defense_success = gr.Textbox(label="Defense Success", lines=1, interactive=False)
                        termination_reason_box = gr.Textbox(label="Termination Reason", lines=1, interactive=False)

                    with gr.Column():
                        gr.Markdown("### Configure View")
                        show_actions_cb = gr.Checkbox(label="Actions", value=True)
                        show_thoughts_cb = gr.Checkbox(label="Thoughts", value=True)
                        show_screenshot_cb = gr.Checkbox(label="Screenshot", value=True)
                        show_screenshot_censored_cb = gr.Checkbox(label="Screenshot Censored", value=True)
                        show_screenshot_attack_cb = gr.Checkbox(label="Screenshot Attack", value=True)

                        field_checkboxes = {}
                        with gr.Accordion("Observation", open=False):
                            for fname in [
                                "html_full",
                                "full_a11y",
                            ]:
                                field_checkboxes[fname] = gr.Checkbox(label=dict(FIELD_DEFS)[fname], value=False)
                        with gr.Accordion("Attack", open=False):
                            for fname in [
                                "html_attack",
                                "a11y_attack",
                            ]:
                                field_checkboxes[fname] = gr.Checkbox(label=dict(FIELD_DEFS)[fname], value=False)
                        with gr.Accordion("Defense", open=False):
                            for fname in [
                                "html_censored",
                                "a11y_censored",
                                "dom_owners",
                                "dom_metadata",
                                "security_policy",
                                "relevance_labels",
                                "allowed_elements",
                                "annotated_dom",
                                "annotated_dom_readable",
                                "allowed_bids",
                                "rejected_bids_html",
                                "rejected_bids_axtree",
                                "html_diff",
                                "a11y_diff",
                            ]:
                                field_checkboxes[fname] = gr.Checkbox(label=dict(FIELD_DEFS)[fname], value=False)

                gr.Markdown("### Task Details")
                task = gr.Textbox(label="Task", lines=1)
                url = gr.Textbox(label="Starting URL", lines=1)
                with gr.Accordion("Success Criteria", open=False):
                    success_criteria_box = gr.Code(
                        label="Success Criteria",
                        language="json",
                        visible=True,
                        elem_classes=["chat-code-box-half"],
                    )
                with gr.Accordion("Chat Log", open=False):
                    chat_log = gr.Code(language=None, visible=True, elem_classes=["chat-code-box"])

                gr.Markdown("### Step-by-Step Execution")
                step_tabs = gr.Tabs()
                visualizer.step_tabs = step_tabs
                with step_tabs:
                    # Will be populated dynamically
                    for i in range(MAX_STEPS):  # Pre-create some tabs for reuse
                        with gr.Tab(f"Step {i}", visible=False) as tab:
                            action = gr.Textbox(label="Action", visible=False)
                            thought = gr.Textbox(label="Thought", lines=4, visible=False)
                            with gr.Accordion("Attack Requests", open=False) as attack_acc:
                                attack_reqs = []
                                for j in range(MAX_REQUEST_SLOTS):
                                    title = gr.Markdown(f"### Request {j + 1}", visible=False)
                                    meta_tb = gr.Code(
                                        label="Request",
                                        language="json",
                                        visible=False,
                                        elem_classes=["request-code-box"],
                                        wrap_lines=True,
                                    )
                                    msgs = gr.Code(
                                        label="Messages",
                                        language="json",
                                        visible=False,
                                        elem_classes=["request-code-box"],
                                        wrap_lines=True,
                                    )
                                    resp = gr.Code(
                                        label="Response",
                                        language="json",
                                        visible=False,
                                        elem_classes=["request-code-box"],
                                        wrap_lines=True,
                                    )
                                    attack_reqs.append(RequestComponents(title, meta_tb, msgs, resp))
                            with gr.Accordion("Defense Requests", open=False) as defense_acc:
                                defense_reqs = []
                                for j in range(MAX_REQUEST_SLOTS):
                                    title = gr.Markdown(f"### Request {j + 1}", visible=False)
                                    meta_tb = gr.Code(
                                        label="Request",
                                        language="json",
                                        visible=False,
                                        elem_classes=["request-code-box"],
                                        wrap_lines=True,
                                    )
                                    msgs = gr.Code(
                                        label="Messages",
                                        language="json",
                                        visible=False,
                                        elem_classes=["request-code-box"],
                                        wrap_lines=True,
                                    )
                                    resp = gr.Code(
                                        label="Response",
                                        language="json",
                                        visible=False,
                                        elem_classes=["request-code-box"],
                                        wrap_lines=True,
                                    )
                                    defense_reqs.append(RequestComponents(title, meta_tb, msgs, resp))
                            with gr.Accordion("Logs", open=False) as logs_acc:
                                attack_logs = gr.Code(
                                    label="Attack Logs",
                                    language=None,
                                    visible=False,
                                    elem_classes=["request-code-box"],
                                )
                                defense_logs = gr.Code(
                                    label="Defense Logs",
                                    language=None,
                                    visible=False,
                                    elem_classes=["request-code-box"],
                                )
                                environment_logs = gr.Code(
                                    label="Environment Logs",
                                    language=None,
                                    visible=False,
                                    elem_classes=["request-code-box"],
                                )
                            data_fields = {}
                            for fname, flabel in FIELD_DEFS:
                                data_fields[fname] = gr.Code(
                                    label=flabel,
                                    language=None,
                                    visible=False,
                                    elem_classes=["step-code-box"],
                                )
                            img = gr.Image(label="Screenshot", visible=False)
                            img_censored = gr.Image(label="Screenshot Censored", visible=False)
                            img_attack = gr.Image(label="Screenshot Attack", visible=False)
                            visualizer.step_components.append(
                                StepComponents(
                                    tab,
                                    action,
                                    thought,
                                    attack_acc,
                                    attack_reqs,
                                    defense_acc,
                                    defense_reqs,
                                    logs_acc,
                                    attack_logs,
                                    defense_logs,
                                    environment_logs,
                                    img,
                                    img_censored,
                                    img_attack,
                                    data_fields,
                                )
                            )

                # Update visibility when any checkbox changes
                all_cbs = [
                    show_actions_cb,
                    show_thoughts_cb,
                    show_screenshot_cb,
                    show_screenshot_censored_cb,
                    show_screenshot_attack_cb,
                ] + list(field_checkboxes.values())
                for cb in all_cbs:
                    cb.change(
                        fn=visualizer.apply_view_config,
                        inputs=[
                            show_actions_cb,
                            show_thoughts_cb,
                            show_screenshot_cb,
                            show_screenshot_censored_cb,
                            show_screenshot_attack_cb,
                            *[field_checkboxes[name] for name in FIELD_NAMES],
                        ],
                        outputs=visualizer.step_detail_components(),
                        show_progress="hidden",
                    )

                # Update event handlers
                experiments_dropdown.change(
                    fn=visualizer.clear_main_column,
                    inputs=[],
                    outputs=[
                        step_tabs,
                        url,
                        task,
                        success_criteria_box,
                        chat_log,
                        task_success,
                        reward,
                        attack_success,
                        defense_success,
                        termination_reason_box,
                        *visualizer.step_output_components(),
                    ],
                    show_progress="hidden",
                ).then(
                    fn=clear_experiment_state,
                    inputs=[],
                    outputs=[
                        task_filter,
                        attack_success_filter,
                        defense_success_filter,
                        task_idx,
                        experiment_status,
                        experiment_description,
                        metadata_table,
                        success_rate,
                        attack_success_rate,
                        defense_success_rate,
                        failed_trajectories,
                        experiment_config,
                    ],
                    show_progress="hidden",
                ).then(
                    fn=handle_load_group,
                    inputs=[experiments_dropdown],
                    outputs=[
                        trajs_state,
                        task_filter,
                        attack_success_filter,
                        defense_success_filter,
                        task_idx,
                        experiment_status,
                        experiment_description,
                        metadata_table,
                        success_rate,
                        attack_success_rate,
                        defense_success_rate,
                        failed_trajectories,
                        experiment_config,
                    ],
                    show_progress="hidden",
                )

                # Experiment prefix filtering
                experiment_prefix.input(
                    fn=update_experiment_dropdown,
                    inputs=[
                        experiment_prefix,
                        show_tests,
                        user_filter,
                        model_filter,
                        attack_filter_exp,
                        defense_filter_exp,
                        dataset_filter,
                        show_hidden_cb,
                        favorites_only_cb,
                    ],
                    outputs=[experiments_dropdown],
                    show_progress="hidden",
                )

                user_filter.change(
                    fn=update_experiment_dropdown,
                    inputs=[
                        experiment_prefix,
                        show_tests,
                        user_filter,
                        model_filter,
                        attack_filter_exp,
                        defense_filter_exp,
                        dataset_filter,
                        show_hidden_cb,
                        favorites_only_cb,
                    ],
                    outputs=[experiments_dropdown],
                    show_progress="hidden",
                )
                model_filter.change(
                    fn=update_experiment_dropdown,
                    inputs=[
                        experiment_prefix,
                        show_tests,
                        user_filter,
                        model_filter,
                        attack_filter_exp,
                        defense_filter_exp,
                        dataset_filter,
                        show_hidden_cb,
                        favorites_only_cb,
                    ],
                    outputs=[experiments_dropdown],
                    show_progress="hidden",
                )
                show_tests.change(
                    fn=update_experiment_dropdown,
                    inputs=[
                        experiment_prefix,
                        show_tests,
                        user_filter,
                        model_filter,
                        attack_filter_exp,
                        defense_filter_exp,
                        dataset_filter,
                        show_hidden_cb,
                        favorites_only_cb,
                    ],
                    outputs=[experiments_dropdown],
                    show_progress="hidden",
                )
                attack_filter_exp.change(
                    fn=update_experiment_dropdown,
                    inputs=[
                        experiment_prefix,
                        show_tests,
                        user_filter,
                        model_filter,
                        attack_filter_exp,
                        defense_filter_exp,
                        dataset_filter,
                        show_hidden_cb,
                        favorites_only_cb,
                    ],
                    outputs=[experiments_dropdown],
                    show_progress="hidden",
                )
                defense_filter_exp.change(
                    fn=update_experiment_dropdown,
                    inputs=[
                        experiment_prefix,
                        show_tests,
                        user_filter,
                        model_filter,
                        attack_filter_exp,
                        defense_filter_exp,
                        dataset_filter,
                        show_hidden_cb,
                        favorites_only_cb,
                    ],
                    outputs=[experiments_dropdown],
                    show_progress="hidden",
                )
                dataset_filter.change(
                    fn=update_experiment_dropdown,
                    inputs=[
                        experiment_prefix,
                        show_tests,
                        user_filter,
                        model_filter,
                        attack_filter_exp,
                        defense_filter_exp,
                        dataset_filter,
                        show_hidden_cb,
                        favorites_only_cb,
                    ],
                    outputs=[experiments_dropdown],
                    show_progress="hidden",
                )
                show_hidden_cb.change(
                    fn=update_experiment_dropdown,
                    inputs=[
                        experiment_prefix,
                        show_tests,
                        user_filter,
                        model_filter,
                        attack_filter_exp,
                        defense_filter_exp,
                        dataset_filter,
                        show_hidden_cb,
                        favorites_only_cb,
                    ],
                    outputs=[experiments_dropdown],
                    show_progress="hidden",
                )
                favorites_only_cb.change(
                    fn=update_experiment_dropdown,
                    inputs=[
                        experiment_prefix,
                        show_tests,
                        user_filter,
                        model_filter,
                        attack_filter_exp,
                        defense_filter_exp,
                        dataset_filter,
                        show_hidden_cb,
                        favorites_only_cb,
                    ],
                    outputs=[experiments_dropdown],
                    show_progress="hidden",
                )

                refresh_button.click(
                    fn=update_experiment_dropdown,
                    inputs=[
                        experiment_prefix,
                        show_tests,
                        user_filter,
                        model_filter,
                        attack_filter_exp,
                        defense_filter_exp,
                        dataset_filter,
                        show_hidden_cb,
                        favorites_only_cb,
                    ],
                    outputs=[experiments_dropdown],
                    show_progress="hidden",
                )

                # Update task_idx change handler
                task_idx.change(
                    fn=visualizer.clear_main_column,
                    inputs=[],
                    outputs=[
                        step_tabs,
                        url,
                        task,
                        success_criteria_box,
                        chat_log,
                        task_success,
                        reward,
                        attack_success,
                        defense_success,
                        termination_reason_box,
                        *visualizer.step_output_components(),
                    ],
                    show_progress="hidden",
                ).then(
                    fn=visualizer.extract_labels_from_trajectory,
                    inputs=[
                        trajs_state,
                        task_idx,
                        show_actions_cb,
                        show_thoughts_cb,
                        show_screenshot_cb,
                        show_screenshot_censored_cb,
                        show_screenshot_attack_cb,
                        *[field_checkboxes[name] for name in FIELD_NAMES],
                    ],
                    outputs=[
                        step_tabs,
                        url,
                        task,
                        success_criteria_box,
                        chat_log,
                        task_success,
                        reward,
                        attack_success,
                        defense_success,
                        termination_reason_box,
                        *visualizer.step_output_components(),
                    ],
                    show_progress="hidden",
                )

                # Update other event handlers
                task_filter.change(
                    fn=update_task_dropdown,
                    inputs=[trajs_state, task_filter, attack_success_filter, defense_success_filter],
                    outputs=[task_idx],
                    show_progress="hidden",
                ).then(
                    fn=visualizer.clear_main_column,
                    inputs=[],
                    outputs=[
                        step_tabs,
                        url,
                        task,
                        success_criteria_box,
                        chat_log,
                        task_success,
                        reward,
                        attack_success,
                        defense_success,
                        termination_reason_box,
                        *visualizer.step_output_components(),
                    ],
                    show_progress="hidden",
                ).then(
                    fn=visualizer.extract_labels_from_trajectory,
                    inputs=[
                        trajs_state,
                        task_idx,
                        show_actions_cb,
                        show_thoughts_cb,
                        show_screenshot_cb,
                        show_screenshot_censored_cb,
                        show_screenshot_attack_cb,
                        *[field_checkboxes[name] for name in FIELD_NAMES],
                    ],
                    outputs=[
                        step_tabs,
                        url,
                        task,
                        success_criteria_box,
                        chat_log,
                        task_success,
                        reward,
                        attack_success,
                        defense_success,
                        termination_reason_box,
                        *visualizer.step_output_components(),
                    ],
                    show_progress="hidden",
                )

                attack_success_filter.change(
                    fn=update_task_dropdown,
                    inputs=[trajs_state, task_filter, attack_success_filter, defense_success_filter],
                    outputs=[task_idx],
                    show_progress="hidden",
                ).then(
                    fn=visualizer.clear_main_column,
                    inputs=[],
                    outputs=[
                        step_tabs,
                        url,
                        task,
                        success_criteria_box,
                        chat_log,
                        task_success,
                        reward,
                        attack_success,
                        defense_success,
                        termination_reason_box,
                        *visualizer.step_output_components(),
                    ],
                    show_progress="hidden",
                ).then(
                    fn=visualizer.extract_labels_from_trajectory,
                    inputs=[
                        trajs_state,
                        task_idx,
                        show_actions_cb,
                        show_thoughts_cb,
                        show_screenshot_cb,
                        show_screenshot_censored_cb,
                        show_screenshot_attack_cb,
                        *[field_checkboxes[name] for name in FIELD_NAMES],
                    ],
                    outputs=[
                        step_tabs,
                        url,
                        task,
                        success_criteria_box,
                        chat_log,
                        task_success,
                        reward,
                        attack_success,
                        defense_success,
                        termination_reason_box,
                        *visualizer.step_output_components(),
                    ],
                    show_progress="hidden",
                )

        demo.load(
            fn=visualizer.clear_main_column,
            inputs=[],
            outputs=[
                step_tabs,
                url,
                task,
                success_criteria_box,
                chat_log,
                task_success,
                reward,
                attack_success,
                defense_success,
                termination_reason_box,
                *visualizer.step_output_components(),
            ],
            show_progress="hidden",
        ).then(
            fn=clear_experiment_state,
            inputs=[],
            outputs=[
                task_filter,
                attack_success_filter,
                defense_success_filter,
                task_idx,
                experiment_status,
                experiment_description,
                metadata_table,
                success_rate,
                attack_success_rate,
                defense_success_rate,
                failed_trajectories,
                experiment_config,
            ],
            show_progress="hidden",
        ).then(
            fn=handle_load_group,
            inputs=[experiments_dropdown],
            outputs=[
                trajs_state,
                task_filter,
                attack_success_filter,
                defense_success_filter,
                task_idx,
                experiment_status,
                experiment_description,
                metadata_table,
                success_rate,
                attack_success_rate,
                defense_success_rate,
                failed_trajectories,
                experiment_config,
            ],
            show_progress="hidden",
        ).then(
            fn=visualizer.extract_labels_from_trajectory,
            inputs=[
                trajs_state,
                task_idx,
                show_actions_cb,
                show_thoughts_cb,
                show_screenshot_cb,
                show_screenshot_censored_cb,
                show_screenshot_attack_cb,
                *[field_checkboxes[name] for name in FIELD_NAMES],
            ],
            outputs=[
                step_tabs,
                url,
                task,
                success_criteria_box,
                chat_log,
                task_success,
                reward,
                attack_success,
                defense_success,
                termination_reason_box,
                *visualizer.step_output_components(),
            ],
            show_progress="hidden",
        )

        demo.launch(
            share=share,
            server_port=server_port,
            server_name="127.0.0.1",
            _frontend=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch trajectory safety visualizer")
    parser.add_argument(
        "--no-share",
        action="store_false",
        dest="share",
        help="Disable sharing the interface publicly (default: shared)",
    )
    parser.add_argument("--port", type=int, default=9781, help="Port to run the server on (default: 9781)")
    args = parser.parse_args()
    main(share=args.share, server_port=args.port)
