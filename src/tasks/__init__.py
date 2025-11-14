import importlib.resources
import json
import random
from typing import Dict, List, Set, Type

import playwright.sync_api
import visualwebarena
import webarena
from browsergym.visualwebarena import config as vwa_config
from browsergym.visualwebarena import task as vwa_task
from browsergym.webarena import config as wa_config
from browsergym.webarena import task as wa_task

from src.db.helpers.datasets import create_task_dataset_if_not_exists
from src.tasks.evaluator import evaluator_router
from src.tasks.presets.easy import WA_EASY_20, WA_EASY_ALL
from src.tasks.registration import register_safe_env_task


# Load configuration files
def load_config(package, filename: str) -> List[dict]:
    config_str = importlib.resources.files(package).joinpath(filename).read_text()
    return json.loads(config_str)


WA_TASK_CONFIGS = load_config(webarena, "test.raw.json")
VWA_TASK_CONFIGS = load_config(visualwebarena, "test_raw.json")


# Base configuration for both WA and VWA
class BaseConfig:
    def __init__(self, prefix: str, configs: List[dict], task_class: Type, config_module):
        self.site_task_map: Dict[str, Set[str]] = {}
        self.task_site_map: Dict[str, str] = {}
        self.all_safe_tasks: List[str] = []
        self.prefix = prefix
        self.task_class = task_class
        self.config_module = config_module

        self._process_configs(configs)
        self._register_tasks()

    def _process_configs(self, configs: List[dict]) -> None:
        for config in configs:
            site = config["sites"]
            task_id = self.task_class.get_task_id_name(int(config["task_id"]))

            if len(site) != 1:
                continue

            if task_id in self.task_site_map:
                raise ValueError(f"Task {task_id} already registered")

            site_name = site[0]
            self.task_site_map[task_id] = site_name

            if site_name not in self.site_task_map:
                self.site_task_map[site_name] = set()
            self.site_task_map[site_name].add(task_id)

    def _register_tasks(self) -> None:
        for task_id in self.config_module.TASK_IDS:
            gym_id = self.task_class.get_task_id_name(task_id, prefix=False)
            register_safe_env_task(
                gym_id,
                self.task_class,
                task_kwargs={"task_id": task_id},
            )
            self.all_safe_tasks.append(f"browsergym/{gym_id}")


# Base Task class with common functionality
class BaseSafeTask:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeout = 160_000

    @staticmethod
    def get_task_id_name(task_id: int, prefix: bool = True) -> str:
        raise NotImplementedError


class SafeWebArenaTask(wa_task.GenericWebArenaTask, BaseSafeTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Override task grading
        for task in self.task_configs:
            if (task_eval := task.get("eval")) and (eval_types := task_eval.get("eval_types")):
                if "string_match" not in eval_types:
                    continue

                assert len(eval_types) == 1, f"Task {task['task_id']} has multiple eval types: {eval_types}"
                if "exact_match" in task_eval["reference_answers"]:
                    task_eval["reference_answers"]["fuzzy_match"] = [task_eval["reference_answers"].pop("exact_match")]
                    print(f"Updated eval for task {task['task_id']} to use fuzzy match!")

    @staticmethod
    def get_task_id_name(task_id: int, prefix: bool = True) -> str:
        return ("browsergym/" if prefix else "") + f"webarena.safe.{task_id}"

    def setup(self, page: playwright.sync_api.Page) -> tuple[str, dict]:
        page.set_default_navigation_timeout(160_000)
        page.set_default_timeout(160_000)
        setup = super().setup(page)
        # Override evaluator with our own
        self.evaluator = evaluator_router(self.config_file)
        return setup


class SafeVisualWebArenaTask(vwa_task.GenericVisualWebArenaTask, BaseSafeTask):
    @staticmethod
    def get_task_id_name(task_id: int, prefix: bool = True) -> str:
        return ("browsergym/" if prefix else "") + f"visualwebarena.safe.{task_id}"


# Initialize configurations
wa_config = BaseConfig("webarena", WA_TASK_CONFIGS, SafeWebArenaTask, wa_config)
vwa_config = BaseConfig("visualwebarena", VWA_TASK_CONFIGS, SafeVisualWebArenaTask, vwa_config)


# Create task lists by site
def get_site_tasks(config: BaseConfig, site_name: str) -> List[str]:
    return sorted(list(config.site_task_map.get(site_name, set())))


def get_random_tasks(tasks: List[str], seed: int, num_tasks: int = 1) -> List[str]:
    """Get a list of random tasks from the available safe tasks."""
    rng = random.Random(seed)
    return rng.sample(tasks, num_tasks)


def get_equal_random_samples(config: BaseConfig, seed: int, num_tasks: int = 1) -> List[str]:
    """Get an equal number of random tasks from each site.

    Args:
        config: BaseConfig instance (wa_config or vwa_config)
        seed: Random seed for reproducibility
        num_tasks: Total number of tasks to sample (distributed as evenly as possible among sites)

    Returns:
        List of task IDs sampled from each site. If num_tasks is not evenly divisible by the number
        of sites, the remainder tasks are distributed across sites from the beginning.
    """
    rng = random.Random(seed)
    sites = list(config.site_task_map.keys())
    num_sites = len(sites)

    # Calculate base tasks per site and remainder
    base_tasks_per_site = num_tasks // num_sites
    remainder = num_tasks % num_sites

    # Sample tasks from each site
    result = []
    for i, site in enumerate(sites):
        # Add one extra task to this site if there are remaining tasks to distribute
        site_tasks = base_tasks_per_site + (1 if i < remainder else 0)
        if site_tasks > 0:
            site_task_list = get_site_tasks(config, site)
            result.extend(rng.sample(site_task_list, min(site_tasks, len(site_task_list))))

    return result


# Combine and shuffle user tasks
def get_user_tasks(seed: int, num_tasks: int) -> List[str]:
    """Get a specified number of randomly sampled tasks from shopping, reddit, and gitlab."""
    rng = random.Random(seed)
    combined_tasks = WA_SHOPPING_TASKS + WA_REDDIT_TASKS + WA_GITLAB_TASKS
    return rng.sample(combined_tasks, min(num_tasks, len(combined_tasks)))


WA_SHOPPING_ADMIN_TASKS = get_site_tasks(wa_config, "shopping_admin")
WA_MAP_TASKS = get_site_tasks(wa_config, "map")
WA_SHOPPING_TASKS = get_site_tasks(wa_config, "shopping")
WA_REDDIT_TASKS = get_site_tasks(wa_config, "reddit")
WA_GITLAB_TASKS = get_site_tasks(wa_config, "gitlab")

VWA_CLASSIFIEDS_TASKS = get_site_tasks(vwa_config, "classifieds")
VWA_REDDIT_TASKS = get_site_tasks(vwa_config, "reddit")
VWA_SHOPPING_TASKS = get_site_tasks(vwa_config, "shopping")


VWA_3 = get_equal_random_samples(vwa_config, seed=42, num_tasks=3)
VWA_10 = get_equal_random_samples(vwa_config, seed=42, num_tasks=9)
VWA_30 = get_equal_random_samples(vwa_config, seed=42, num_tasks=27)
VWA_50 = get_equal_random_samples(vwa_config, seed=42, num_tasks=50)
VWA_100 = get_equal_random_samples(vwa_config, seed=42, num_tasks=100)

WA_3 = get_equal_random_samples(wa_config, seed=42, num_tasks=3)
WA_5 = get_equal_random_samples(wa_config, seed=42, num_tasks=5)
WA_10 = get_equal_random_samples(wa_config, seed=42, num_tasks=10)
WA_30 = get_equal_random_samples(wa_config, seed=42, num_tasks=30)
WA_50 = get_equal_random_samples(wa_config, seed=42, num_tasks=50)
WA_100 = get_equal_random_samples(wa_config, seed=42, num_tasks=100)

WA_EASY_100 = get_random_tasks(WA_EASY_ALL, seed=42, num_tasks=100)

_WA_EASY_ALLOWED_SITES = {"gitlab", "reddit", "shopping"}
WA_EASY_USR = [task_id for task_id in WA_EASY_ALL if wa_config.task_site_map.get(task_id) in _WA_EASY_ALLOWED_SITES]

WA_ALL = sorted(wa_config.all_safe_tasks)
WA_ALL_FIRST_HALF = WA_ALL[: len(WA_ALL) // 2]
WA_ALL_SECOND_HALF = WA_ALL[len(WA_ALL) // 2 :]

WA_USER_3 = get_user_tasks(seed=42, num_tasks=3)
WA_USER_10 = get_user_tasks(seed=42, num_tasks=10)
WA_USER_30 = get_user_tasks(seed=42, num_tasks=30)
WA_USER_50 = get_user_tasks(seed=42, num_tasks=50)
WA_USER_100 = get_user_tasks(seed=42, num_tasks=100)


create_task_dataset_if_not_exists("WA_EASY_20", WA_EASY_20)
create_task_dataset_if_not_exists("WA_EASY_ALL", WA_EASY_ALL)
create_task_dataset_if_not_exists("WA_EASY_100", WA_EASY_100)
create_task_dataset_if_not_exists("WA_EASY_USR", WA_EASY_USR)

create_task_dataset_if_not_exists("VWA_3", VWA_3)
create_task_dataset_if_not_exists("VWA_10", VWA_10)
create_task_dataset_if_not_exists("VWA_30", VWA_30)
create_task_dataset_if_not_exists("VWA_50", VWA_50)
create_task_dataset_if_not_exists("VWA_100", VWA_100)

create_task_dataset_if_not_exists("WA_3", WA_3)
create_task_dataset_if_not_exists("WA_5", WA_5)
create_task_dataset_if_not_exists("WA_10", WA_10)
create_task_dataset_if_not_exists("WA_30", WA_30)
create_task_dataset_if_not_exists("WA_50", WA_50)
create_task_dataset_if_not_exists("WA_100", WA_100)
create_task_dataset_if_not_exists("WA_ALL", WA_ALL)
create_task_dataset_if_not_exists("WA_ALL_FIRST_HALF", WA_ALL_FIRST_HALF)
create_task_dataset_if_not_exists("WA_ALL_SECOND_HALF", WA_ALL_SECOND_HALF)

create_task_dataset_if_not_exists("WA_USER_3", WA_USER_3)
create_task_dataset_if_not_exists("WA_USER_10", WA_USER_10)
create_task_dataset_if_not_exists("WA_USER_30", WA_USER_30)
create_task_dataset_if_not_exists("WA_USER_50", WA_USER_50)
create_task_dataset_if_not_exists("WA_USER_100", WA_USER_100)
