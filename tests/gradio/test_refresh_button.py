import os
import time

from playwright.sync_api import sync_playwright

from src.db.helpers.experiments import create_experiment
from src.experiments.config import get_experiment_config

PAGE_LOAD_TIMEOUT_S = 20


def test_refresh_button_updates_experiments(gradio_server, monkeypatch):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(gradio_server)
        start = time.time()
        while True:
            if "Experiment Name" in page.content():
                break
            if time.time() - start > PAGE_LOAD_TIMEOUT_S:
                raise AssertionError("Gradio page failed to load in time")
            time.sleep(0.5)

        get_names_js = (
            "window.gradio_config.components.find(c => c.props && c.props.label === 'Experiment Name').props.choices"
        )
        names_before = page.evaluate(get_names_js)

        def unpack(choices):
            result = []
            for item in choices:
                if isinstance(item, list) or isinstance(item, tuple):
                    if item:
                        result.append(item[0])
                else:
                    result.append(item)
            return result

        names_before = unpack(names_before)

        experiment_name = f"test_refresh_{int(time.time())}"
        config = get_experiment_config(
            name=experiment_name,
            description="test",
            tasks=[],
            engine_model="gpt-4.1",
            webagent_src=["html"],
        ).model_dump()
        monkeypatch.setattr(os, "getlogin", lambda: "testuser")
        create_experiment(experiment_name, [], config)

        page.get_by_role("button", name="Refresh Latest Experiment").click()
        start = time.time()
        names_after = unpack(page.evaluate(get_names_js))
        while experiment_name not in names_after:
            if time.time() - start > PAGE_LOAD_TIMEOUT_S:
                break
            time.sleep(0.5)
            names_after = unpack(page.evaluate(get_names_js))
        browser.close()

    assert experiment_name not in names_before
    assert experiment_name in names_after
