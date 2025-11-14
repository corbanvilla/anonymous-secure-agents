import time

import pytest
from playwright.sync_api import sync_playwright

PAGE_LOAD_TIMEOUT_S = 20


def test_gradio_interface(gradio_server):
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
        page.screenshot(path="gradio.png", full_page=True)
        with open("gradio.html", "w") as f:
            f.write(page.content())
        browser.close()


if __name__ == "__main__":
    pytest.main([__file__])
