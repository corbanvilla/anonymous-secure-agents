# copied in from browsergym/core/action/utils.py with increased timeouts

import playwright.sync_api
from playwright.sync_api import FrameLocator

BID_TIMEOUT = 10_000


def get_elem_by_bid(
    page: playwright.sync_api.Page,
    bid: str,
    scroll_into_view: bool = False,
) -> playwright.sync_api.Locator:
    """Return locator for a BrowserGym element, optionally scrolling into view."""
    if not isinstance(bid, str):
        raise ValueError(f"expected a string, got {repr(bid)}")

    current_frame: playwright.sync_api.Page | FrameLocator = page
    i = 0
    while bid[i:] and not bid[i:].isnumeric():
        i += 1
        while bid[i:] and bid[i].isalpha() and bid[i].isupper():
            i += 1
        frame_bid = bid[:i]
        frame_elem = current_frame.get_by_test_id(frame_bid)
        if not frame_elem.count():
            raise ValueError(f'Could not find element with bid "{bid}"')
        if scroll_into_view:
            frame_elem.scroll_into_view_if_needed(timeout=BID_TIMEOUT)
    current_frame = frame_elem.frame_locator(":scope")

    elem = current_frame.get_by_test_id(bid)
    if not elem.count():
        raise ValueError(f'Could not find element with bid "{bid}"')
    if scroll_into_view:
        elem.scroll_into_view_if_needed(timeout=BID_TIMEOUT)
    return elem


# re-export the rest of browsergym.utils functions for convenience
from browsergym.core.action.utils import (  # noqa: E402
    add_demo_mode_effects,
    call_fun,
    highlight_by_box,
    smooth_move_visual_cursor_to,
)

__all__ = [
    "get_elem_by_bid",
    "add_demo_mode_effects",
    "call_fun",
    "highlight_by_box",
    "smooth_move_visual_cursor_to",
]
