from typing import Literal, Tuple


def parse_action(action: str) -> Tuple[str, int | None]:
    """
    Parse action string and return relevant bid and function name.

    If no bid is involved in the action, returns None for the bid.
    """

    def _extract_first_arg(action: str) -> str:
        return action.split("(")[1].split(")")[0].split(",")[0].strip("'").strip('"').strip()

    function_name = action.split("(")[0].strip("`").strip()
    match function_name:
        case "send_msg_to_user":
            return function_name, None
        case "go_back":
            return function_name, None
        case "go_forward":
            return function_name, None
        case "scroll":
            return function_name, None
        case "fill":
            bid = int(_extract_first_arg(action))
            return function_name, bid
        case "select_option":
            bid = int(_extract_first_arg(action))
            return function_name, bid
        case "click":
            bid = int(_extract_first_arg(action))
            return function_name, bid
        case "dblclick":
            bid = int(_extract_first_arg(action))
            return function_name, bid
        case "hover":
            bid = int(_extract_first_arg(action))
            return function_name, bid
        case "press":
            bid = int(_extract_first_arg(action))
            return function_name, bid
        case "focus":
            bid = int(_extract_first_arg(action))
            return function_name, bid
        case "clear":
            bid = int(_extract_first_arg(action))
            return function_name, bid
        case "drag_and_drop":
            return function_name, None
        case "report_infeasible":
            return function_name, None
        case "noop":
            return function_name, None
        case _:
            raise ValueError(f"Unknown action: {action}")


# def get_required_capabilities(action: str) -> Literal["viewable", "clickable", "typable"]:
#     """
#     Returns the required capabilities for the action.

#     viewable: the action requires the ability to view the element (default)
#     clickable: the action requires the ability to click the element
#     typable: the action requires the ability to type into the element
#     """

#     match action:
#         case "fill":
#             return "typable"
#         case "select_option":
#             return "clickable"
#         case "click":
#             return "clickable"
#         case "dblclick":
#             return "clickable"
#         case "hover":
#             return "clickable"
#         case "press":
#             return "typable"  # This is debatable, could be clickable too
#         case "focus":
#             return "clickable"
#         case "clear":
#             return "clickable"
#         case _:
#             raise ValueError(f"Unknown action: {action}")


# def has_capabilities_for_action(
#     action: str, bid: int, cap_set: dict[int, Literal["viewable", "clickable", "typable"]]
# ) -> bool:
#     """
#     Returns True if the action has the required capabilities for the action.
#     """
#     cap_order = ["viewable", "clickable", "typable"]

#     required_cap = get_required_capabilities(action)
#     required_cap_index = cap_order.index(required_cap)

#     selected_cap = cap_set.get(bid, "viewable")
#     selected_cap_index = cap_order.index(selected_cap)

#     print(f"Required cap: {required_cap} ({required_cap_index})")
#     print(f"Selected cap: {selected_cap} ({selected_cap_index})")
#     print(f"Returning {(selected_cap_index >= required_cap_index)=} for action {action} with bid {bid}")

#     return selected_cap_index >= required_cap_index


def get_required_capabilities(action: str) -> Literal["viewable", "interactable"]:
    """
    Returns the required capabilities for the action.

    viewable: the action requires the ability to view the element (default)
    interactable: the action requires the ability to interact with the element
    """

    match action:
        case "fill":
            return "interactable"
        case "select_option":
            return "interactable"
        case "click":
            return "interactable"
        case "dblclick":
            return "interactable"
        case "hover":
            return "viewable"
        case "press":
            return "interactable"
        case "focus":
            return "viewable"
        case "clear":
            return "interactable"
        case _:
            raise ValueError(f"Unknown action: {action}")


def has_capabilities_for_action(action: str, bid: int, cap_set: dict[int, Literal["viewable", "interactable"]]) -> bool:
    """
    Returns True if the action has the required capabilities for the action.
    """
    cap_order = ["viewable", "interactable"]

    required_cap = get_required_capabilities(action)
    required_cap_index = cap_order.index(required_cap)

    selected_cap = cap_set.get(bid)
    if selected_cap is None:
        print(f"Required cap: {required_cap} ({required_cap_index})")
        print(f"No cap found for bid {bid}")
        return False

    selected_cap_index = cap_order.index(selected_cap)

    print(f"Required cap: {required_cap} ({required_cap_index})")
    print(f"Selected cap: {selected_cap} ({selected_cap_index})")
    print(f"Returning {(selected_cap_index >= required_cap_index)=} for action {action} with bid {bid}")

    return selected_cap_index >= required_cap_index
