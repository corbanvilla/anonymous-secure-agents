from browsergym.core.action.highlevel import HighLevelActionSet

import src.environments.actions.functions as functions

WA_ACTION_SET = HighLevelActionSet(
    subsets=["custom"],
    custom_actions=[
        functions.send_msg_to_user,
        # patch_functions.tab_close,
        # patch_functions.tab_focus,
        # patch_functions.new_tab,
        functions.go_back,
        functions.go_forward,
        # patch_functions.goto,
        functions.scroll,
        functions.fill,
        functions.select_option,
        functions.click,
        functions.dblclick,
        functions.hover,
        functions.press,
        functions.focus,
        functions.clear,
        functions.drag_and_drop,
        # functions.upload_file,
        functions.report_infeasible,
    ],
    strict=False,
    multiaction=False,
    demo_mode="off",
)
