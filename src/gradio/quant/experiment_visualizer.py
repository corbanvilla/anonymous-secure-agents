import argparse
import functools
import os
from typing import Any, cast
import gradio as gr

import seaborn as sns


from src.db.helpers.datasets import get_all_datasets
from src.db.helpers.experiments import (
    filter_experiment_names,
    get_all_attacks,
    get_all_defenses,
    load_recent_experiments,
)
from src.experiments.config import ExperimentConfig
from src.gradio.quant.plots.stop_reason import (
    plot_stop_reason_aggregated,
    plot_stop_reason_by_experiment,
)
from src.gradio.quant.plots.task_success import (
    plot_steps_box_plot,
    plot_steps_histogram_by_site,
    plot_steps_histogram_failed,
    plot_steps_histogram_success,
    plot_task_success_rate,
    plot_task_success_rate_by_site,
)
from src.gradio.quant.tables.stats import calculate_experiment_metrics

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")


def clean_plot_state(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sns.reset_defaults()
        sns.set_theme(context="paper", style="white")
        fig = func(*args, **kwargs)
        for ax in fig.axes:
            try:
                # only apply if no legend is already present
                if not ax.get_legend():
                    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
            except Exception as e:
                print("Legend failed:", e)
        fig.tight_layout(pad=2.0)
        return fig

    return wrapper


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
    choices = filter_experiment_names(
        show_tests=show_tests,
        selected_user=user,
        selected_model=model,
        selected_attack=attack,
        selected_defense=defense,
        prefix=prefix,
        selected_dataset=dataset,
        show_hidden=show_hidden,
        favorites_only=favorites_only,
    )
    return gr.update(choices=choices)


def make_all_plots(exps):
    # First calculate metrics table
    metrics_df = calculate_experiment_metrics(exps)

    return (
        metrics_df,
        clean_plot_state(plot_stop_reason_by_experiment)(exps),
        clean_plot_state(plot_stop_reason_aggregated)(exps),
        clean_plot_state(plot_task_success_rate)(exps),
        clean_plot_state(plot_task_success_rate_by_site)(exps),
        clean_plot_state(plot_steps_histogram_failed)(exps),
        clean_plot_state(plot_steps_histogram_success)(exps),
        clean_plot_state(plot_steps_box_plot)(exps),
        clean_plot_state(plot_steps_histogram_by_site)(exps),
    )


def build_interface() -> gr.Blocks:
    all_experiments = load_recent_experiments()
    user_choices = ["All"] + sorted(str(exp.username) for exp in all_experiments)

    engine_set = set()
    for exp in all_experiments:
        try:
            cfg = ExperimentConfig.model_validate(exp.config)
            engine_set.add(cfg.engine_options.sampling_params.model)
        except Exception:
            continue
    engine_choices = ["All"] + sorted(str(model) for model in engine_set)

    attack_choices = ["Any", "None"] + get_all_attacks()
    defense_choices = ["Any", "None"] + get_all_defenses()
    dataset_choices = ["All", "Unknown"] + get_all_datasets()

    default_choices = filter_experiment_names(
        show_tests=True,
        selected_user="All",
        selected_model="All",
        selected_attack="Any",
        selected_defense="Any",
        prefix="",
        selected_dataset="All",
        show_hidden=False,
        favorites_only=False,
    )

    with gr.Blocks() as demo:
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

                gr.Markdown("### Select Experiment(s)")
                experiment_prefix = gr.Textbox(
                    label=None, placeholder="Search experiments", lines=1, interactive=True, show_label=False
                )
                experiments_dropdown = gr.Dropdown(
                    label="Experiment Name",
                    choices=default_choices,
                    value=default_choices[:3] if len(default_choices) >= 3 else default_choices,
                    multiselect=True,
                    interactive=True,
                )
            with gr.Column(scale=3):
                gr.Markdown("## Experiment Metrics")
                metrics_table = gr.Dataframe(
                    headers=["Metric"],
                    type="pandas",
                    interactive=False,
                )
                gr.Markdown("## Stop Reason Distribution by Experiment")
                stop_reason_by_exp_plot = gr.Plot()
                gr.Markdown("## Aggregated Stop Reason Distribution")
                stop_reason_agg_plot = gr.Plot()
                gr.Markdown("## Task Success Rate")
                task_success_plot = gr.Plot()
                gr.Markdown("## Task Success Rate by Site")
                task_success_site_plot = gr.Plot()
                gr.Markdown("## Steps Distribution for Failed Trajectories")
                steps_hist_failed_plot = gr.Plot()
                gr.Markdown("## Steps Distribution for Successful Trajectories")
                steps_hist_success_plot = gr.Plot()
                gr.Markdown("## Steps Distribution by Outcome (Box Plot)")
                steps_box_plot = gr.Plot()
                gr.Markdown("## Steps Histogram by Site")
                steps_site_hist_plot = gr.Plot()

        input_controls = [
            experiment_prefix,
            show_tests,
            user_filter,
            model_filter,
            attack_filter_exp,
            defense_filter_exp,
            dataset_filter,
            show_hidden_cb,
            favorites_only_cb,
        ]

        for ctrl in input_controls:
            cast(Any, ctrl).change(
                fn=update_experiment_dropdown,
                inputs=input_controls,
                outputs=[experiments_dropdown],
                show_progress="hidden",
            )
        refresh_button.click(
            fn=update_experiment_dropdown,
            inputs=input_controls,
            outputs=[experiments_dropdown],
            show_progress="hidden",
        )

        # Wire up all plots to update sequentially when experiments_dropdown changes
        experiments_dropdown.change(
            fn=make_all_plots,
            inputs=[experiments_dropdown],
            outputs=[
                metrics_table,
                stop_reason_by_exp_plot,
                stop_reason_agg_plot,
                task_success_plot,
                task_success_site_plot,
                steps_hist_failed_plot,
                steps_hist_success_plot,
                steps_box_plot,
                steps_site_hist_plot,
            ],
            queue=True,
        )

        # Trigger initial plots on load
        demo.load(
            fn=lambda: (
                calculate_experiment_metrics(default_choices[:3] if len(default_choices) >= 3 else default_choices),
                clean_plot_state(plot_stop_reason_by_experiment)(
                    default_choices[:3] if len(default_choices) >= 3 else default_choices
                ),
                clean_plot_state(plot_stop_reason_aggregated)(
                    default_choices[:3] if len(default_choices) >= 3 else default_choices
                ),
                clean_plot_state(plot_task_success_rate)(
                    default_choices[:3] if len(default_choices) >= 3 else default_choices
                ),
                clean_plot_state(plot_task_success_rate_by_site)(
                    default_choices[:3] if len(default_choices) >= 3 else default_choices
                ),
                clean_plot_state(plot_steps_histogram_failed)(
                    default_choices[:3] if len(default_choices) >= 3 else default_choices
                ),
                clean_plot_state(plot_steps_histogram_success)(
                    default_choices[:3] if len(default_choices) >= 3 else default_choices
                ),
                clean_plot_state(plot_steps_box_plot)(
                    default_choices[:3] if len(default_choices) >= 3 else default_choices
                ),
                clean_plot_state(plot_steps_histogram_by_site)(
                    default_choices[:3] if len(default_choices) >= 3 else default_choices
                ),
            ),
            outputs=[
                metrics_table,
                stop_reason_by_exp_plot,
                stop_reason_agg_plot,
                task_success_plot,
                task_success_site_plot,
                steps_hist_failed_plot,
                steps_hist_success_plot,
                steps_box_plot,
                steps_site_hist_plot,
            ],
        )

    return demo


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

    demo = build_interface()
    demo.launch(share=args.share, server_port=args.port, server_name="127.0.0.1")
