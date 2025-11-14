When creating a new plot, please refer to `plots/stop_reason.py` for examples.

Specifically:
- Note that `clean_plot_state` is provided to set the sns theme and tight_layout automatically, don't set colors or much else.
- If you need to add a legend, use `ax.legend(loc="upper left", bbox_to_anchor=(1, 1))`
- Ensure you use `fig, ax = plt.subplots(figsize=(12, 6), dpi=300)` to set the figure size and dpi.
- Don't put more than one plot per figure.
- Attempt to keep everything in one sql query per graph, if possible.
- Don't overclutter plot files with too many different plots. Separate out into new files if it makes sense.
- Ensure plots are working by invoking them directly. Make sure to include an `if __name__ == "__main__":` block to save the plot to a file, and invoke it to test.
