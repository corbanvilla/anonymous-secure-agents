#!/bin/bash

TRAJECTORY_VISUALIZER=src/gradio/qual/trajectory_visualizer.py
EXPERIMENT_VISUALIZER=src/gradio/quant/experiment_visualizer.py

# Get the directory for each visualizer
TRAJECTORY_DIR=$(dirname "$TRAJECTORY_VISUALIZER")
EXPERIMENT_DIR=$(dirname "$EXPERIMENT_VISUALIZER")

# Start both visualizers in parallel, each with their own entr watcher
find "$TRAJECTORY_DIR" -type f | entr -r uv run "$TRAJECTORY_VISUALIZER" --no-share --port 9080 &
PID1=$!
find "$EXPERIMENT_DIR" -type f | entr -r uv run "$EXPERIMENT_VISUALIZER" --no-share --port 9081 &
PID2=$!

# Trap to kill both on exit
trap "kill $PID1 $PID2" SIGINT SIGTERM EXIT

# Wait for both to finish
wait $PID1 $PID2
