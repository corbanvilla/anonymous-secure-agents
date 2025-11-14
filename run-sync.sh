#!/bin/bash

while true; do
    uv run src/db/sync.py
    sleep 5
done
