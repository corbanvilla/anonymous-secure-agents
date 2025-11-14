#!/bin/bash
cd src

uv run alembic revision --autogenerate -m "Updated column"

read -p "Do you want to continue with the database migration? (y/n): " answer

if [[ $answer == "y" ]]; then
    uv run alembic upgrade head
else
    echo "Database migration cancelled."
fi
