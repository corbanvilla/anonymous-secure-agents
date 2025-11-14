import os
from pathlib import Path

POSTGRES_DB = os.environ.get("POSTGRES_DB", "database")
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASS = os.environ.get("POSTGRES_PASS", "wnx8nfg5ekeHFH35ymx")

POSTGRES_DATABASE_URL = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASS}@{POSTGRES_HOST}/{POSTGRES_DB}"

LARGE_FILES_DIR = Path(os.environ.get("LARGE_FILES_DIR", "/data/user_one/agent-file-db"))
SCREENSHOT_DIR = Path(os.environ.get("SCREENSHOT_DIR", "/data/user_one/screenshots"))
DROP_FIELDS = ["chat_messages", "goal_object"]
STORE_FILE_FIELD_PREFIXES = ["dom_object", "axtree_object", "extra_element_properties"]
STORE_SCREENSHOT_FIELDS = ["screenshot", "screenshot_attack", "screenshot_censored"]
LLM_LOG_DROP_FIELDS = ["response", "messages"]
