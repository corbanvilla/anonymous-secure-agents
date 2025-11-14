import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import sqlalchemy

from src.db.config import POSTGRES_DATABASE_URL

SELECT_COUNT = sqlalchemy.text("SELECT COUNT(*) FROM screenshots_v2")
SELECT_QUERY = sqlalchemy.text(
    "SELECT id, observation_id, data FROM screenshots_v2 ORDER BY id LIMIT :limit OFFSET :offset"
)
SELECT_OBS = sqlalchemy.text("SELECT data FROM observations_v2 WHERE id=:id")
UPDATE_OBS = sqlalchemy.text("UPDATE observations_v2 SET data=:data WHERE id=:id")

BATCH_SIZE = 100  # Process 100 screenshots at a time


@dataclass
class Stats:
    files_existed: int = 0
    files_created: int = 0
    screenshots_skipped: int = 0
    start_time: float = 0.0

    @property
    def total_processed(self) -> int:
        return self.files_existed + self.files_created + self.screenshots_skipped

    def print_summary(self) -> None:
        elapsed_time = time.time() - self.start_time
        rate = self.total_processed / elapsed_time if elapsed_time > 0 else 0

        print("\n=== Export Statistics ===")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Processing rate: {rate:.2f} screenshots/second")
        print("\nFile Statistics:")
        print(f"- Already existed: {self.files_existed}")
        print(f"- Newly created: {self.files_created}")
        print(f"- Skipped (no data): {self.screenshots_skipped}")
        print(f"Total screenshots processed: {self.total_processed}")


def process_screenshot_batch(
    conn, out_dir: Path, rows, counters: dict[int, int], start_idx: int, total_screenshots: int, stats: Stats
) -> None:
    for i, row in enumerate(rows, start_idx):
        print(f"\nProcessing screenshot {i}/{total_screenshots} (ID: {row.id})")
        data: bytes | None = row.data

        if data is not None:
            digest = hashlib.sha256(data).hexdigest()
            filename = f"{digest}.png"
            filepath = out_dir / filename
            if not filepath.exists():
                print(f"Saving new file: {filename}")
                filepath.write_bytes(data)
                stats.files_created += 1
            else:
                print(f"File already exists: {filename}")
                stats.files_existed += 1

            obs_id: int = row.observation_id
            obs_res = conn.execute(SELECT_OBS, {"id": obs_id})
            obs_data = obs_res.scalar_one()
            if isinstance(obs_data, str):
                obs_json = json.loads(obs_data)
            else:
                obs_json = obs_data

            count = counters.get(obs_id, 0)
            key = "screenshot" if count == 0 else "screenshot_censored"
            obs_json[key] = filename
            counters[obs_id] = count + 1

            conn.execute(UPDATE_OBS, {"data": json.dumps(obs_json), "id": obs_id})
        else:
            print("Skipping screenshot with no data")
            stats.screenshots_skipped += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", help="Directory to save screenshots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving screenshots to directory: {out_dir}")

    engine = sqlalchemy.create_engine(POSTGRES_DATABASE_URL, future=True)
    print("Connected to database, starting screenshot export...")

    stats = Stats(start_time=time.time())

    with engine.begin() as conn:
        # Get total count first
        total_screenshots = conn.execute(SELECT_COUNT).scalar_one()
        print(f"Found {total_screenshots} screenshots to process")

        counters: dict[int, int] = {}
        offset = 0

        while offset < total_screenshots:
            # Fetch batch of screenshots
            result = conn.execute(SELECT_QUERY, {"limit": BATCH_SIZE, "offset": offset})
            rows = result.fetchall()

            if not rows:
                break

            process_screenshot_batch(conn, out_dir, rows, counters, offset + 1, total_screenshots, stats)

            offset += BATCH_SIZE
            print(f"\nCompleted batch. Progress: {min(offset, total_screenshots)}/{total_screenshots}")

    stats.print_summary()
    print("\nExport completed successfully!")


if __name__ == "__main__":
    main()
