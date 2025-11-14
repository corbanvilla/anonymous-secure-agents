"""
This script synchronizes trajectory records from Redis to the SQL database.
It processes records in order, ensuring each is successfully committed before removal.
"""

import sys
import time

from src.db.commit_queue.defense_harness import (
    commit_defense_skip_ids,
    commit_incremental_defense_steps,
)
from src.db.commit_queue.experiments import commit_latest_records_to_db
from src.db.commit_queue.sync import get_pending_records_count
from src.log import log


def sync_records(max_retries: int = 3, retry_delay: float = 1.0) -> bool:
    """
    Attempt to sync a single record with retries on failure.

    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Delay in seconds between retries

    Returns:
        bool: True if sync was successful, False if failed
    """
    for attempt in range(max_retries):
        try:
            commit_latest_records_to_db()
            commit_incremental_defense_steps()
            commit_defense_skip_ids()
            return True  # Return True on success
        except Exception as e:
            if attempt < max_retries - 1:
                log.warning(f"Sync attempt {attempt + 1} failed: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                log.error(f"Failed to sync record after {max_retries} attempts: {e}")
                return False  # Return False on failure
    return False  # Return False if we somehow exit the loop without returning


def main():
    """
    Main sync process that processes the current batch of pending records.
    Will not process any new records that arrive after starting.
    """
    initial_count = get_pending_records_count()
    if initial_count == 0:
        log.info("No records pending synchronization")
        return

    log.info(f"Starting sync of {initial_count} pending records")

    processed_count = 0
    error_count = 0

    try:
        for _ in range(initial_count):
            # Process one record
            if sync_records():  # If sync was successful
                processed_count += 1
                if processed_count % 10 == 0:  # Log progress every 10 records
                    log.info(f"Processed {processed_count}/{initial_count} records")
            else:
                error_count += 1
                if error_count >= 3:  # Stop if we hit too many errors in a row
                    log.error("Too many consecutive errors, stopping sync")
                    break
    except KeyboardInterrupt:
        log.info("\nSync interrupted by user")
    except Exception as e:
        log.error(f"Critical error during sync: {e}")
        sys.exit(1)
    finally:
        remaining = get_pending_records_count()
        log.info(
            f"Sync complete. Processed {processed_count} records with {error_count} errors. "
            f"{remaining} records remaining."
        )


if __name__ == "__main__":
    main()
