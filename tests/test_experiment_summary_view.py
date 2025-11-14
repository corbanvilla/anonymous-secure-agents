from sqlalchemy import text

from src.db.client import Session, safe_commit
from src.db.views import EXPERIMENT_SUMMARY_VIEW


def test_experiment_summary_view_attack_defense_columns():
    # Recreate the view to make sure latest SQL is applied
    with Session() as session:
        session.execute(text(EXPERIMENT_SUMMARY_VIEW))
        safe_commit(session)

    with Session() as session:
        row = session.execute(text("SELECT attack_id, defense_id FROM experiment_summary_v2 LIMIT 1")).fetchone()
        assert row is not None
        assert set(row._mapping.keys()) == {"attack_id", "defense_id"}
