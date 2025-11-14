import pandas as pd
from sqlalchemy import text

from src.db.client import Session, engine
from src.db.tables import Experiment


def main() -> None:
    experiment_name = "WA-EASY-100-BASELINE-V3-a11y-none-none-gpt-5-mini"

    # Resolve experiment to an ID via the ORM table definition
    with Session() as session:
        experiment_row = session.query(Experiment).filter(Experiment.name == experiment_name).one_or_none()

    if experiment_row is None:
        raise SystemExit(f"Experiment not found: {experiment_name}")

    # Query the summary view with pandas for ASR, TSR, and DSR
    query = text(
        """
        SELECT asr, tsr, dsr
        FROM experiment_summary_v2
        WHERE id = :experiment_id
        """
    )
    df = pd.read_sql_query(query, con=engine, params={"experiment_id": experiment_row.id})  # type: ignore

    print(df)

    # Render LaTeX table using the modern Styler API
    styler = df.style.format(precision=2)
    latex = styler.to_latex()
    print(latex)


if __name__ == "__main__":
    main()
