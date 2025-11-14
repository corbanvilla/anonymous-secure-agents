from src.db.client import Session
from src.db.helpers.timing import record_timing
from src.db.tables import FunctionTiming


class DummyDefense:
    def __init__(self):
        self.sampling_params = {"model": "dummy-model"}

    @record_timing()
    def run(self):
        return "ok"


def test_record_timing_stores_model_and_defense():
    d = DummyDefense()
    d.run()

    with Session() as session:
        timing = session.query(FunctionTiming).order_by(FunctionTiming.id.desc()).first()
    assert timing.model_name == "dummy-model"
    assert timing.defense_name == "DummyDefense"
