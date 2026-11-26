"""Fixed teaching experiment and dated prediction-file contract."""
import csv
import hashlib
import io
import json
from pathlib import Path

import pandas as pd

from ..models.preprocessing import FEATURE_COLUMNS

CUTOFF = "2021-12-28"
MODELS = ["GaussianNB", "Logistic regression", "Decision tree", "Random forest",
          "LinearSVC", "Hard voting", "MLP", "Training majority baseline"]
BENCHMARK = "SPY buy-and-hold"
SCENARIOS = {"zero_cost": 0, "illustrative_5bps": 5}
VALIDATION_START = "2020-01-02"
RESERVED_START = "2021-12-29"
INITIAL_CASH = 1000


def read_predictions(directory, train, validation):
    directory = Path(directory)
    manifest = json.loads((directory / "provenance.json").read_bytes())
    raw = (directory / "validation_predictions.csv").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest.get("validation_predictions_csv_sha256"):
        raise ValueError("Prediction digest mismatch")
    settings = manifest["models"]
    if (manifest["cutoff"] != CUTOFF or manifest["train_rows"] != len(train)
            or manifest["validation_rows"] != len(validation)
            or manifest["reserved_test_access"] is not False
            or manifest["validation_is_fresh_holdout"] is not False
            or manifest["features"] != list(FEATURE_COLUMNS)
            or manifest["target"] != "close[t+2]/close[t+1]-1 > 0"
            or [s["family"] for s in settings] != MODELS[:-1]
            or "max_iter=500" not in settings[-1]["fixed settings"].split("; ")):
        raise ValueError("Partition or model provenance mismatch")
    records = csv.reader(io.StringIO(raw.decode(), newline=""), strict=True)
    header = next(records, [])
    if header != ["feature_date", *MODELS, "actual"]:
        raise ValueError("Unexpected prediction columns")
    frame = pd.DataFrame(records, columns=header).set_index("feature_date")
    if (frame.index.tolist() != validation.index.strftime("%Y-%m-%d").tolist()
            or not frame.isin(["0", "1"]).all().all()):
        raise ValueError("Invalid prediction dates or binary fields")
    frame.index = validation.index
    frame = frame.astype(int)
    if not frame.pop("actual").equals(validation.execution_target_up):
        raise ValueError("Incorrect actual labels")
    return frame
