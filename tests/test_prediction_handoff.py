import hashlib
import json

import pandas as pd
import pytest

from src.quant_journey.execution import prediction_handoff as handoff


@pytest.mark.parametrize("fault", ["none","digest","malformed","columns","order","shift","duplicate","binary","actual","manifest"])
def test_handoff(tmp_path, fault):
    idx = pd.date_range("2020-01-02", periods=2)
    v = pd.DataFrame({"execution_target_up":[1,0]},index=idx)
    columns = ["feature_date", *handoff.MODELS, "actual"]
    rows = [[d,*(["1"]*8),a] for d,a in zip(idx.strftime("%Y-%m-%d"), ["1","0"])]
    if fault == "columns": columns[1] = columns[2]
    if fault == "order": columns[1:3] = columns[1:3][::-1]
    if fault == "shift": rows[0][0] = "2020-01-01"
    if fault == "duplicate": rows[0][0] = rows[1][0]
    if fault == "binary": rows[0][1] = "1.0"
    if fault == "actual": rows[0][-1] = "0"
    raw = "\n".join(",".join(row) for row in [columns,*rows]).encode()
    manifest = dict(models=[dict(family=m, **{"fixed settings":"max_iter=500"}) for m in handoff.MODELS[:-1]],
                    features=["return_1d","momentum_5d","volatility_5d"],
                    target="close[t+2]/close[t+1]-1 > 0", validation_is_fresh_holdout=False,
                    reserved_test_access=False, cutoff=handoff.CUTOFF, train_rows=2, validation_rows=2,
                    validation_predictions_csv_sha256=hashlib.sha256(raw).hexdigest())
    if fault == "digest": raw += b"\n"
    if fault == "malformed": manifest["validation_predictions_csv_sha256"] = "bad"
    if fault == "manifest": manifest["validation_rows"] = 3
    (tmp_path/"validation_predictions.csv").write_bytes(raw)
    (tmp_path/"provenance.json").write_text(json.dumps(manifest))
    if fault == "none":
        assert handoff.read_predictions(tmp_path,v,v).columns.tolist() == handoff.MODELS
    else:
        with pytest.raises(ValueError):
            handoff.read_predictions(tmp_path,v,v)


def test_optional_runner_routes_dependencies(tmp_path, monkeypatch):
    import sys
    from scripts import execute_teaching_notebooks as runner

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: "revision")
    names = runner.NOTEBOOKS[2:]
    def snapshot(run, selected, revision, candidate):
        assert selected == list(names) and candidate
        (run / runner.TEACHING).mkdir(parents=True)
        for name in selected:
            runner.nbformat.write(runner.nbformat.v4.new_notebook(), run / runner.TEACHING / name)
    monkeypatch.setattr(runner, "prepare_run", snapshot)
    calls = []
    class Client:
        def __init__(self, notebook, *, kernel_name, **kwargs):
            self.kernel = kernel_name

        def execute(self, env):
            calls.append((self.kernel, env["AQJ_ARTIFACT_DIR"]))
    monkeypatch.setattr(runner, "NotebookClient", Client)
    monkeypatch.setattr(sys, "argv", ["runner", "--candidate", names[-1]])
    runner.main()
    assert [kernel for kernel, _ in calls] == ["a-quant-journey", "a-quant-journey", "nb05-native"]
    assert len({directory for _, directory in calls}) == 1
