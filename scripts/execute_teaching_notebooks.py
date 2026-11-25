#!/usr/bin/env python3
"""Execute the four canonical teaching notebooks, each in a fresh kernel."""

from __future__ import annotations

import argparse
import io
import subprocess
import tarfile
import tempfile
from pathlib import Path
from time import perf_counter

import nbformat
from nbclient import NotebookClient


REPO_ROOT = Path(__file__).resolve().parents[1]
TEACHING = Path("notebooks/teaching")
DATA = Path("data/raw/tpq_eod_2014_2023.csv")
DEFAULT_NOTEBOOKS = (
    "01_understand_the_data.ipynb",
    "02_construct_candidate_features.ipynb",
    "03_test_predictive_value.ipynb",
    "04_predictions_to_portfolio_accounting.ipynb",
)


def prepare_run(run: Path, names: list[str], revision: str) -> None:
    """Materialize only committed Python helpers and requested notebook sources."""
    archive = subprocess.check_output(
        ["git", "archive", revision, "src/quant_journey", *[str(TEACHING / n) for n in names]],
        cwd=REPO_ROOT,
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as snapshot:
        snapshot.extractall(run, filter="data")

    (run / DATA).parent.mkdir(parents=True)
    (run / DATA).symlink_to(REPO_ROOT / DATA)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="a-quant-journey", help="registered kernel name")
    parser.add_argument("notebooks", nargs="*", help="optional canonical filenames")
    args = parser.parse_args()
    selected = set(args.notebooks or DEFAULT_NOTEBOOKS)
    unknown = selected.difference(DEFAULT_NOTEBOOKS)
    if unknown:
        parser.error(f"unknown teaching notebook filename(s): {', '.join(sorted(unknown))}")
    if DEFAULT_NOTEBOOKS[3] in selected:
        selected.add(DEFAULT_NOTEBOOKS[2])
    names = [name for name in DEFAULT_NOTEBOOKS if name in selected]
    if not (REPO_ROOT / DATA).is_file():
        parser.error(f"missing authorized local snapshot: {DATA}")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
    ).strip()
    runs = REPO_ROOT / "outputs/teaching_runs"
    runs.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=f"{revision[:8]}-", dir=runs))
    print(f"Teaching run at {revision}: {run}", flush=True)
    prepare_run(run, names, revision)
    executed_dir = run / "executed"
    executed_dir.mkdir()
    for name in names:
        path = run / TEACHING / name
        notebook = nbformat.read(path, as_version=4)
        started = perf_counter()
        client = NotebookClient(
            notebook,
            timeout=180,
            kernel_name=args.kernel,
            resources={"metadata": {"path": str(run)}},
        )
        executed = client.execute()
        elapsed = perf_counter() - started
        nbformat.write(executed, executed_dir / name)
        code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
        output_records = sum(len(cell.get("outputs", [])) for cell in code_cells)
        print(
            f"{name}: PASS; {len(code_cells)} code cells; "
            f"{output_records} output records; runtime={elapsed:.2f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
