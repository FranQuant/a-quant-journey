#!/usr/bin/env python3
"""Execute an identified teaching snapshot in fresh kernels and optionally export HTML."""
import argparse
import hashlib
from html import escape
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from time import perf_counter

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
TEACHING = Path("notebooks/teaching")
DATA = Path("data/raw/tpq_eod_2014_2023.csv")
NOTEBOOKS = (
    "01_understand_the_data.ipynb",
    "02_construct_candidate_features.ipynb",
    "03_test_predictive_value.ipynb",
    "04_predictions_to_portfolio_accounting.ipynb",
    "05_vectorbt_portfolio_reconciliation.ipynb",
)


def prepare_run(run, names, revision, candidate):
    paths = [TEACHING / name for name in names]
    paths.extend(map(Path, ["scripts/execute_teaching_notebooks.py", "requirements.txt"]))
    if NOTEBOOKS[-1] in names:
        paths.append(Path("requirements-vectorbt.lock"))
    if candidate:
        paths.extend(p.relative_to(ROOT) for p in (ROOT / "src/quant_journey").rglob("*.py"))
        for path in paths:
            (run / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / path, run / path)
    else:
        archive = subprocess.check_output(
            ["git", "archive", revision, "src/quant_journey", *map(str, paths)], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as snapshot:
            snapshot.extractall(run, filter="data")
    files = {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(run.rglob("*")) if p.is_file()}
    (run / "source.json").write_text(json.dumps({
        "checkout_head": revision, "working_tree_snapshot": candidate, "files": files,
    }, indent=2))
    (run / DATA).parent.mkdir(parents=True)
    (run / DATA).symlink_to(ROOT / DATA)


def export_html(notebook, destination):
    """Pre-typeset Markdown with the local Pandoc MathML writer, then export outputs."""
    publication = nbformat.from_dict(notebook)
    for cell in publication.cells:
        if cell.cell_type == "markdown":
            cell.source = subprocess.run(
                ["pandoc", "--from=markdown+tex_math_dollars", "--to=html5", "--mathml"],
                input=cell.source, text=True, capture_output=True, check=True).stdout
            cell.source = "<div>\n" + cell.source + "\n</div>"
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if "image/png" in data:
                alt = escape(cell.metadata["alt"])
                image = data.pop("image/png")
                data["text/html"] = (
                    f'<figure><img src="data:image/png;base64,{image}" alt="{alt}">'
                    f'<figcaption>{alt}</figcaption></figure>'
                )
    with destination.open("wb") as file:
        subprocess.run(["jupyter-nbconvert", "--stdin", "--stdout", "--to=html",
                        "--HTMLExporter.mathjax_url=", "--HTMLExporter.require_js_url="],
                       input=nbformat.writes(publication).encode(), stdout=file, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="a-quant-journey")
    parser.add_argument("--analytics-kernel", default="nb05-native")
    parser.add_argument("--all", action="store_true", help="include the optional fifth notebook")
    parser.add_argument("--candidate", action="store_true", help="snapshot current source bytes")
    parser.add_argument("--html", action="store_true", help="export using local Pandoc and nbconvert")
    parser.add_argument("notebooks", nargs="*")
    args = parser.parse_args()
    selected = set(args.notebooks or (NOTEBOOKS if args.all else NOTEBOOKS[:4]))
    if selected.difference(NOTEBOOKS):
        parser.error("Unknown teaching notebook")
    if NOTEBOOKS[4] in selected:
        selected.add(NOTEBOOKS[3])
    if NOTEBOOKS[3] in selected:
        selected.add(NOTEBOOKS[2])
    names = [name for name in NOTEBOOKS if name in selected]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    runs = ROOT / "outputs/teaching_runs"
    runs.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=f"{revision[:8]}-", dir=runs))
    print(f"Teaching snapshot: {run}", flush=True)
    prepare_run(run, names, revision, args.candidate)
    executed = run / "executed"
    executed.mkdir()
    if args.html:
        (run / "html").mkdir()
    for name in names:
        notebook = nbformat.read(run / TEACHING / name, as_version=4)
        kernel = args.analytics_kernel if name == NOTEBOOKS[4] else args.kernel
        started = perf_counter()
        client = NotebookClient(notebook, timeout=240, kernel_name=kernel, store_widget_state=False,
                                resources={"metadata": {"path": str(run)}})
        client.execute(env={**os.environ, "AQJ_ARTIFACT_DIR": str(run / "outputs/track_e_multifamily")})
        for cell in notebook.cells:
            for output in cell.get("outputs", []):
                if output.output_type == "stream":
                    output.text = output.text.replace(str(Path.home()), "~")
        nbformat.write(notebook, executed / name)
        print(f"{name}: PASS ({perf_counter() - started:.2f}s)", flush=True)
        if args.html:
            export_html(notebook, run / "html" / Path(name).with_suffix(".html"))


if __name__ == "__main__":
    main()
