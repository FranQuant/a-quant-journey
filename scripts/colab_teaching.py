"""Initialize the active Colab kernel before teaching notebook imports."""
import hashlib
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "7c24faa07911499fdc90728729aa04f380c8f411"
PREDICTIONS = {
    "provenance.json": "03af81e4f977ef9c471547bbb371e9fd9182cdd2da7ce801504c4039551ec0ad",
    "validation_metrics.csv": "80e2cad21692eb265c0d870bc88ede18154274742e35b144c4fa6ec1b8ccd3e8",
    "validation_predictions.csv": "0f47e3fa4662e17c01bd8f1c9d475471f4f261ca84d564f342f98309c840440e",
}
PORTFOLIO = {
    "portfolio_equity_curves.csv": "2fb9534ac0e2ce1a2a014d03af5cadc057cb4b1881cf11d3731bc5e81b0122e0",
    "portfolio_provenance.json": "0c9bf4e47598c78c13683b7114b3d78297c5ee67926cea9017fa95a777ef8602",
    "portfolio_summary.csv": "c74afb98cf2f78278f94873e68c8bcea885adbd89dc383ab6f244eaf3f368ab7",
}


def dependencies(number):
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Select a Python 3.12 CPU runtime and rerun setup.")
    restart = "Choose Runtime > Restart session, then run setup again."
    if getattr(sys, "_aqj_restart_required", False):
        raise RuntimeError(restart)
    packages = {"numpy", "pandas", "scipy", "matplotlib", "scikit-learn"}
    if number >= 4:
        packages.update(("nbclient", "nbformat"))
    if number == 5:
        packages.update(("vectorbt", "numba", "llvmlite", "plotly", "quantstats",
                         "seaborn", "yfinance"))
    pins = [line for line in (ROOT / "requirements-vectorbt.lock").read_text().splitlines()
            if line.partition("==")[0] in packages]
    for pin in pins:
        name, expected = pin.split("==")
        try:
            if version(name) == expected:
                continue
        except PackageNotFoundError:
            pass
        wheels = "--only-binary=numpy,scipy,scikit-learn,matplotlib,numba,llvmlite"
        subprocess.run([sys.executable, "-m", "pip", "install", wheels, *pins], check=True)
        sys._aqj_restart_required = True
        raise RuntimeError(restart)


def initialize(number):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    subprocess.run(["git", "diff", "--quiet", BASELINE, revision, "--", "src", "data",
                    "requirements.txt", "requirements-vectorbt.lock",
                    "scripts/execute_teaching_notebooks.py"], cwd=ROOT, check=True)
    print("Teaching checkout", revision)
    dependencies(number)
    if number < 4:
        os.chdir(ROOT)
        return

    prior = ("03_test_predictive_value.ipynb", "04_predictions_to_portfolio_accounting.ipynb")[number - 4]
    marker = ROOT / "outputs/teaching_runs" / f"colab-{revision}-{number}.txt"
    if marker.exists():
        run = Path(marker.read_text())
    else:
        kernel = "aqj-colab-prerequisite"
        subprocess.run([sys.executable, "-m", "ipykernel", "install", "--user", "--name", kernel],
                       check=True)
        result = subprocess.run([sys.executable, "scripts/execute_teaching_notebooks.py",
                                 "--kernel", kernel, prior], cwd=ROOT, check=False,
                                stdout=subprocess.PIPE, text=True,
                                env={**os.environ, "AQJ_COLAB_PREREQUISITE": "1"})
        print(result.stdout)
        result.check_returncode()
        run = Path(next(line.removeprefix("Teaching snapshot: ") for line in result.stdout.splitlines()
                        if line.startswith("Teaching snapshot: ")))
    artifacts = run / "outputs/track_e_multifamily"
    for name, expected in (PREDICTIONS if number == 4 else PREDICTIONS | PORTFOLIO).items():
        actual = hashlib.sha256((artifacts / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"{artifacts / name}: expected {expected}, got {actual}")
    marker.write_text(str(run))
    os.environ["AQJ_ARTIFACT_DIR"] = str(artifacts)
    os.chdir(run)
