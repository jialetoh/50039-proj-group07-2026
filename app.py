from datetime import datetime
from pathlib import Path

import modal


APP_NAME = "notebook-04-runner"
RESULTS_VOLUME_NAME = "proj-group07-2026-results"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("jupyter", "ipykernel", "nbconvert")
    .add_local_dir("src", remote_path="/workspace/src")
    .add_local_dir("notebooks", remote_path="/workspace/notebooks")
    .add_local_dir("data", remote_path="/workspace/data")
)

app = modal.App(APP_NAME)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    volumes={"/results": results_volume},
    gpu="A10G",
    timeout=60 * 60 * 8,
)
def run_tuned_pretrained_encoder():
    import os
    import shutil
    import subprocess

    work_root = Path("/workspace")
    (work_root / "checkpoints").mkdir(parents=True, exist_ok=True)

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("/results") / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    notebooks_dir = work_root / "notebooks"
    output_name = "04_tuned_pretrained_encoder.executed.ipynb"
    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "04_tuned_pretrained_encoder.ipynb",
        "--output",
        output_name,
        "--output-dir",
        str(run_dir),
        "--ExecutePreprocessor.timeout=-1",
        "--ExecutePreprocessor.kernel_name=python3",
    ]
    subprocess.run(cmd, cwd=notebooks_dir, env=env, check=True)

    checkpoints_src = work_root / "checkpoints"
    checkpoints_dst = run_dir / "checkpoints"
    if checkpoints_src.exists():
        shutil.copytree(checkpoints_src, checkpoints_dst, dirs_exist_ok=True)

    test_results_src = notebooks_dir / "test_results.pkl"
    test_results_dst = run_dir / "test_results.pkl"
    if test_results_src.exists():
        shutil.copy2(test_results_src, test_results_dst)

    results_volume.commit()

    checkpoint_count = 0
    if checkpoints_dst.exists():
        checkpoint_count = len(list(checkpoints_dst.glob("*.pth")))

    return {
        "run_dir": str(run_dir),
        "executed_notebook": str(run_dir / output_name),
        "has_test_results": test_results_dst.exists(),
        "checkpoint_count": checkpoint_count,
    }


@app.local_entrypoint()
def main():
    result = run_tuned_pretrained_encoder.remote()
    print("Notebook execution finished on Modal.")
    print(result)
    print(
        "Download artifacts with: "
        "modal volume get "
        f"{RESULTS_VOLUME_NAME} "
        f"{result['run_dir'].replace('/results', '')} "
        "./outputs/modal"
    )