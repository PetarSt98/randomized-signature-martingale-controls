"""Run the complete battery delivery study from any working directory."""
import os
import subprocess
import sys
from pathlib import Path


def main():
    folder = Path(__file__).resolve().parent
    env = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        env.setdefault(name, "1")

    def run(script, *args):
        subprocess.run([sys.executable, str(folder / script), *args], env=env, check=True)

    for split in ("1", "2", "3"):
        run("run_pilot.py", "--split", split)
        run("raw_history.py", "--split", split)
        run("confirm.py", "--split", split)
    run("run_pilot.py", "--summarize")
    run("confirm.py", "--summary")
    run("observer.py")
    run("duration_check.py")
    run("make_report.py")


if __name__ == "__main__":
    main()
