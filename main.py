"""HTR Project command-line entry point.

Usage (VS Code terminal, from HTR_Project/):
    python main.py inspect          # inspect the IAM dataset
    python main.py train-crn ...    # train CRNN (see scripts/train_crnn.py)
    python main.py train-trocr ...  # fine-tune TrOCR
    python main.py evaluate ...     # evaluate both models
    python main.py predict ...      # run inference on an image
    python main.py app              # run the Streamlit UI
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1]
    rest = sys.argv[2:]

    if command == "inspect":
        run([sys.executable, "scripts/inspect_dataset.py", "--examples", "5"] + rest)

    elif command == "train-cnn" or command == "train-crnn":
        run([sys.executable, "scripts/train_crnn.py"] + rest)

    elif command == "train-trocr":
        run([sys.executable, "scripts/train_trocr.py"] + rest)

    elif command == "train":
        run([sys.executable, "src/train.py"] + rest)

    elif command == "evaluate":
        run([sys.executable, "src/evaluate.py"] + rest)

    elif command == "predict":
        run([sys.executable, "scripts/predict.py"] + rest)

    elif command == "app":
        run([sys.executable, "-m", "streamlit", "run", "app.py"] + rest)

    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()