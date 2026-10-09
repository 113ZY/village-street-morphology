import os
import warnings
from pathlib import Path
import sys

warnings.filterwarnings("ignore", message=".*Found Intel OpenMP.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

os.chdir(PROJECT_ROOT)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from village_morphology_clustering.pipeline import main


if __name__ == "__main__":
    main()
