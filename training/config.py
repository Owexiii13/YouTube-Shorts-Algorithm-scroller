"""Central configuration for the offline training pipeline."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MIN_ENTRIES = 500
TRAIN_RATIO = 0.8
MAX_VOCAB_SIZE = 5000
MIN_TOKEN_COUNT = 2
RANDOM_SEED = 42

DEFAULT_DATA_FILE = Path("shorts_ai_data.json")
DEFAULT_VOCAB_FILE = MODELS_DIR / "vocab.json"
METRICS_HISTORY_FILE = MODELS_DIR / "metrics_history.json"

for directory in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
