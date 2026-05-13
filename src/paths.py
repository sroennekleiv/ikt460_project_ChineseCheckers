# Keep output paths in one place.

from pathlib import Path

# Everything else builds from the project root, so path changes only need to
# happen here.
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
PLOTS_DIR = OUTPUTS_DIR / "plots"

# The training code writes into these folders over and over, so we keep the
# exact layout as named constants instead of scattering string paths around.
AFTERSTATE_MODEL_DIR = MODELS_DIR / "afterstate"
AFTERSTATE_CHECKPOINT_DIR = AFTERSTATE_MODEL_DIR / "checkpoints"
AFTERSTATE_BACKUP_DIR = AFTERSTATE_MODEL_DIR / "backups"
ALPHAZERO_MODEL_DIR = MODELS_DIR / "alphazero"
ALPHAZERO_CHECKPOINT_DIR = ALPHAZERO_MODEL_DIR / "checkpoints"

MPL_CACHE_DIR = ROOT_DIR / ".mplcache"

AFTERSTATE_BEST_MODEL = AFTERSTATE_MODEL_DIR / "best.pth"
AFTERSTATE_TRAINED_MODEL = AFTERSTATE_MODEL_DIR / "trained.pth"
AFTERSTATE_FINAL_MODEL = AFTERSTATE_MODEL_DIR / "final.pth"

ALPHAZERO_BEST_MODEL = ALPHAZERO_MODEL_DIR / "best.pth"
ALPHAZERO_EXTERNAL_MODEL = ALPHAZERO_MODEL_DIR / "external_best.pth"
ALPHAZERO_TRAINED_MODEL = ALPHAZERO_MODEL_DIR / "trained.pth"
ALPHAZERO_FINAL_MODEL = ALPHAZERO_MODEL_DIR / "final.pth"

AFTERSTATE_LEARNING_CURVE = PLOTS_DIR / "afterstate_learning_curve.png"
AFTERSTATE_PHASE2_LEARNING_CURVE = PLOTS_DIR / "afterstate_phase2_learning_curve.png"
ALPHAZERO_LEARNING_CURVE = PLOTS_DIR / "alphazero_learning_curve.png"
ALPHAZERO_HISTORY_JSON   = PLOTS_DIR / "alphazero_history.json"

def ensure_project_dirs():
    # Training scripts call this before saving so a fresh clone can create the
    # whole outputs tree automatically.
    for path in (
        OUTPUTS_DIR,
        MODELS_DIR,
        PLOTS_DIR,
        AFTERSTATE_MODEL_DIR,
        AFTERSTATE_CHECKPOINT_DIR,
        AFTERSTATE_BACKUP_DIR,
        ALPHAZERO_MODEL_DIR,
        ALPHAZERO_CHECKPOINT_DIR,
        MPL_CACHE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)

def first_existing(*paths):
    # Most loaders try a few checkpoint names. This helper returns the first
    # real file and falls back to the first candidate if none exist yet.
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            return candidate
    return Path(paths[0])
