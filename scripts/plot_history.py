# Plot the full AlphaZero training history as one continuous chart.
# Reads the accumulated JSON written by train_alphazero.py and
# continue_alphazero.py.  Run from the project root:
#   python scripts/plot_history.py

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.paths import ALPHAZERO_HISTORY_JSON, MPL_CACHE_DIR, PLOTS_DIR

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_PATH = PLOTS_DIR / "alphazero_full_history.png"


def main():
    if not ALPHAZERO_HISTORY_JSON.exists():
        print(
            f"No history file found at {ALPHAZERO_HISTORY_JSON}.\n"
            "Run at least one round of train_alphazero.py or continue_alphazero.py first."
        )
        return

    with open(ALPHAZERO_HISTORY_JSON) as f:
        h = json.load(f)

    if not h.get("round_game"):
        print("History file exists but contains no data yet.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    fig.suptitle("AlphaZero Training History — Full Run", fontsize=13, fontweight="bold")

    # ── panel 1: win rates ──────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(h["round_game"], h["eval_wins_easy"],   "c-o", ms=3, label="Win % vs EasyRandom")
    ax.plot(h["round_game"], h["eval_wins_greedy"], "g-o", ms=3, label="Win % vs Greedy")
    ax.plot(h["round_game"], h["arena_score"],      "m-o", ms=3, label="Arena score vs best")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(-2, 102)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── panel 2: pins ───────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(h["round_game"], h["eval_pins_easy"],   "c-", label="Avg pins vs EasyRandom")
    ax.plot(h["round_game"], h["eval_pins_greedy"], "g-", label="Avg pins vs Greedy")
    ax.set_ylabel("Avg pins / 10")
    ax.set_ylim(-0.2, 10.2)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── panel 3: loss ───────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(h["train_game"], h["policy_loss"], "b-", alpha=0.8, label="Policy loss")
    ax.plot(h["train_game"], h["value_loss"],  "r-", alpha=0.8, label="Value loss")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Self-play games (cumulative)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(OUTPUT_PATH), dpi=120)
    plt.close()
    print(f"Saved → {OUTPUT_PATH}   ({len(h['round_game'])} data points)")


if __name__ == "__main__":
    main()
