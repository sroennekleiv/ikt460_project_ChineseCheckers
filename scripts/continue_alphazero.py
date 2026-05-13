# Continue AlphaZero training from the strongest saved checkpoint.

import json
import math
import os
import random
import sys
import time
from collections import deque

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.paths import MPL_CACHE_DIR

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.agents import GreedyAgent, HomeFirstRandomAgent, RandomAgent
from src.alphazero import AlphaZeroAgent
from src.env import ChineseCheckersEnv
from src.paths import (
    AFTERSTATE_BEST_MODEL,
    AFTERSTATE_FINAL_MODEL,
    AFTERSTATE_TRAINED_MODEL,
    ALPHAZERO_BEST_MODEL,
    ALPHAZERO_CHECKPOINT_DIR,
    ALPHAZERO_EXTERNAL_MODEL,
    ALPHAZERO_FINAL_MODEL,
    ALPHAZERO_HISTORY_JSON,
    ALPHAZERO_LEARNING_CURVE,
    ALPHAZERO_TRAINED_MODEL,
    ensure_project_dirs,
    first_existing,
)
from src.perspective import PLAYABLE_COLOR_PAIRS, color_pair_for_game
from src.training import (
    arena,
    arena_against_pool,
    blended_external_score,
    evaluate,
    evaluate_by_pair,
    evaluate_by_pair_with_override,
    evaluate_with_override,
    build_checkpoint_pool,
    load_candidate_agent,
    run_game,
    weakest_pair_summary,
    ARENA_GAMES,
    EVAL_GAMES,
    MAX_TURNS,
)
# These defaults give a medium-sized continuation run that is long enough to
# show a trend, but still practical to rerun when tuning guards or curriculum.
CONTINUE_GAMES      = 1200
SELFPLAY_BATCH      = 12
TRAIN_STEPS         = 100
CHECKPOINT_EVERY    = 5
PROMOTION_THRESHOLD = 55.0
PRINT_WINDOW        = 3
CONTINUATION_CURRICULUM_ROUNDS   = 10
POST_CURRICULUM_TEACHER_INTERVAL  = 6
PROMOTION_SCORE_TOLERANCE         = 12.0
NO_IMPROVE_PATIENCE_ROUNDS        = 24
PAIR_EVAL_GAMES                   = 4
HISTORICAL_POOL_SIZE              = 4
HISTORICAL_SELFPLAY_INTERVAL      = 4
HISTORICAL_LEAGUE_THRESHOLD       = 50.0
HISTORICAL_WEAKEST_THRESHOLD      = 45.0
LANE_SCORE_TOLERANCE              = 8.0

def generate_selfplay_batch(best_agent, agent_for_encoding, games, round_index, greedy_opponent, opponent_pool=None):
    batch_examples = []
    first_wins = second_wins = draws = 0
    pins, steps = [], []
    easy_opponent = HomeFirstRandomAgent(name="ContEasy")
    selfplay_games = 0
    greedy_games = 0
    easy_games = 0
    historical_games = 0
    pool = list(opponent_pool or [])

    # A light teacher mix helps continuation runs avoid drifting into odd
    # first-player habits without giving up the benefits of mostly self-play.
    curriculum_active = round_index < CONTINUATION_CURRICULUM_ROUNDS

    # Even later continuation rounds keep a light teacher mix to avoid drift.
    for game in range(games):
        player_colors = color_pair_for_game(round_index * max(1, games) + game)
        historical_entry = None
        if curriculum_active and game % 3 == 0:
            opponent = greedy_opponent
        elif curriculum_active and game % 3 == 1:
            opponent = easy_opponent
        elif pool and round_index >= 2 and game % HISTORICAL_SELFPLAY_INTERVAL == 0:
            historical_entry = pool[(round_index + game) % len(pool)]
            opponent = historical_entry["agent"]
        elif (not curriculum_active) and game % POST_CURRICULUM_TEACHER_INTERVAL == 0:
            opponent = greedy_opponent
        elif (not curriculum_active) and game % POST_CURRICULUM_TEACHER_INTERVAL == 1:
            opponent = easy_opponent
        else:
            opponent = None

        if opponent is not None:
            if historical_entry is not None:
                historical_games += 1
            elif opponent is greedy_opponent:
                greedy_games += 1
            else:
                easy_games += 1
            if (round_index + game) % 2 == 0:
                result = run_game(best_agent, opponent, agent_for_encoding,
                                  store_colours={player_colors[0]}, noisy=True, player_colors=player_colors)
                trained_colour = player_colors[0]
            else:
                result = run_game(opponent, best_agent, agent_for_encoding,
                                  store_colours={player_colors[1]}, noisy=True, player_colors=player_colors)
                trained_colour = player_colors[1]
        else:
            selfplay_games += 1
            result = run_game(best_agent, best_agent, agent_for_encoding,
                              store_colours=set(player_colors), noisy=True, player_colors=player_colors)
            trained_colour = None

        batch_examples.extend(result["examples"])
        first_wins += 1 if result["winner"] == player_colors[0] else 0
        second_wins += 1 if result["winner"] == player_colors[1] else 0
        draws       += 1 if result["draw"] else 0
        if trained_colour is None:
            pins.append((result["first_pins"] + result["second_pins"]) / 2.0)
        else:
            pins.append(result["pins_by_color"][trained_colour])
        steps.append(result["steps"])

    return {
        "examples":     batch_examples,
        "first_rate":   100.0 * first_wins / max(1, games),
        "second_rate":  100.0 * second_wins / max(1, games),
        "draw_rate":    100.0 * draws / max(1, games),
        "avg_pins":     sum(pins)  / max(1, len(pins)),
        "avg_steps":    sum(steps) / max(1, len(steps)),
        "selfplay_games": selfplay_games,
        "greedy_games": greedy_games,
        "easy_games": easy_games,
        "historical_games": historical_games,
    }

def save_curve(history, label):
    if not history["round_game"]:
        return
    # Each continuation run keeps its own curve file so comparisons stay easy.
    curve_path = str(ALPHAZERO_LEARNING_CURVE).replace(
        ".png", f"_continue_{label}.png"
    )
    plt.figure(figsize=(10, 9))

    plt.subplot(3, 1, 1)
    plt.plot(history["round_game"], history["eval_wins_easy"],   "c-o", ms=3, label="Win % vs EasyRandom")
    plt.plot(history["round_game"], history["eval_wins_greedy"], "g-o", ms=3, label="Win % vs Greedy")
    plt.plot(history["round_game"], history["arena_score"],      "m-o", ms=3, label="Arena score vs best")
    plt.ylabel("Rate (%)")
    plt.ylim(-2, 102)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(history["round_game"], history["eval_pins_easy"],   "c-", label="Pins vs EasyRandom")
    plt.plot(history["round_game"], history["eval_pins_greedy"], "g-", label="Pins vs Greedy")
    plt.ylabel("Avg pins / 10")
    plt.ylim(-0.2, 10.2)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(history["train_game"], history["policy_loss"], "b-", alpha=0.8, label="Policy loss")
    plt.plot(history["train_game"], history["value_loss"],  "r-", alpha=0.8, label="Value loss")
    plt.ylabel("Loss")
    plt.xlabel("Self-play games (total)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(curve_path, dpi=100)
    plt.close()

def main():
    ensure_project_dirs()
    # We deliberately reseed here so two continuation runs do not replay the
    # exact same self-play sequence by accident.
    random.seed(None)
    np.random.seed(None)

    cli_args    = [a for a in sys.argv[1:] if not a.startswith("--")]
    total_games = int(cli_args[0]) if cli_args else CONTINUE_GAMES
    batch_games = SELFPLAY_BATCH
    total_rounds = max(1, math.ceil(total_games / batch_games))

    probe_env = ChineseCheckersEnv(num_players=2, player_colors=["yellow", "purple"], max_turns=MAX_TURNS)
    probe_env.reset()
    state_size = len(probe_env.get_state_for_player("yellow"))

    best_agent = AlphaZeroAgent(state_size=state_size, name="AlphaZeroBest")
    challenger  = AlphaZeroAgent(state_size=state_size, name="AlphaZeroChallenger")
    easy_opponent   = HomeFirstRandomAgent(name="EasyOpp")
    greedy_opponent = GreedyAgent(name="GreedyOpp")
    replay_buffer   = deque(maxlen=best_agent.training_buffer.maxlen)

    print()
    print("AlphaZero Continuation Training")
    print()

    # Continuation should begin from the strongest known tournament checkpoint,
    # not just the most recently promoted model.
    loaded = False
    for candidate in (ALPHAZERO_EXTERNAL_MODEL, ALPHAZERO_BEST_MODEL,
                      ALPHAZERO_FINAL_MODEL, ALPHAZERO_TRAINED_MODEL):
        if candidate.exists() and best_agent.load_model(candidate, verbose=False):
            print(f"  Checkpoint   {candidate.name}   ({best_agent.train_updates} updates)")
            loaded = True
            break
    if not loaded:
        print("ERROR: no AlphaZero checkpoint found — run trainAlphaZero.py first.")
        return

    # Continuation starts from the strongest saved checkpoint, not just the newest one.
    # The guide still helps here because it nudges search toward good moves
    # without taking over the decision process completely.
    guide_path = first_existing(AFTERSTATE_BEST_MODEL, AFTERSTATE_TRAINED_MODEL, AFTERSTATE_FINAL_MODEL)
    guide_enabled = False
    if guide_path.exists():
        guide_enabled = best_agent.enable_afterstate_guide(guide_path, verbose=False)

    challenger.copy_weights_from(best_agent)

    # The baseline should reflect the real MCTS model we are trying to improve,
    # not a hybrid mode that would hide regressions.
    best_agent.afterstate_search_override = False
    baseline_easy_wr, baseline_easy_pins, _ = evaluate(best_agent, easy_opponent)
    baseline_greedy_pair_eval = evaluate_by_pair_with_override(
        best_agent,
        greedy_opponent,
        games_per_pair=PAIR_EVAL_GAMES,
        hybrid=False,
    )
    baseline_greedy_wr = baseline_greedy_pair_eval["overall_win_rate"]
    baseline_greedy_pins = baseline_greedy_pair_eval["overall_pins"]
    best_lane_summary = weakest_pair_summary(baseline_greedy_pair_eval["pairs"])
    best_lane_floor = best_lane_summary["score"]
    best_agent.afterstate_search_override = True

    best_external_score = blended_external_score(
        baseline_greedy_wr, baseline_greedy_pins,
        baseline_easy_wr, baseline_easy_pins,
    )

    # Each continuation run gets its own label so curves do not overwrite each other.
    run_label = str(int(time.time()))[-6:]

    print(f"  Guide        {'yes' if guide_enabled else 'no'}   "
          f"Games  {total_games}  ({total_rounds} rounds × {batch_games})   "
          f"Sims  {best_agent.num_simulations}")
    print(f"  Baseline     Greedy {baseline_greedy_wr:.0f}%  pins {baseline_greedy_pins:.1f}   "
          f"score {best_external_score:.1f}   promote at {PROMOTION_THRESHOLD:.0f}%")
    print(f"  Weakest lane {best_lane_summary['pair']}   score {best_lane_floor:.1f}")
    print()
    print("Training")
    print()

    # Load accumulated history from previous runs so the learning curve is
    # continuous across all training phases, not just this continuation.
    if ALPHAZERO_HISTORY_JSON.exists():
        with open(ALPHAZERO_HISTORY_JSON) as _f:
            history = json.load(_f)
        history_offset = history["round_game"][-1] if history["round_game"] else 0
        print(f"  History      loaded {len(history['round_game'])} prior rounds   offset={history_offset} games")
    else:
        history = {
            "round_game":       [],
            "eval_wins_easy":   [],
            "eval_pins_easy":   [],
            "eval_wins_greedy": [],
            "eval_pins_greedy": [],
            "arena_score":      [],
            "train_game":       [],
            "policy_loss":      [],
            "value_loss":       [],
        }
        history_offset = 0

    last_stats    = {"policy_loss": 0.0, "value_loss": 0.0}
    best_score    = float("-inf")
    best_round    = 0
    promotions    = 0
    rounds_since_external_best = 0
    recent_rounds = deque(maxlen=PRINT_WINDOW)
    start_time    = time.time()
    games_seen    = 0

    for round_index in range(total_rounds):
        games_this_round = min(batch_games, total_games - games_seen)
        games_seen += games_this_round

        historical_pool = build_checkpoint_pool(
            state_size,
            guide_path,
            exclude_paths=(
                ALPHAZERO_BEST_MODEL,
                ALPHAZERO_EXTERNAL_MODEL,
                ALPHAZERO_TRAINED_MODEL,
                ALPHAZERO_FINAL_MODEL,
            ),
            limit=HISTORICAL_POOL_SIZE,
        )

        # One round means: gather self-play, train a challenger, then test it.
        batch_info = generate_selfplay_batch(
            best_agent,
            best_agent,
            games_this_round,
            round_index,
            greedy_opponent,
            opponent_pool=historical_pool,
        )
        replay_buffer.extend(batch_info["examples"])

        challenger.copy_weights_from(best_agent)
        challenger.training_buffer = deque(replay_buffer, maxlen=challenger.training_buffer.maxlen)

        for _ in range(TRAIN_STEPS):
            stats = challenger.train_step()
            if stats is not None:
                last_stats = stats

        # Fair comparison here means pure AlphaZero search with the guide still
        # folded into priors, but without the stronger Afterstate search override.
        best_agent.afterstate_search_override  = False
        easy_wr,   easy_pins,   _ = evaluate_with_override(challenger, easy_opponent,   hybrid=False)
        greedy_pair_eval = evaluate_by_pair_with_override(
            challenger,
            greedy_opponent,
            games_per_pair=PAIR_EVAL_GAMES,
            hybrid=False,
        )
        greedy_wr = greedy_pair_eval["overall_win_rate"]
        greedy_pins = greedy_pair_eval["overall_pins"]
        challenger_lane_summary = weakest_pair_summary(greedy_pair_eval["pairs"])
        arena_result = arena(challenger, best_agent)
        historical_league = arena_against_pool(challenger, historical_pool, games=ARENA_GAMES)
        best_agent.afterstate_search_override  = True

        external_score = blended_external_score(greedy_wr, greedy_pins, easy_wr, easy_pins)
        score         = external_score + 0.25 * arena_result["score_percent"]
        decisive_arena = arena_result["wins"] > arena_result["losses"]
        improved_external = external_score > best_external_score
        promotion_floor = best_external_score - PROMOTION_SCORE_TOLERANCE
        lane_ok = challenger_lane_summary["score"] >= best_lane_floor - LANE_SCORE_TOLERANCE
        league_ok = (
            not historical_pool
            or (
                historical_league["score_percent"] >= HISTORICAL_LEAGUE_THRESHOLD
                and historical_league["weakest"]["score_percent"] >= HISTORICAL_WEAKEST_THRESHOLD
            )
        )

        promoted = (
            arena_result["score_percent"] >= PROMOTION_THRESHOLD
            and decisive_arena
            and (improved_external or external_score >= promotion_floor)
            and lane_ok
            and league_ok
        )

        if promoted:
            best_agent.copy_weights_from(challenger)
            challenger.save_model(ALPHAZERO_BEST_MODEL)
            promotions += 1
            best_lane_floor = challenger_lane_summary["score"]
            best_lane_summary = challenger_lane_summary

        if improved_external:
            challenger.save_model(ALPHAZERO_EXTERNAL_MODEL)
            best_external_score = external_score
            rounds_since_external_best = 0
        else:
            rounds_since_external_best += 1

        challenger.save_model(ALPHAZERO_TRAINED_MODEL)

        if score > best_score:
            best_score = score
            best_round = round_index + 1

        elapsed = int(time.time() - start_time)
        recent_rounds.append((
            batch_info["first_rate"], batch_info["second_rate"],
            batch_info["draw_rate"],   batch_info["avg_pins"],
            batch_info["avg_steps"],
        ))
        avg_first = sum(x[0] for x in recent_rounds) / len(recent_rounds)
        avg_second = sum(x[1] for x in recent_rounds) / len(recent_rounds)
        avg_d     = sum(x[2] for x in recent_rounds) / len(recent_rounds)
        avg_pins  = sum(x[3] for x in recent_rounds) / len(recent_rounds)
        avg_steps = sum(x[4] for x in recent_rounds) / len(recent_rounds)

        print(
            f"  round {round_index + 1:3d}/{total_rounds}   games={games_seen:5d}/{total_games}   "
            f"First={avg_first:4.1f}%   Second={avg_second:4.1f}%   D={avg_d:4.1f}%   "
            f"pins={avg_pins:3.1f}/10   steps={avg_steps:5.1f}   "
            f"pi={last_stats['policy_loss']:+.3f}   v={last_stats['value_loss']:.3f}   t={elapsed}s"
        )
        print(f"       eval EasyRandom  wins={easy_wr:4.1f}%   pins={easy_pins:.1f}/10")
        print(f"       eval Greedy      wins={greedy_wr:4.1f}%   pins={greedy_pins:.1f}/10")
        print(
            f"       weakest lane     {challenger_lane_summary['pair']}   "
            f"score={challenger_lane_summary['score']:.1f}   "
            f"wins={challenger_lane_summary['win_rate']:.1f}%   "
            f"pins={challenger_lane_summary['avg_pins']:.1f}/10"
        )
        print(f"       external score   {external_score:.1f}   (best so far {best_external_score:.1f})")
        print(
            f"       arena vs best    score={arena_result['score_percent']:4.1f}%   "
            f"wins={arena_result['wins']}  losses={arena_result['losses']}  draws={arena_result['draws']}"
        )
        if historical_pool:
            print(
                f"       league vs pool   score={historical_league['score_percent']:4.1f}%   "
                f"matchups={historical_league['matchup_wins']}-{historical_league['matchup_losses']}-"
                f"{historical_league['matchup_draws']}   weakest={historical_league['weakest']['name']} "
                f"({historical_league['weakest']['score_percent']:.1f}%)"
            )
        print(
            f"       batch mix        self-play={batch_info['selfplay_games']:2d}   "
            f"greedy={batch_info['greedy_games']:2d}   easy={batch_info['easy_games']:2d}   "
            f"historical={batch_info['historical_games']:2d}"
        )
        if promoted:
            print("       promoted challenger to best model")
        elif arena_result["score_percent"] >= PROMOTION_THRESHOLD and decisive_arena:
            if not lane_ok:
                print(
                    f"       promotion blocked: weakest lane {challenger_lane_summary['pair']} "
                    f"fell below floor {best_lane_floor - LANE_SCORE_TOLERANCE:.1f}"
                )
            elif not league_ok and historical_pool:
                print(
                    f"       promotion blocked: league score {historical_league['score_percent']:.1f}% "
                    f"or weakest matchup {historical_league['weakest']['score_percent']:.1f}% was too low"
                )
            else:
                print(
                    f"       promotion blocked: external score {external_score:.1f} "
                    f"below floor {promotion_floor:.1f}"
                )
        if improved_external:
            print(f"       *** new best external score — saved {ALPHAZERO_EXTERNAL_MODEL.name}")

        history["round_game"].append(games_seen + history_offset)
        history["eval_wins_easy"].append(easy_wr)
        history["eval_pins_easy"].append(easy_pins)
        history["eval_wins_greedy"].append(greedy_wr)
        history["eval_pins_greedy"].append(greedy_pins)
        history["arena_score"].append(arena_result["score_percent"])
        history["train_game"].append(games_seen + history_offset)
        history["policy_loss"].append(last_stats["policy_loss"])
        history["value_loss"].append(last_stats["value_loss"])
        save_curve(history, run_label)
        with open(ALPHAZERO_HISTORY_JSON, "w") as _f:
            json.dump(history, _f)

        if (round_index + 1) % CHECKPOINT_EVERY == 0:
            ckpt = ALPHAZERO_CHECKPOINT_DIR / f"cont_round{round_index + 1}.pth"
            best_agent.save_model(ckpt)

        if (
            round_index + 1 > CONTINUATION_CURRICULUM_ROUNDS
            and rounds_since_external_best >= NO_IMPROVE_PATIENCE_ROUNDS
        ):
            print(
                f"       early stop: no new external best for {rounds_since_external_best} rounds "
                f"(best {best_external_score:.1f})"
            )
            break

    # The final selection compares the promoted line against the best external
    # checkpoint one more time, so the run finishes with the strongest model
    # instead of blindly trusting whichever one happened to be promoted last.
    pure_easy_wr,   pure_easy_pins,   _ = evaluate_with_override(best_agent, easy_opponent,   hybrid=False)
    pure_greedy_wr, pure_greedy_pins, _ = evaluate_with_override(best_agent, greedy_opponent, hybrid=False)
    pure_greedy_pair_eval = evaluate_by_pair_with_override(
        best_agent,
        greedy_opponent,
        games_per_pair=PAIR_EVAL_GAMES,
        hybrid=False,
    )
    pure_lane_summary = weakest_pair_summary(pure_greedy_pair_eval["pairs"])
    pure_best_score = blended_external_score(pure_greedy_wr, pure_greedy_pins, pure_easy_wr, pure_easy_pins)

    selection_note = "kept promoted best model"
    external_candidate_summary = None
    external_agent = load_candidate_agent(state_size, ALPHAZERO_EXTERNAL_MODEL, guide_path,
                                          name="AlphaZeroExternalCandidate")
    if external_agent is not None:
        ext_easy_wr,   ext_easy_pins,   _ = evaluate_with_override(external_agent, easy_opponent,   hybrid=False)
        ext_greedy_wr, ext_greedy_pins, _ = evaluate_with_override(external_agent, greedy_opponent, hybrid=False)
        ext_score   = blended_external_score(ext_greedy_wr, ext_greedy_pins, ext_easy_wr, ext_easy_pins)
        ext_pair_eval = evaluate_by_pair_with_override(
            external_agent,
            greedy_opponent,
            games_per_pair=PAIR_EVAL_GAMES,
            hybrid=False,
        )
        ext_lane_summary = weakest_pair_summary(ext_pair_eval["pairs"])
        historical_pool = build_checkpoint_pool(
            state_size,
            guide_path,
            exclude_paths=(
                ALPHAZERO_BEST_MODEL,
                ALPHAZERO_EXTERNAL_MODEL,
                ALPHAZERO_TRAINED_MODEL,
                ALPHAZERO_FINAL_MODEL,
            ),
            limit=HISTORICAL_POOL_SIZE,
        )
        ext_arena   = arena(external_agent, best_agent)
        ext_league  = arena_against_pool(external_agent, historical_pool, games=ARENA_GAMES)
        external_candidate_summary = {
            "score":      ext_score,
            "arena":      ext_arena,
            "league":     ext_league,
            "easy_wr":    ext_easy_wr,
            "easy_pins":  ext_easy_pins,
            "greedy_wr":  ext_greedy_wr,
            "greedy_pins": ext_greedy_pins,
            "lane":       ext_lane_summary,
        }
        prefer_external = (
            ext_arena["wins"] > ext_arena["losses"]
            and ext_score >= pure_best_score + 3.0
            and ext_lane_summary["score"] >= pure_lane_summary["score"] - LANE_SCORE_TOLERANCE
            and (
                not historical_pool
                or (
                    ext_league["score_percent"] >= HISTORICAL_LEAGUE_THRESHOLD
                    and ext_league["weakest"]["score_percent"] >= HISTORICAL_WEAKEST_THRESHOLD
                )
            )
        ) or (
            ext_score >= pure_best_score + 8.0
            and ext_arena["score_percent"] >= 50.0
            and ext_lane_summary["score"] >= pure_lane_summary["score"] - LANE_SCORE_TOLERANCE
        )
        if prefer_external:
            best_agent.copy_weights_from(external_agent)
            selection_note = "replaced promoted best with external-best candidate"
            pure_easy_wr, pure_easy_pins     = ext_easy_wr,   ext_easy_pins
            pure_greedy_wr, pure_greedy_pins = ext_greedy_wr, ext_greedy_pins
            pure_best_score = ext_score
            pure_lane_summary = ext_lane_summary
        else:
            selection_note = "kept promoted best over external candidate"

    best_agent.save_model(ALPHAZERO_FINAL_MODEL)
    best_agent.save_model(ALPHAZERO_TRAINED_MODEL)
    best_agent.save_model(ALPHAZERO_BEST_MODEL)

    pure_random_wr,   pure_random_pins,   _ = evaluate_with_override(best_agent, RandomAgent(name="Rnd"), hybrid=False)
    hybrid_easy_wr,   hybrid_easy_pins,   _ = evaluate_with_override(best_agent, easy_opponent,           hybrid=True)
    hybrid_random_wr, hybrid_random_pins, _ = evaluate_with_override(best_agent, RandomAgent(name="Rnd"), hybrid=True)
    hybrid_greedy_wr, hybrid_greedy_pins, _ = evaluate_with_override(best_agent, greedy_opponent,         hybrid=True)

    print()
    print("AlphaZero Continuation Complete")
    print()
    print(f"  Time          {int(time.time() - start_time)}s")
    print(f"  Best round    {best_round}")
    print(f"  Promotions    {promotions}")
    print(f"  Best ext.     {best_external_score:.1f}")
    print(f"  Selection     {selection_note}")
    if external_candidate_summary is not None:
        ec = external_candidate_summary
        print(
            f"  External cand. score {ec['score']:.1f}   "
            f"arena {ec['arena']['wins']}-{ec['arena']['losses']}-{ec['arena']['draws']}"
        )
        print(
            f"  External lane  {ec['lane']['pair']}   score {ec['lane']['score']:.1f}   "
            f"league {ec['league']['score_percent']:.1f}%"
        )
    print()
    print("Final pure MCTS+guide results")
    print()
    print(f"  vs EasyRandom   wins={pure_easy_wr:5.1f}%   pins={pure_easy_pins:.1f}/10")
    print(f"  vs Random       wins={pure_random_wr:5.1f}%   pins={pure_random_pins:.1f}/10")
    print(f"  vs Greedy       wins={pure_greedy_wr:5.1f}%   pins={pure_greedy_pins:.1f}/10")
    print()
    print("Final hybrid results  (MCTS + afterstate override)")
    print()
    print(f"  vs EasyRandom   wins={hybrid_easy_wr:5.1f}%   pins={hybrid_easy_pins:.1f}/10")
    print(f"  vs Random       wins={hybrid_random_wr:5.1f}%   pins={hybrid_random_pins:.1f}/10")
    print(f"  vs Greedy       wins={hybrid_greedy_wr:5.1f}%   pins={hybrid_greedy_pins:.1f}/10")

if __name__ == "__main__":
    main()
