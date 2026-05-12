# Train AlphaZero from scratch.

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
from src.alphazero import AlphaZeroAgent, AlphaZeroBoardState
from src.env import ChineseCheckersEnv
from src.paths import (
    ALPHAZERO_BEST_MODEL,
    ALPHAZERO_CHECKPOINT_DIR,
    ALPHAZERO_EXTERNAL_MODEL,
    ALPHAZERO_FINAL_MODEL,
    ALPHAZERO_LEARNING_CURVE,
    ALPHAZERO_TRAINED_MODEL,
    AFTERSTATE_BEST_MODEL,
    AFTERSTATE_FINAL_MODEL,
    AFTERSTATE_TRAINED_MODEL,
    ensure_project_dirs,
    first_existing,
)
from src.perspective import PLAYABLE_COLOR_PAIRS, color_pair_for_game
from src.training import (
    arena,
    blended_external_score,
    challenger_score_from_result,
    evaluate,
    evaluate_with_override,
    finish_example_repeats,
    game_result_value,
    load_candidate_agent,
    one_hot_policy,
    run_game,
    select_controller_action,
    ARENA_GAMES,
    EVAL_GAMES,
    MAX_TURNS,
)

TOTAL_SELFPLAY_GAMES   = 600
WARMUP_GAMES           = 120
SELFPLAY_BATCH_GAMES   = 12
TRAIN_STEPS_PER_ROUND  = 100
CHECKPOINT_EVERY_ROUNDS = 5
CURRICULUM_ROUNDS      = 30
PROMOTION_THRESHOLD    = 55.0
PRINT_WINDOW           = 3
SHORT_TEST_SIMULATIONS = 12
SHORT_TEST_EVAL_GAMES  = 3

def greedy_warmup(agent, games, better_teacher=None):
    if games <= 0:
        return []

    print()
    teacher_name = better_teacher.name if better_teacher is not None else "GreedyAgent"
    print(f"Warm-start   {games} games   teacher={teacher_name}")
    # Warm-start fills the replay buffer with sensible examples before raw
    # self-play takes over.
    teacher = better_teacher or GreedyAgent(name="WarmupTeacher")
    easy = HomeFirstRandomAgent(name="WarmupEasy")

    warmup_examples = []
    batch_states = []
    batch_policies = []
    batch_masks = []
    batch_values = []
    teacher_wins = 0

    for game in range(games):
        player_colors = color_pair_for_game(game)
        teacher_colour = player_colors[0] if game % 2 == 0 else player_colors[1]
        if game % 2 == 0:
            result = run_game(teacher, easy, agent, store_colours={teacher_colour}, noisy=False, player_colors=player_colors)
        else:
            result = run_game(easy, teacher, agent, store_colours={teacher_colour}, noisy=False, player_colors=player_colors)

        warmup_examples.extend(result["examples"])
        for state_vector, policy_target, value_target, mask in result["examples"]:
            batch_states.append(state_vector)
            batch_policies.append(policy_target)
            batch_masks.append(mask)
            batch_values.append(value_target)

        if result["winner"] == teacher_colour:
            teacher_wins += 1

        if (game + 1) % 10 == 0 and batch_states:
            loss = agent.supervised_update(batch_states, batch_policies, batch_masks,
                                           value_targets=batch_values, epochs=2)
            print(
                f"  {game + 1:3d}/{games}   samples {len(batch_states)}   "
                f"wins {teacher_wins}/{game + 1}   loss {loss:.3f}"
            )
            batch_states = []
            batch_policies = []
            batch_masks = []
            batch_values = []

    if batch_states:
        loss = agent.supervised_update(batch_states, batch_policies, batch_masks,
                                       value_targets=batch_values, epochs=2)
        print(f"  final warmup batch   samples {len(batch_states)}   loss {loss:.3f}")

    print()
    return warmup_examples

def generate_selfplay_batch(best_agent, agent_for_encoding, games, round_index, greedy_opponent):
    batch_examples = []
    first_wins = 0
    second_wins = 0
    draws = 0
    pins = []
    steps = []

    curriculum_active = round_index < CURRICULUM_ROUNDS
    easy_opponent = HomeFirstRandomAgent(name="CurriculumEasy")

    # Early rounds lean on teachers more often so self-play has better shape.
    for game in range(games):
        player_colors = color_pair_for_game(round_index * max(1, games) + game)
        if curriculum_active and round_index < 8 and game % 3 != 2:
            opponent = easy_opponent
        elif curriculum_active and game % 2 == 0:
            opponent = greedy_opponent
        elif game % 4 == 0:
            opponent = greedy_opponent
        else:
            opponent = None

        if opponent is not None:
            if (round_index + game) % 2 == 0:
                result = run_game(
                    best_agent, opponent, agent_for_encoding,
                    store_colours={player_colors[0]}, noisy=True, player_colors=player_colors,
                )
                trained_colour = player_colors[0]
            else:
                result = run_game(
                    opponent, best_agent, agent_for_encoding,
                    store_colours={player_colors[1]}, noisy=True, player_colors=player_colors,
                )
                trained_colour = player_colors[1]
        else:
            result = run_game(
                best_agent, best_agent, agent_for_encoding,
                store_colours=set(player_colors), noisy=True, player_colors=player_colors,
            )
            trained_colour = None

        batch_examples.extend(result["examples"])
        first_wins += 1 if result["winner"] == player_colors[0] else 0
        second_wins += 1 if result["winner"] == player_colors[1] else 0
        draws += 1 if result["draw"] else 0
        if trained_colour is None:
            pins.append((result["first_pins"] + result["second_pins"]) / 2.0)
        else:
            pins.append(result["pins_by_color"][trained_colour])
        steps.append(result["steps"])

    return {
        "examples": batch_examples,
        "first_rate": 100.0 * first_wins / max(1, games),
        "second_rate": 100.0 * second_wins / max(1, games),
        "draw_rate": 100.0 * draws / max(1, games),
        "avg_pins": sum(pins) / max(1, len(pins)),
        "avg_steps": sum(steps) / max(1, len(steps)),
    }

def save_curve(history):
    if not history["round_game"]:
        return

    # The AlphaZero curve tracks eval strength and training losses together.
    plt.figure(figsize=(10, 9))

    plt.subplot(3, 1, 1)
    plt.plot(history["round_game"], history["eval_wins_easy"], "c-o", ms=3, label="Win % vs EasyRandom")
    plt.plot(history["round_game"], history["eval_wins_greedy"], "g-o", ms=3, label="Win % vs Greedy")
    plt.plot(history["round_game"], history["arena_score"], "m-o", ms=3, label="Arena score vs best")
    plt.ylabel("Rate (%)")
    plt.ylim(-2, 102)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(history["round_game"], history["eval_pins_easy"], "c-", label="Pins vs EasyRandom")
    plt.plot(history["round_game"], history["eval_pins_greedy"], "g-", label="Pins vs Greedy")
    plt.ylabel("Avg pins / 10")
    plt.ylim(-0.2, 10.2)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(history["train_game"], history["policy_loss"], "b-", alpha=0.8, label="Policy loss")
    plt.plot(history["train_game"], history["value_loss"], "r-", alpha=0.8, label="Value loss")
    plt.ylabel("Loss")
    plt.xlabel("Self-play games")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(ALPHAZERO_LEARNING_CURVE, dpi=100)
    plt.close()

def main():
    ensure_project_dirs()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    cli_args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    continue_training = "--continue" in sys.argv
    refresh_warmup = "--refresh-warmup" in sys.argv
    total_games = int(cli_args[0]) if cli_args else TOTAL_SELFPLAY_GAMES
    batch_games = min(SELFPLAY_BATCH_GAMES, max(1, total_games))
    total_rounds = max(1, math.ceil(total_games / batch_games))
    warmup_games = WARMUP_GAMES if (not continue_training or refresh_warmup) else 0
    eval_games = SHORT_TEST_EVAL_GAMES if total_games <= 60 else EVAL_GAMES
    arena_games = SHORT_TEST_EVAL_GAMES if total_games <= 60 else ARENA_GAMES

    probe_env = ChineseCheckersEnv(num_players=2, player_colors=["yellow", "purple"], max_turns=MAX_TURNS)
    probe_env.reset()
    state_size = len(probe_env.get_state_for_player("yellow"))

    best_agent = AlphaZeroAgent(state_size=state_size, name="AlphaZeroBest")
    challenger = AlphaZeroAgent(state_size=state_size, name="AlphaZeroChallenger")
    if total_games <= 60:
        best_agent.num_simulations = SHORT_TEST_SIMULATIONS
        challenger.num_simulations = SHORT_TEST_SIMULATIONS
    easy_opponent = HomeFirstRandomAgent(name="EasyOpp")
    greedy_opponent = GreedyAgent(name="GreedyOpp")
    replay_buffer = deque(maxlen=best_agent.training_buffer.maxlen)

    print()
    print("Chinese Checkers   AlphaZero Training")
    print()
    if continue_training and ALPHAZERO_BEST_MODEL.exists():
        if best_agent.load_model(ALPHAZERO_BEST_MODEL, verbose=False):
            print("Continuing from previous AlphaZero best model.")
            if total_games <= 60:
                best_agent.num_simulations = SHORT_TEST_SIMULATIONS
        print()

    guide_path = first_existing(AFTERSTATE_BEST_MODEL, AFTERSTATE_TRAINED_MODEL, AFTERSTATE_FINAL_MODEL)
    guide_enabled = False
    warmup_teacher = None
    if guide_path.exists():
        guide_enabled = best_agent.enable_afterstate_guide(guide_path)
        if guide_enabled and best_agent.afterstate_search_agent is not None:
            # If the Afterstate search agent is available, it gives much richer
            # demonstrations than plain greedy play and makes the initial policy
            # network far less random.
            warmup_teacher = best_agent.afterstate_search_agent
            warmup_teacher.epsilon = 0.0

    last_stats = {"policy_loss": 0.0, "value_loss": 0.0, "loss": 0.0}
    warmup_examples = greedy_warmup(best_agent, warmup_games, better_teacher=warmup_teacher)
    replay_buffer.extend(warmup_examples)
    best_agent.add_examples(warmup_examples)
    warmup_updates = 0
    if warmup_examples:
        # Warmup examples get extra updates so round one is not nearly random.
        warmup_updates = min(200, max(20, len(warmup_examples) // max(1, best_agent.batch_size) * 4))
        for _ in range(warmup_updates):
            stats = best_agent.train_step()
            if stats is not None:
                last_stats = stats

    if not continue_training or not ALPHAZERO_BEST_MODEL.exists():
        best_agent.save_model(ALPHAZERO_BEST_MODEL)

    challenger.copy_weights_from(best_agent)

    # The baseline is measured with pure MCTS plus guide only. That way the
    # number we compare against later matches the promotion arena exactly.
    baseline_easy_win_rate, baseline_easy_pins, _ = evaluate_with_override(
        best_agent,
        easy_opponent,
        games=eval_games,
        hybrid=False,
    )
    baseline_greedy_win_rate, baseline_greedy_pins, _ = evaluate_with_override(
        best_agent,
        greedy_opponent,
        games=eval_games,
        hybrid=False,
    )
    best_external_score = blended_external_score(
        baseline_greedy_win_rate,
        baseline_greedy_pins,
        baseline_easy_win_rate,
        baseline_easy_pins,
    )

    print("Training Setup")
    print()
    print(f"  Self-play games   {total_games}")
    print(f"  Rounds            {total_rounds}   ({batch_games} games per round)")
    print(f"  State             {state_size} features")
    print(f"  Search            {best_agent.num_simulations} MCTS simulations   c_puct {best_agent.c_puct}")
    print(f"  Continue          {'yes' if continue_training else 'no'}")
    print(f"  Refresh warmup    {'yes' if refresh_warmup else 'no'}")
    print(f"  Afterstate guide  {'yes' if guide_enabled else 'no'}   weight {best_agent.afterstate_guide_weight:.1f}")
    print(f"  Color lanes       {', '.join(f'{a}/{b}' for a, b in PLAYABLE_COLOR_PAIRS)}")
    warmup_teacher_name = warmup_teacher.name if warmup_teacher is not None else "GreedyAgent"
    print(f"  Warmup            {warmup_games} games  teacher={warmup_teacher_name}   {warmup_updates} value updates")
    print(f"  Training          {TRAIN_STEPS_PER_ROUND} challenger updates per round")
    print(f"  Eval games        {eval_games}")
    print(f"  Arena             {arena_games} games per round   promote at {PROMOTION_THRESHOLD:.1f}%")
    print(f"  Current best      external score {best_external_score:.1f}")
    print()
    print("Training")
    print()

    history = {
        "round_game": [],
        "eval_wins_easy": [],
        "eval_pins_easy": [],
        "eval_wins_greedy": [],
        "eval_pins_greedy": [],
        "arena_score": [],
        "train_game": [],
        "policy_loss": [],
        "value_loss": [],
    }

    best_score = float("-inf")
    best_round = 0
    promoted_rounds = 0
    recent_rounds = deque(maxlen=PRINT_WINDOW)
    start_time = time.time()
    games_seen = 0

    for round_index in range(total_rounds):
        games_this_round = min(batch_games, total_games - games_seen)
        games_seen += games_this_round

        batch_info = generate_selfplay_batch(best_agent, best_agent, games_this_round, round_index, greedy_opponent)
        replay_buffer.extend(batch_info["examples"])

        challenger.copy_weights_from(best_agent)
        challenger.training_buffer = deque(replay_buffer, maxlen=challenger.training_buffer.maxlen)

        for _ in range(TRAIN_STEPS_PER_ROUND):
            stats = challenger.train_step()
            if stats is not None:
                last_stats = stats

        # Arena results should measure the AlphaZero policy and search, not a
        # full handoff to Afterstate search. The additive guide stays on, but
        # the hard override is switched off for fair evaluation.
        challenger.afterstate_search_override = False
        best_agent.afterstate_search_override = False
        easy_win_rate, easy_pins, _ = evaluate(challenger, easy_opponent, games=eval_games)
        greedy_win_rate, greedy_pins, _ = evaluate(challenger, greedy_opponent, games=eval_games)
        arena_result = arena(challenger, best_agent, games=arena_games)
        challenger.afterstate_search_override = True
        best_agent.afterstate_search_override = True

        external_score = blended_external_score(
            greedy_win_rate,
            greedy_pins,
            easy_win_rate,
            easy_pins,
        )
        score = external_score + 0.25 * arena_result["score_percent"]
        improved_external = external_score > best_external_score
        decisive_arena = arena_result["wins"] > arena_result["losses"]
        # Promotion is meant to reward clearly stronger play, not one lucky
        # metric spike. The arena score and external score work together so the
        # best model is both stronger in matches and stronger in evaluation.
        promoted = (
            arena_result["score_percent"] >= PROMOTION_THRESHOLD
            and (decisive_arena or external_score >= best_external_score - 5.0)
        )

        if promoted:
            best_agent.copy_weights_from(challenger)
            challenger.save_model(ALPHAZERO_BEST_MODEL)
            promoted_rounds += 1

        if improved_external:
            challenger.save_model(ALPHAZERO_EXTERNAL_MODEL)
            best_external_score = external_score

        challenger.save_model(ALPHAZERO_TRAINED_MODEL)

        if score > best_score:
            best_score = score
            best_round = round_index + 1

        elapsed = int(time.time() - start_time)
        recent_rounds.append(
            (
                batch_info["first_rate"],
                batch_info["second_rate"],
                batch_info["draw_rate"],
                batch_info["avg_pins"],
                batch_info["avg_steps"],
            )
        )
        avg_first = sum(item[0] for item in recent_rounds) / len(recent_rounds)
        avg_second = sum(item[1] for item in recent_rounds) / len(recent_rounds)
        avg_draw = sum(item[2] for item in recent_rounds) / len(recent_rounds)
        avg_pins = sum(item[3] for item in recent_rounds) / len(recent_rounds)
        avg_steps = sum(item[4] for item in recent_rounds) / len(recent_rounds)
        print(
            f"  round {round_index + 1:3d}/{total_rounds}   games={games_seen:4d}/{total_games}   "
            f"First={avg_first:4.1f}%   Second={avg_second:4.1f}%   D={avg_draw:4.1f}%   "
            f"pins={avg_pins:3.1f}/10   steps={avg_steps:5.1f}   "
            f"pi={last_stats['policy_loss']:+.3f}   v={last_stats['value_loss']:.3f}   t={elapsed}s"
        )
        print(f"       eval EasyRandom  wins={easy_win_rate:4.1f}%   pins={easy_pins:.1f}/10")
        print(f"       eval Greedy      wins={greedy_win_rate:4.1f}%   pins={greedy_pins:.1f}/10")
        print(f"       external score   {external_score:.1f}")
        print(
            f"       arena vs best    score={arena_result['score_percent']:4.1f}%   "
            f"wins={arena_result['wins']}  losses={arena_result['losses']}  draws={arena_result['draws']}"
        )
        if arena_result["score_percent"] >= PROMOTION_THRESHOLD and not promoted:
            print("       promotion blocked: arena threshold met but not decisive and external score dropped")
        if promoted:
            print("       promoted challenger to best model")
        if improved_external:
            print(f"       saved challenger as external candidate ({ALPHAZERO_EXTERNAL_MODEL.name})")

        history["round_game"].append(games_seen)
        history["eval_wins_easy"].append(easy_win_rate)
        history["eval_pins_easy"].append(easy_pins)
        history["eval_wins_greedy"].append(greedy_win_rate)
        history["eval_pins_greedy"].append(greedy_pins)
        history["arena_score"].append(arena_result["score_percent"])
        history["train_game"].append(games_seen)
        history["policy_loss"].append(last_stats["policy_loss"])
        history["value_loss"].append(last_stats["value_loss"])
        save_curve(history)

        if (round_index + 1) % CHECKPOINT_EVERY_ROUNDS == 0:
            checkpoint_path = ALPHAZERO_CHECKPOINT_DIR / f"phase_round{round_index + 1}.pth"
            best_agent.save_model(checkpoint_path)

    pure_easy_win_rate, pure_easy_pins, _ = evaluate_with_override(
        best_agent,
        easy_opponent,
        games=eval_games,
        hybrid=False,
    )
    pure_greedy_win_rate, pure_greedy_pins, _ = evaluate_with_override(
        best_agent,
        greedy_opponent,
        games=eval_games,
        hybrid=False,
    )
    pure_best_score = blended_external_score(
        pure_greedy_win_rate,
        pure_greedy_pins,
        pure_easy_win_rate,
        pure_easy_pins,
    )

    selection_note = "kept promoted best model"
    external_candidate_summary = None
    external_agent = load_candidate_agent(
        state_size,
        ALPHAZERO_EXTERNAL_MODEL,
        guide_path,
        name="AlphaZeroExternalCandidate",
    )
    if external_agent is not None:
        external_easy_win_rate, external_easy_pins, _ = evaluate_with_override(
            external_agent,
            easy_opponent,
            games=eval_games,
            hybrid=False,
        )
        external_greedy_win_rate, external_greedy_pins, _ = evaluate_with_override(
            external_agent,
            greedy_opponent,
            games=eval_games,
            hybrid=False,
        )
        external_score = blended_external_score(
            external_greedy_win_rate,
            external_greedy_pins,
            external_easy_win_rate,
            external_easy_pins,
        )
        external_arena = arena(external_agent, best_agent, games=arena_games)
        external_candidate_summary = {
            "score": external_score,
            "arena": external_arena,
            "easy_win_rate": external_easy_win_rate,
            "easy_pins": external_easy_pins,
            "greedy_win_rate": external_greedy_win_rate,
            "greedy_pins": external_greedy_pins,
        }
        prefer_external = (
            external_arena["wins"] > external_arena["losses"]
            and external_score >= pure_best_score - 5.0
        ) or (
            external_score >= pure_best_score + 10.0
            and external_arena["score_percent"] >= 50.0
        )
        if prefer_external:
            best_agent.copy_weights_from(external_agent)
            selection_note = "replaced promoted best with stronger external candidate"
            pure_easy_win_rate = external_easy_win_rate
            pure_easy_pins = external_easy_pins
            pure_greedy_win_rate = external_greedy_win_rate
            pure_greedy_pins = external_greedy_pins
            pure_best_score = external_score
            best_agent.save_model(ALPHAZERO_BEST_MODEL)
        else:
            selection_note = "kept promoted best over external candidate"

    best_agent.save_model(ALPHAZERO_FINAL_MODEL)
    best_agent.save_model(ALPHAZERO_TRAINED_MODEL)

    pure_random_win_rate, pure_random_pins, _ = evaluate_with_override(
        best_agent,
        RandomAgent(name="RandomOpp"),
        games=eval_games,
        hybrid=False,
    )

    final_easy_win_rate, final_easy_pins, _ = evaluate_with_override(
        best_agent,
        easy_opponent,
        games=eval_games,
        hybrid=True,
    )
    final_random_win_rate, final_random_pins, _ = evaluate_with_override(
        best_agent,
        RandomAgent(name="RandomOpp"),
        games=eval_games,
        hybrid=True,
    )
    final_greedy_win_rate, final_greedy_pins, _ = evaluate_with_override(
        best_agent,
        greedy_opponent,
        games=eval_games,
        hybrid=True,
    )

    print()
    print("AlphaZero Training Complete")
    print()
    print(f"  Time            {int(time.time() - start_time)}s")
    print(f"  Best round      {best_round}")
    print(f"  Promotions      {promoted_rounds}")
    print(f"  Train updates   best {best_agent.train_updates}   challenger {challenger.train_updates}")
    print(f"  Best model      {ALPHAZERO_BEST_MODEL}")
    print(f"  Final selection {selection_note}")
    print(f"  Pure best score {pure_best_score:.1f}")
    if external_candidate_summary is not None:
        print(
            "  External cand.  "
            f"score {external_candidate_summary['score']:.1f}   "
            f"arena {external_candidate_summary['arena']['wins']}-"
            f"{external_candidate_summary['arena']['losses']}-"
            f"{external_candidate_summary['arena']['draws']}"
        )
    print()
    print("Final pure MCTS+guide results")
    print()
    print(f"  vs EasyRandom   wins={pure_easy_win_rate:4.1f}%   pins={pure_easy_pins:.1f}/10")
    print(f"  vs Random       wins={pure_random_win_rate:4.1f}%   pins={pure_random_pins:.1f}/10")
    print(f"  vs Greedy       wins={pure_greedy_win_rate:4.1f}%   pins={pure_greedy_pins:.1f}/10")
    print()
    print("Final hybrid results")
    print()
    print(f"  vs EasyRandom   wins={final_easy_win_rate:4.1f}%   pins={final_easy_pins:.1f}/10")
    print(f"  vs Random       wins={final_random_win_rate:4.1f}%   pins={final_random_pins:.1f}/10")
    print(f"  vs Greedy       wins={final_greedy_win_rate:4.1f}%   pins={final_greedy_pins:.1f}/10")

if __name__ == "__main__":
    main()
