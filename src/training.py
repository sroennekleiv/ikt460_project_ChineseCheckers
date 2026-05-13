# Shared AlphaZero training helpers.

import torch
from pathlib import Path

from src.alphazero import AlphaZeroAgent, AlphaZeroBoardState
from src.env import ChineseCheckersEnv
from src.paths import (
    ALPHAZERO_BEST_MODEL,
    ALPHAZERO_CHECKPOINT_DIR,
    ALPHAZERO_EXTERNAL_MODEL,
    ALPHAZERO_FINAL_MODEL,
    ALPHAZERO_TRAINED_MODEL,
)
from src.perspective import PLAYABLE_COLOR_PAIRS, color_pair_for_game
from src.rewards import model_selection_score

# These defaults are tuned for the two-player training setup used by the
# scripts in this project.
MAX_TURNS                = 400
MAX_REPETITIONS          = 6
SELFPLAY_TEMP_TURNS      = 30
ARENA_GAMES              = 8
EVAL_GAMES               = 10
FINISH_EXAMPLE_REPEATS   = 5
FINISH_PINS_THRESHOLD    = 6
FINISH_DISTANCE_THRESHOLD = 18

def one_hot_policy(agent, action, state=None):
    # Teacher games only need a single chosen move, so we turn that move into
    # a policy target with one active slot.
    policy = torch.zeros(agent.action_size, dtype=torch.float32)
    if action is not None:
        policy[agent.action_to_policy_index(action, state)] = 1.0
    return policy

def select_controller_action(controller, agent, env, state, valid_actions, noisy):
    # AlphaZero can return a full search policy target. Simpler teachers only
    # return one chosen action, so we wrap them into a one-hot target.
    if isinstance(controller, AlphaZeroAgent):
        temperature = 1.0 if noisy and state.move_count < SELFPLAY_TEMP_TURNS else 0.0
        return controller.search_policy(state, temperature=temperature, add_noise=noisy)
    action = controller.choose_action(env, valid_actions)
    return action, one_hot_policy(agent, action, state)

def game_result_value(final_state, winner, player_color):
    if winner is None:
        return 0.25 * final_state.heuristic_value(player_color)
    if winner == player_color:
        return 1.0
    # A flat -1 target made every loss look equally bad. This softer target
    # preserves the idea that some losses are still strategically promising.
    progress = final_state.heuristic_value(player_color)
    return -1.0 + 0.3 * max(0.0, progress)

def finish_example_repeats(winner, player_color, pins_in_goal, distance_to_target):
    # Clean finishing positions are useful training examples, so we replay them
    # a few extra times instead of letting them get lost in the buffer.
    if winner != player_color:
        return 1
    if pins_in_goal >= FINISH_PINS_THRESHOLD or distance_to_target <= FINISH_DISTANCE_THRESHOLD:
        return FINISH_EXAMPLE_REPEATS
    return 1

def run_game(first_controller, second_controller, agent,
             store_colours=None, noisy=False, player_colors=None):
    # This is the shared game runner for self-play, teacher games, eval, and arena.
    player_colors = list(player_colors or ["yellow", "purple"])
    env = ChineseCheckersEnv(
        num_players=2, player_colors=player_colors,
        max_turns=MAX_TURNS, max_repetitions=MAX_REPETITIONS,
    )
    env.reset()
    controllers = {
        player_colors[0]: first_controller,
        player_colors[1]: second_controller,
    }
    store_colours = set(store_colours or [])
    pending_examples = []
    done = False
    info = {}

    while not done:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break
        current_player = env.get_current_player()
        state = AlphaZeroBoardState.from_env(env)
        controller = controllers[current_player]
        action, policy_target = select_controller_action(
            controller, agent, env, state, valid_actions,
            noisy=noisy and isinstance(controller, AlphaZeroAgent),
        )
        if action is None:
            break
        if current_player in store_colours:
            mask = agent.build_mask(
                state.valid_actions(current_player),
                state=state, player_color=current_player,
            )
            pending_examples.append((
                state.state_vector(current_player),
                policy_target,
                current_player,
                mask,
                state.pins_in_goal(current_player),
                state.total_distance_to_target(current_player),
            ))
        _, _, done, info = env.step(action)

    winner = None
    for colour in player_colors:
        if f"{colour} wins" in info.get("message", ""):
            winner = colour
            break

    final_state = AlphaZeroBoardState.from_env(env)
    examples = []
    for sv, pt, colour, mask, pins, dist in pending_examples:
        value_target = game_result_value(final_state, winner, colour)
        repeats = finish_example_repeats(winner, colour, pins, dist)
        for _ in range(repeats):
            examples.append((sv, pt, value_target, mask))

    pins_by_color = {c: env.count_player_pins_in_target(c) for c in player_colors}
    dist_by_color = {c: env.total_distance_to_target(c) for c in player_colors}
    return {
        "examples":       examples,
        "winner":         winner,
        "draw":           winner is None,
        "first_pins":     pins_by_color[player_colors[0]],
        "second_pins":    pins_by_color[player_colors[1]],
        "first_distance": dist_by_color[player_colors[0]],
        "second_distance": dist_by_color[player_colors[1]],
        "pins_by_color":  pins_by_color,
        "distance_by_color": dist_by_color,
        "player_colors":  tuple(player_colors),
        "steps":          env.turn_count,
        "final_state":    final_state,
    }

def evaluate(agent, opponent, games=EVAL_GAMES):
    # Evaluation alternates sides and lanes so one lucky starting side does not
    # make the numbers look better than the policy really is.
    wins, pins, steps = 0, [], []
    for game in range(games):
        player_colors = color_pair_for_game(game)
        if game % 2 == 0:
            result = run_game(agent, opponent, agent,
                              store_colours=set(), noisy=False, player_colors=player_colors)
            agent_colour = player_colors[0]
        else:
            result = run_game(opponent, agent, agent,
                              store_colours=set(), noisy=False, player_colors=player_colors)
            agent_colour = player_colors[1]
        if result["winner"] == agent_colour:
            wins += 1
        pins.append(result["pins_by_color"][agent_colour])
        steps.append(result["steps"])
    return wins / games * 100.0, sum(pins) / games, sum(steps) / games

def evaluate_by_pair(agent, opponent, games_per_pair=4):
    # Pair-by-pair eval is slower, but it catches lane-specific weaknesses that
    # disappear when all colours are merged into one average.
    games_per_pair = max(2, int(games_per_pair))
    if games_per_pair % 2 == 1:
        games_per_pair += 1

    pair_results = {}
    total_wins = 0
    total_pins = 0.0
    total_steps = 0.0
    total_games = 0

    for pair_index, pair in enumerate(PLAYABLE_COLOR_PAIRS):
        pair_key = f"{pair[0]}/{pair[1]}"
        wins = 0
        pins = []
        steps = []

        for game in range(games_per_pair):
            if game % 2 == 0:
                result = run_game(
                    agent,
                    opponent,
                    agent,
                    store_colours=set(),
                    noisy=False,
                    player_colors=pair,
                )
                agent_colour = pair[0]
            else:
                result = run_game(
                    opponent,
                    agent,
                    agent,
                    store_colours=set(),
                    noisy=False,
                    player_colors=pair,
                )
                agent_colour = pair[1]

            if result["winner"] == agent_colour:
                wins += 1
                total_wins += 1
            pins.append(result["pins_by_color"][agent_colour])
            steps.append(result["steps"])
            total_pins += result["pins_by_color"][agent_colour]
            total_steps += result["steps"]
            total_games += 1

        pair_results[pair_key] = {
            "pair": tuple(pair),
            "games": games_per_pair,
            "win_rate": 100.0 * wins / games_per_pair,
            "avg_pins": sum(pins) / games_per_pair,
            "avg_steps": sum(steps) / games_per_pair,
            "pair_index": pair_index,
        }

    return {
        "overall_win_rate": 100.0 * total_wins / max(1, total_games),
        "overall_pins": total_pins / max(1, total_games),
        "overall_steps": total_steps / max(1, total_games),
        "pairs": pair_results,
        "games": total_games,
    }

def evaluate_with_override(agent, opponent, games=EVAL_GAMES, hybrid=False):
    # Some reports want pure MCTS numbers and others want the hybrid override.
    # This helper flips the flag temporarily and then restores it.
    prev = getattr(agent, "afterstate_search_override", None)
    if prev is not None:
        agent.afterstate_search_override = bool(hybrid)
    try:
        return evaluate(agent, opponent, games=games)
    finally:
        if prev is not None:
            agent.afterstate_search_override = prev

def evaluate_by_pair_with_override(agent, opponent, games_per_pair=4, hybrid=False):
    # Pair-by-pair checks need the same temporary override handling as the
    # aggregate evaluation helper above.
    prev = getattr(agent, "afterstate_search_override", None)
    if prev is not None:
        agent.afterstate_search_override = bool(hybrid)
    try:
        return evaluate_by_pair(agent, opponent, games_per_pair=games_per_pair)
    finally:
        if prev is not None:
            agent.afterstate_search_override = prev

def weakest_pair_summary(pair_results):
    # The weakest lane is usually the first place where colour imbalance shows up.
    scored_pairs = []
    for pair_key, pair_result in pair_results.items():
        pair_score = model_selection_score(pair_result["win_rate"], pair_result["avg_pins"])
        scored_pairs.append((pair_score, pair_key, pair_result))

    if not scored_pairs:
        return {
            "pair": "n/a",
            "score": 0.0,
            "win_rate": 0.0,
            "avg_pins": 0.0,
            "avg_steps": 0.0,
        }

    pair_score, pair_key, pair_result = min(scored_pairs, key=lambda item: item[0])
    return {
        "pair": pair_key,
        "score": float(pair_score),
        "win_rate": float(pair_result["win_rate"]),
        "avg_pins": float(pair_result["avg_pins"]),
        "avg_steps": float(pair_result["avg_steps"]),
    }

def blended_external_score(greedy_win_rate, greedy_pins, easy_win_rate, easy_pins):
    # Greedy is the main yardstick, but EasyRandom still catches obvious
    # collapses, so the external score blends both.
    return (
        model_selection_score(greedy_win_rate, greedy_pins)
        + 0.50 * model_selection_score(easy_win_rate, easy_pins)
    )

def load_candidate_agent(state_size, model_path, guide_path, name):
    # Arena code uses this to load historical checkpoints without mutating the
    # main training agent in memory.
    if not model_path.exists():
        return None
    candidate = AlphaZeroAgent(state_size=state_size, name=name)
    if not candidate.load_model(model_path, verbose=False):
        return None
    if guide_path.exists():
        candidate.enable_afterstate_guide(guide_path, verbose=False)
    return candidate

def checkpoint_candidate_paths(exclude_paths=None, limit=4):
    # Historical checkpoints make strong sparring partners because they expose
    # older styles that the current best model may have forgotten how to beat.
    exclude = {
        str(Path(path).resolve())
        for path in (exclude_paths or [])
        if path is not None
    }
    seen = set()
    candidates = []

    ordered_paths = [
        ALPHAZERO_EXTERNAL_MODEL,
        ALPHAZERO_BEST_MODEL,
        ALPHAZERO_TRAINED_MODEL,
        ALPHAZERO_FINAL_MODEL,
        *sorted(ALPHAZERO_CHECKPOINT_DIR.glob("*.pth"), key=lambda path: path.stat().st_mtime, reverse=True),
    ]

    for path in ordered_paths:
        if not path.exists():
            continue
        resolved = str(path.resolve())
        if resolved in exclude or resolved in seen:
            continue
        seen.add(resolved)
        candidates.append(path)
        if len(candidates) >= int(limit):
            break

    return candidates

def build_checkpoint_pool(state_size, guide_path, exclude_paths=None, limit=4):
    # This loads a small pool of old AlphaZero checkpoints for league matches
    # and historical self-play without polluting the main agent in memory.
    pool = []
    for path in checkpoint_candidate_paths(exclude_paths=exclude_paths, limit=limit):
        candidate = load_candidate_agent(
            state_size,
            path,
            guide_path,
            name=f"Hist:{path.stem}",
        )
        if candidate is None:
            continue
        pool.append({
            "name": candidate.name,
            "path": path,
            "agent": candidate,
        })
    return pool

def challenger_score_from_result(result, challenger_colour):
    # Draws still carry signal, so the arena gives partial credit based on the
    # final heuristic margin instead of treating every draw the same.
    challenger_colour = str(challenger_colour)
    if result["winner"] == challenger_colour:
        return 1.0
    if result["winner"] is not None:
        return 0.0
    heuristic = result["final_state"].heuristic_value(challenger_colour)
    return max(0.0, min(1.0, 0.5 + 0.5 * heuristic))

def arena(challenger, champion, games=ARENA_GAMES):
    # Arena is the promotion gate. A challenger must score well against the
    # current best model before we let it replace that model.
    total_score, wins, losses, draws = 0.0, 0, 0, 0
    for game in range(games):
        player_colors = color_pair_for_game(game)
        if game % 2 == 0:
            result = run_game(challenger, champion, challenger,
                              store_colours=set(), noisy=False, player_colors=player_colors)
            challenger_colour = player_colors[0]
        else:
            result = run_game(champion, challenger, challenger,
                              store_colours=set(), noisy=False, player_colors=player_colors)
            challenger_colour = player_colors[1]
        score = challenger_score_from_result(result, challenger_colour)
        total_score += score
        if result["winner"] == challenger_colour:
            wins += 1
        elif result["winner"] is None:
            draws += 1
        else:
            losses += 1
    return {
        "score_percent": 100.0 * total_score / games,
        "wins": wins, "losses": losses, "draws": draws,
    }

def arena_against_pool(challenger, opponent_pool, games=ARENA_GAMES):
    # One arena can be noisy. A tiny league against older checkpoints is a much
    # better test of whether the challenger is actually becoming more general.
    if not opponent_pool:
        return {
            "score_percent": 100.0,
            "matchup_wins": 0,
            "matchup_losses": 0,
            "matchup_draws": 0,
            "matchups": [],
            "weakest": None,
        }

    matchups = []
    for candidate in opponent_pool:
        result = arena(challenger, candidate["agent"], games=games)
        matchups.append({
            "name": candidate["name"],
            "path": str(candidate["path"]),
            **result,
        })

    matchup_wins = sum(1 for result in matchups if result["wins"] > result["losses"])
    matchup_losses = sum(1 for result in matchups if result["wins"] < result["losses"])
    matchup_draws = len(matchups) - matchup_wins - matchup_losses
    weakest = min(matchups, key=lambda result: result["score_percent"])

    return {
        "score_percent": sum(result["score_percent"] for result in matchups) / len(matchups),
        "matchup_wins": matchup_wins,
        "matchup_losses": matchup_losses,
        "matchup_draws": matchup_draws,
        "matchups": matchups,
        "weakest": weakest,
    }
