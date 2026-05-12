# Quick evaluation script for baseline, Afterstate, and AlphaZero agents.

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# The default run stays reasonably quick. Full AlphaZero checks are opt-in
# because MCTS makes them much slower than the baseline agents.
STANDARD_GAMES = 20
GREEDY_GAMES = 30
AFTERSTATE_MODEL_NAME = None
RUN_ALPHAZERO_EVAL = "--alphazero" in sys.argv or "--full-alphazero" in sys.argv
FULL_ALPHAZERO_EVAL = "--full-alphazero" in sys.argv
ALPHAZERO_FAST_GAMES = 4
ALPHAZERO_FAST_SIMS = 12

positional_args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
if len(positional_args) > 0:
    GREEDY_GAMES = int(positional_args[0])
if len(positional_args) > 1:
    AFTERSTATE_MODEL_NAME = positional_args[1]

from src.afterstate import AfterstateSearchAgent, AfterstateValueAgent
from src.agents import GreedyAgent, HomeFirstRandomAgent, RandomAgent
from src.alphazero import AlphaZeroAgent
from src.env import ChineseCheckersEnv
from src.paths import (
    ALPHAZERO_BEST_MODEL,
    ALPHAZERO_EXTERNAL_MODEL,
    ALPHAZERO_FINAL_MODEL,
    ALPHAZERO_TRAINED_MODEL,
    AFTERSTATE_BEST_MODEL,
    AFTERSTATE_FINAL_MODEL,
    AFTERSTATE_TRAINED_MODEL,
    first_existing,
)
from src.perspective import color_pair_for_game

def _explicit_model_path(model_name):
    # Allow either an absolute path or a project-relative path on the command line.
    if not model_name:
        return None
    candidate = model_name if os.path.isabs(model_name) else os.path.join(PROJECT_ROOT, model_name)
    return candidate if os.path.exists(candidate) else None

def load_afterstate_agent(model_name=None):
    # We probe a fresh environment just to recover the current state size.
    explicit_path = _explicit_model_path(model_name)
    if explicit_path:
        model_path = explicit_path
    else:
        model_path = first_existing(
            AFTERSTATE_BEST_MODEL,
            AFTERSTATE_TRAINED_MODEL,
            AFTERSTATE_FINAL_MODEL,
        )
    if not os.path.exists(model_path):
        return None
    env = ChineseCheckersEnv(num_players=2, player_colors=["yellow", "purple"], max_turns=500)
    env.reset()
    agent = AfterstateValueAgent(
        state_size=len(env.get_state_for_player("yellow")),
        player_color="yellow",
        name="AfterstateAgent",
    )
    if not agent.load_model(model_path):
        return None
    agent.epsilon = 0.0
    return agent

def load_afterstate_search_agent(model_name=None):
    explicit_path = _explicit_model_path(model_name)
    if explicit_path:
        model_path = explicit_path
    else:
        model_path = first_existing(
            AFTERSTATE_BEST_MODEL,
            AFTERSTATE_TRAINED_MODEL,
            AFTERSTATE_FINAL_MODEL,
        )
    if not os.path.exists(model_path):
        return None
    env = ChineseCheckersEnv(num_players=2, player_colors=["yellow", "purple"], max_turns=500)
    env.reset()
    agent = AfterstateSearchAgent(
        state_size=len(env.get_state_for_player("yellow")),
        player_color="yellow",
        name="AfterstateSearchAgent",
    )
    if not agent.load_model(model_path):
        return None
    agent.epsilon = 0.0
    return agent

def load_alphazero_agent(model_name=None):
    # Fast mode keeps the same weights but cuts search down so the script is
    # practical to run while iterating.
    explicit_path = _explicit_model_path(model_name)
    if explicit_path:
        model_path = explicit_path
    else:
        model_path = first_existing(
            ALPHAZERO_BEST_MODEL,
            ALPHAZERO_EXTERNAL_MODEL,
            ALPHAZERO_TRAINED_MODEL,
            ALPHAZERO_FINAL_MODEL,
        )
    if not os.path.exists(model_path):
        return None
    env = ChineseCheckersEnv(num_players=2, player_colors=["yellow", "purple"], max_turns=500)
    env.reset()
    agent = AlphaZeroAgent(state_size=len(env.get_state_for_player("yellow")), name="AlphaZeroAgent")
    if not agent.load_model(model_path):
        return None
    guide_path = first_existing(AFTERSTATE_BEST_MODEL, AFTERSTATE_TRAINED_MODEL, AFTERSTATE_FINAL_MODEL)
    if os.path.exists(guide_path):
        agent.enable_afterstate_guide(guide_path)
    if not FULL_ALPHAZERO_EVAL:
        agent.num_simulations = min(agent.num_simulations, ALPHAZERO_FAST_SIMS)
    return agent

def get_action(agent, env, valid_actions):
    # All agent families expose a slightly different interface, so evaluation
    # normalizes them here.
    if isinstance(agent, AfterstateValueAgent):
        return agent.choose_action(env, valid_actions)
    if isinstance(agent, AfterstateSearchAgent):
        return agent.choose_action(env, valid_actions)
    if isinstance(agent, AlphaZeroAgent):
        return agent.choose_action(env, valid_actions, deterministic=True)
    return agent.choose_action(env, valid_actions)

def run_match(first_agent, second_agent, max_turns=200, player_colors=None):
    # A match always names the colours explicitly so lane rotation tests are fair.
    player_colors = list(player_colors or ["yellow", "purple"])
    env = ChineseCheckersEnv(num_players=2, player_colors=player_colors, max_turns=max_turns)
    env.reset()

    done = False
    info = {}
    move_count = 0

    while not done:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            info = {"message": "No valid actions"}
            break

        current_player = env.get_current_player()
        agent = first_agent if current_player == player_colors[0] else second_agent
        action = get_action(agent, env, valid_actions)
        if action is None:
            info = {"message": "No action selected"}
            break

        _, _, done, info = env.step(action)
        move_count += 1

    return {
        "result":           info.get("message", "Unknown"),
        "player_colors":    tuple(player_colors),
        "moves":            move_count,
        "first_goal":       env.count_player_pins_in_target(player_colors[0]),
        "second_goal":      env.count_player_pins_in_target(player_colors[1]),
        "first_distance":   env.total_distance_to_target(player_colors[0]),
        "second_distance":  env.total_distance_to_target(player_colors[1]),
    }

def evaluate(label, first_agent, second_agent, games=20, max_turns=200):
    # We rotate through the supported colour pairs instead of evaluating only
    # on yellow/purple, so these numbers catch lane-specific weaknesses too.
    results = [
        run_match(first_agent, second_agent, max_turns, player_colors=color_pair_for_game(game))
        for game in range(games)
    ]
    first_wins = sum(1 for r in results if r["result"] == f"{r['player_colors'][0]} wins")
    second_wins = sum(1 for r in results if r["result"] == f"{r['player_colors'][1]} wins")
    draws = games - first_wins - second_wins

    print(f"\n{label}")
    print(f"  games={games}  first={first_wins}  second={second_wins}  draws={draws}")
    print(f"  avg moves:              {sum(r['moves'] for r in results) / games:.1f}")
    print(f"  avg first pins/goal:    {sum(r['first_goal'] for r in results) / games:.2f}")
    print(f"  avg second pins/goal:   {sum(r['second_goal'] for r in results) / games:.2f}")
    print(f"  avg first distance:     {sum(r['first_distance'] for r in results) / games:.2f}")
    print(f"  avg second distance:    {sum(r['second_distance'] for r in results) / games:.2f}")

if __name__ == "__main__":
    # The baseline checks always run so you can tell whether the environment or
    # move generator broke even when no trained model is present.
    evaluate("Greedy vs Random",  GreedyAgent("Yellow"), RandomAgent("Purple"), games=STANDARD_GAMES)
    evaluate("Random vs Random",  RandomAgent("Yellow"), RandomAgent("Purple"), games=STANDARD_GAMES)

    afterstate = load_afterstate_agent(AFTERSTATE_MODEL_NAME)
    if afterstate:
        evaluate("Afterstate vs EasyRandom (500t)", afterstate, HomeFirstRandomAgent("Purple"), games=STANDARD_GAMES, max_turns=500)
        evaluate("Afterstate vs Random (500t)",     afterstate, RandomAgent("Purple"), games=STANDARD_GAMES, max_turns=500)
        evaluate(f"Afterstate vs Greedy (300t, {GREEDY_GAMES}g)", afterstate, GreedyAgent("Purple"), games=GREEDY_GAMES, max_turns=300)
        evaluate(f"Afterstate vs Greedy (500t, {GREEDY_GAMES}g)", afterstate, GreedyAgent("Purple"), games=GREEDY_GAMES, max_turns=500)
    else:
        print("\nNo trained afterstate model found, skipping afterstate evaluations.")

    afterstate_search = load_afterstate_search_agent(AFTERSTATE_MODEL_NAME)
    if afterstate_search:
        evaluate("Afterstate+Search vs EasyRandom (500t)", afterstate_search, HomeFirstRandomAgent("Purple"), games=STANDARD_GAMES, max_turns=500)
        evaluate("Afterstate+Search vs Random (500t)",     afterstate_search, RandomAgent("Purple"), games=STANDARD_GAMES, max_turns=500)
        evaluate(f"Afterstate+Search vs Greedy (300t, {GREEDY_GAMES}g)", afterstate_search, GreedyAgent("Purple"), games=GREEDY_GAMES, max_turns=300)
        evaluate(f"Afterstate+Search vs Greedy (500t, {GREEDY_GAMES}g)", afterstate_search, GreedyAgent("Purple"), games=GREEDY_GAMES, max_turns=500)
    else:
        print("\nNo trained afterstate-search agent found, skipping afterstate-search evaluations.")

    if not RUN_ALPHAZERO_EVAL:
        print("\nAlphaZero eval skipped by default because MCTS is slow. Use --alphazero for a fast check.")
    else:
        alphazero = load_alphazero_agent()
        alpha_games = STANDARD_GAMES if FULL_ALPHAZERO_EVAL else min(STANDARD_GAMES, ALPHAZERO_FAST_GAMES)
        alpha_greedy_games = GREEDY_GAMES if FULL_ALPHAZERO_EVAL else min(GREEDY_GAMES, ALPHAZERO_FAST_GAMES)
        alpha_max_turns = 500 if FULL_ALPHAZERO_EVAL else 300
        alpha_label = "AlphaZero" if FULL_ALPHAZERO_EVAL else "AlphaZero fast"

        if alphazero:
            evaluate(f"{alpha_label} vs EasyRandom ({alpha_max_turns}t)", alphazero, HomeFirstRandomAgent("Purple"), games=alpha_games, max_turns=alpha_max_turns)
            evaluate(f"{alpha_label} vs Random ({alpha_max_turns}t)",     alphazero, RandomAgent("Purple"), games=alpha_games, max_turns=alpha_max_turns)
            evaluate(f"{alpha_label} vs Greedy ({alpha_max_turns}t, {alpha_greedy_games}g)", alphazero, GreedyAgent("Purple"), games=alpha_greedy_games, max_turns=alpha_max_turns)
        else:
            print("\nNo trained AlphaZero model found, skipping AlphaZero evaluations.")
