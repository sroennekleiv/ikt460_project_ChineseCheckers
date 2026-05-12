# Reward and evaluation helpers for the learned agents.

from src.env import ChineseCheckersEnv
from src.perspective import color_pair_for_game

def compute_potential(env, player_color):
    # This packs "pins already home" and "distance still left" into one number
    # so shaped rewards can compare before and after a move.
    pins = env.count_player_pins_in_target(player_color)
    dist = env.total_distance_to_target(player_color)
    return (pins * 10.0 - dist) / 14.0

def model_selection_score(win_rate_percent, avg_pins):
    # The report-friendly score gives most of the weight to winning, while
    # still rewarding agents that consistently finish more pins.
    return float(win_rate_percent) + float(avg_pins) * 10.0

def afterstate_shaped_reward(done, info, current_player, phi_s, phi_sp, action_progress):
    # Big outcomes still dominate, but we keep a small progress bonus so the
    # agent gets useful feedback long before the game finishes.
    clipped_progress = max(-1.0, min(1.0, action_progress / 5.0))
    progress = 0.25 * clipped_progress

    if done:
        message = info.get("message", "").lower()
        if f"{current_player} wins" in message:
            return 100.0 + progress
        if any(token in message for token in ("max turns", "repetition", "draw")):
            return -30.0 + progress
        return -50.0 + progress

    pbrs = 0.05 * (0.99 * phi_sp - phi_s)
    return progress + pbrs

def evaluate_afterstate(agent, opponent, games=10, max_turns=400):
    # Evaluation always runs with epsilon off so we measure the learned policy,
    # not whatever random exploration happened to fire.
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    wins, pins = 0, []

    for game in range(games):
        player_colors = color_pair_for_game(game)
        agent_color = player_colors[0]
        env = ChineseCheckersEnv(num_players=2, player_colors=player_colors, max_turns=max_turns)
        env.reset()
        done = False
        info = {}

        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            if env.get_current_player() == agent_color:
                action = agent.choose_action(env, valid_actions)
            else:
                action = opponent.choose_action(env, valid_actions)
            if action is None:
                break
            _, _, done, info = env.step(action)

        if f"{agent_color} wins" in info.get("message", ""):
            wins += 1
        pins.append(env.count_player_pins_in_target(agent_color))

    agent.epsilon = old_epsilon
    return wins / games * 100.0, sum(pins) / games
