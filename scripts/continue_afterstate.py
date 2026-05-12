# Continue training from an existing Afterstate checkpoint.

import os
import random
import shutil
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.afterstate import AfterstateValueAgent
from src.agents import GreedyAgent, HomeFirstRandomAgent, MinimaxAgent
from src.env import ChineseCheckersEnv
from src.paths import (
    AFTERSTATE_BACKUP_DIR,
    AFTERSTATE_BEST_MODEL,
    AFTERSTATE_CHECKPOINT_DIR,
    AFTERSTATE_PHASE2_LEARNING_CURVE,
    AFTERSTATE_TRAINED_MODEL,
    MPL_CACHE_DIR,
    ensure_project_dirs,
)
from src.perspective import PLAYABLE_COLOR_PAIRS, color_pair_for_game
from src.rewards import afterstate_shaped_reward, compute_potential, evaluate_afterstate, model_selection_score

ensure_project_dirs()
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

CONTINUE_EPISODES = 300
MAX_TURNS = 400
EVAL_FREQ = 50
EVAL_GAMES = 10
CHECKPOINT_FREQ = 250
TARGET_UPDATE_STEPS = 250
# Depth 2 is the practical sweet spot here: strong enough to teach something,
# but still fast enough for long continuation runs on CPU.
MINIMAX_DEPTH = 2
TRAINING_OPPONENT_MIX = (
    ("greedy",  0.60),
    ("minimax", 0.40),
)

CHALLENGER_BEST_MODEL = str(AFTERSTATE_BACKUP_DIR / "challenger_mixed_minimax_best.pth")
CHALLENGER_TRAINED_MODEL = str(AFTERSTATE_BACKUP_DIR / "challenger_mixed_minimax_trained.pth")
CURVE_PNG = str(AFTERSTATE_PHASE2_LEARNING_CURVE)

shaped_reward = afterstate_shaped_reward
evaluate = evaluate_afterstate

def choose_training_opponent(opponents):
    # The continuation run samples opponents from a fixed training mix.
    labels = [name for name, _ in TRAINING_OPPONENT_MIX]
    weights = [weight for _, weight in TRAINING_OPPONENT_MIX]
    label = random.choices(labels, weights=weights, k=1)[0]
    return label, opponents[label]

def choose_teacher_action(env, valid_actions, player_color="yellow"):
    if not valid_actions:
        return None

    # Demo collection uses a deterministic teacher so copied examples stay stable.
    best_action = None
    best_score = None
    best_index = None

    for action in valid_actions:
        score = env.evaluate_action_progress(action, player_color)
        action_index = int(action[0]) * 121 + int(action[1])
        if best_score is None or score > best_score or (score == best_score and action_index < best_index):
            best_action = action
            best_score = score
            best_index = action_index

    return best_action

def collect_demo_experiences(agent, opponents, episodes=120):
    # Fresh demo data helps continuation keep up with the tougher opponent mix.
    print(f"\nCollecting mixed demo experiences   {episodes} games")
    wins = 0

    for episode in range(episodes):
        _, opponent = choose_training_opponent(opponents)
        player_colors = color_pair_for_game(episode)
        agent_color = player_colors[0]
        env = ChineseCheckersEnv(num_players=2, player_colors=player_colors, max_turns=MAX_TURNS)
        env.reset()
        done = False
        info = {}
        steps = 0

        while not done and steps < MAX_TURNS:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            if env.get_current_player() != agent_color:
                action = opponent.choose_action(env, valid_actions)
                if action is None:
                    break
                _, _, done, info = env.step(action)
                steps += 1
                continue

            action = choose_teacher_action(env, valid_actions, agent_color)
            if action is None:
                break

            phi_s = compute_potential(env, agent_color)
            progress = env.evaluate_action_progress(action, agent_color)
            afterstate = agent.afterstate_from_env_action(env, action, agent_color)
            _, _, done, info = env.step(action)
            steps += 1

            if not done:
                opponent_actions = env.get_valid_actions()
                if opponent_actions:
                    opponent_action = opponent.choose_action(env, opponent_actions)
                    if opponent_action is not None:
                        _, _, done, info = env.step(opponent_action)
                        steps += 1

            phi_sp = compute_potential(env, agent_color)
            reward = shaped_reward(done, info, agent_color, phi_s, phi_sp, progress)
            next_position_state = agent.position_state_from_env(env, agent_color) if not done else None
            agent.remember(afterstate, reward, next_position_state, done, demo=True)

        if f"{agent_color} wins" in info.get("message", ""):
            wins += 1

        if (episode + 1) % 30 == 0:
            print(f"  {episode + 1}/{episodes}   demo {len(agent.demo_memory)}   wins {wins}/{episode + 1}")

    print(f"  Done   {len(agent.demo_memory)} demo experiences   wins {wins}/{episodes}")

def save_curve(history):
    if not PLOT_AVAILABLE or not history["eval_ep"]:
        return

    # This plot is mainly for spotting collapse during continuation runs.
    figure, axes = plt.subplots(3, 1, figsize=(10, 10))
    figure.suptitle(f"Afterstate Continuation — vs Greedy + Minimax(d={MINIMAX_DEPTH})", fontsize=13)

    axes[0].plot(history["eval_ep"], history["eval_pins_greedy"], "b-o", ms=4, label="Pins vs Greedy")
    axes[0].plot(history["eval_ep"], history["eval_pins_easy"], "c-o", ms=4, label="Pins vs EasyRandom")
    axes[0].axhline(10, color="green", ls="--", alpha=0.4)
    axes[0].set_ylabel("Pins / 10")
    axes[0].set_ylim(-0.5, 11)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["eval_ep"], history["eval_wins_greedy"], "g-o", ms=4, label="Wins vs Greedy")
    axes[1].plot(history["eval_ep"], history["eval_wins_easy"], "m-o", ms=4, label="Wins vs EasyRandom")
    axes[1].set_ylabel("Win rate (%)")
    axes[1].set_ylim(-5, 105)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(history["train_ep"], history["avg_reward"], "r-", alpha=0.7, label="Avg reward")
    axes[2].axhline(0, color="black", ls="--", alpha=0.3)
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Avg reward")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(CURVE_PNG, dpi=100)
    plt.close(figure)
    print(f"       curve -> {CURVE_PNG}")

def main():
    print(f"\nAfterstate Continuation   (vs Greedy + Minimax depth {MINIMAX_DEPTH})\n")

    env_probe = ChineseCheckersEnv(num_players=2, player_colors=["yellow", "purple"])
    state_size = len(env_probe.reset())

    agent = AfterstateValueAgent(state_size=state_size, player_color="yellow", name="AfterstateAgent")
    agent.demo_fraction = 0.60

    loaded = False
    for candidate in (
        str(AFTERSTATE_TRAINED_MODEL),
        str(AFTERSTATE_BEST_MODEL),
    ):
        if os.path.exists(candidate) and agent.load_model(candidate):
            loaded = True
            break
    if not loaded:
        print(f"ERROR: could not load an afterstate checkpoint from {AFTERSTATE_TRAINED_MODEL} or {AFTERSTATE_BEST_MODEL}")
        return

    # keep the good replay memory and turn the optimizer down. The whole
    # point of continuation is to preserve what already works while adapting
    # to a stronger opponent mix.
    agent.epsilon = 0.05
    agent.epsilon_decay = 0.9999
    agent.epsilon_min = 0.02
    agent.learning_rate = 0.000010
    agent.optimizer = agent.optimizer.__class__(agent.value_network.parameters(), lr=agent.learning_rate)
    agent.train_updates = 0

    print(f"  State size    {state_size}")
    print(f"  Epsilon       {agent.epsilon:.2f} -> {agent.epsilon_min:.2f}   decay {agent.epsilon_decay}")
    print(f"  Learning rate {agent.learning_rate}")
    print(f"  Opponent mix  60% Greedy   40% Minimax(depth={MINIMAX_DEPTH})  (depth 2 for speed)")
    print(f"  Color lanes   {', '.join(f'{a}/{b}' for a, b in PLAYABLE_COLOR_PAIRS)}")
    print(f"  Memory        online {len(agent.memory)} (preserved)   demo {len(agent.demo_memory)}")

    training_opponents = {
        "greedy":  GreedyAgent(name="GreedyOpponent"),
        "minimax": MinimaxAgent(name="MinimaxOpponent", depth=MINIMAX_DEPTH),
    }
    collect_demo_experiences(agent, training_opponents, episodes=120)

    greedy_opp = GreedyAgent(name="GreedyOpponent")
    minimax_opp = MinimaxAgent(name="MinimaxOpponent", depth=MINIMAX_DEPTH)
    easy_opp = HomeFirstRandomAgent(name="EasyOpponent")

    teacher_guidance_episodes = 250
    teacher_start_prob = 0.20
    early_stop_patience = 2
    min_episodes_before_early_stop = 150
    collapse_stop_margin = 40.0

    best_eval_score = float("-inf")
    best_eval_episode = 0
    evals_without_improvement = 0

    history = {
        "train_ep": [],
        "avg_reward": [],
        "eval_ep": [],
        "eval_wins_greedy": [],
        "eval_pins_greedy": [],
        "eval_wins_easy": [],
        "eval_pins_easy": [],
    }

    print(f"\nTraining   {CONTINUE_EPISODES} episodes vs Greedy + Minimax(depth={MINIMAX_DEPTH})\n")
    start_time = time.time()
    recent_wins = []
    recent_rewards = []
    recent_pins = []
    print_freq = 20

    for episode in range(CONTINUE_EPISODES):
        opponent_label, opponent = choose_training_opponent(training_opponents)
        player_colors = color_pair_for_game(episode)
        agent_color = player_colors[0]
        env = ChineseCheckersEnv(num_players=2, player_colors=player_colors, max_turns=MAX_TURNS)

        env.reset()
        done = False
        info = {}
        total_reward = 0.0
        steps = 0
        agent_step = 0
        won = False

        while not done and steps < MAX_TURNS:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            if env.get_current_player() == agent_color:
                # Continuation still learns from the agent move even when a
                # temporary teacher hint is steering the choice.
                teacher_prob = 0.0
                if episode < teacher_guidance_episodes:
                    teacher_prob = teacher_start_prob * (1.0 - episode / teacher_guidance_episodes)

                guided = False
                if teacher_prob > 0.0 and random.random() < teacher_prob:
                    action = choose_teacher_action(env, valid_actions, agent_color)
                    guided = action is not None
                else:
                    action = agent.choose_action(env, valid_actions)

                if action is None:
                    break

                phi_s = compute_potential(env, agent_color)
                progress = env.evaluate_action_progress(action, agent_color)
                afterstate = agent.afterstate_from_env_action(env, action, agent_color)
                _, _, done, info = env.step(action)
                steps += 1

                if not done:
                    opponent_actions = env.get_valid_actions()
                    if opponent_actions:
                        opponent_action = opponent.choose_action(env, opponent_actions)
                        if opponent_action is not None:
                            _, _, done, info = env.step(opponent_action)
                            steps += 1

                phi_sp = compute_potential(env, agent_color)
                reward = shaped_reward(done, info, agent_color, phi_s, phi_sp, progress)
                next_position_state = agent.position_state_from_env(env, agent_color) if not done else None
                agent.remember(afterstate, reward, next_position_state, done, demo=guided)
                total_reward += reward

                agent_step += 1
                if agent_step % 3 == 0:
                    did_replay = agent.replay()
                    if did_replay and agent.train_updates % TARGET_UPDATE_STEPS == 0:
                        agent.update_target_network()

                if done and f"{agent_color} wins" in info.get("message", ""):
                    won = True
            else:
                action = opponent.choose_action(env, valid_actions)
                if action is None:
                    break
                _, _, done, info = env.step(action)
                steps += 1

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        recent_wins.append(1 if won else 0)
        recent_rewards.append(total_reward)
        recent_pins.append(env.count_player_pins_in_target(agent_color))

        if (episode + 1) % CHECKPOINT_FREQ == 0:
            agent.save_model(str(AFTERSTATE_CHECKPOINT_DIR / f"continue_ep{episode + 1}.pth"))

        if (episode + 1) % print_freq == 0:
            sample_count = min(len(recent_wins), print_freq)
            win_rate = sum(recent_wins[-sample_count:]) / sample_count * 100.0
            avg_reward = sum(recent_rewards[-sample_count:]) / sample_count
            avg_pins = sum(recent_pins[-sample_count:]) / sample_count
            elapsed = time.time() - start_time
            print(
                f"  ep {episode + 1:4d}/{CONTINUE_EPISODES}   [{opponent_label.capitalize():7s}]   "
                f"e={agent.epsilon:.3f}   "
                f"steps={steps:3d}   wins={win_rate:4.1f}%   avgR={avg_reward:+.2f}   "
                f"pins={avg_pins:.1f}/10   t={elapsed:.0f}s"
            )
            history["train_ep"].append(episode + 1)
            history["avg_reward"].append(avg_reward)

        if (episode + 1) % EVAL_FREQ == 0:
            wr_g, pins_g = evaluate(agent, greedy_opp, games=EVAL_GAMES, max_turns=MAX_TURNS)
            wr_e, pins_e = evaluate(agent, easy_opp, games=EVAL_GAMES, max_turns=500)
            print(f"       eval Greedy      wins={wr_g:4.1f}%   pins={pins_g:.1f}/10")
            print(f"       eval EasyRandom  wins={wr_e:4.1f}%   pins={pins_e:.1f}/10")

            greedy_score = model_selection_score(wr_g, pins_g)
            easy_score = model_selection_score(wr_e, pins_e)
            eval_score = greedy_score + 0.15 * easy_score

            history["eval_ep"].append(episode + 1)
            history["eval_wins_greedy"].append(wr_g)
            history["eval_pins_greedy"].append(pins_g)
            history["eval_wins_easy"].append(wr_e)
            history["eval_pins_easy"].append(pins_e)
            save_curve(history)

            if eval_score > best_eval_score:
                best_eval_score = eval_score
                best_eval_episode = episode + 1
                evals_without_improvement = 0
                agent.save_model(CHALLENGER_BEST_MODEL)
                agent.save_model(CHALLENGER_TRAINED_MODEL)
                print("       *** New best model saved ***")
            else:
                evals_without_improvement += 1
                collapsed = best_eval_score > float("-inf") and eval_score <= best_eval_score - collapse_stop_margin
                if ((episode + 1) >= min_episodes_before_early_stop and
                        (evals_without_improvement >= early_stop_patience or collapsed)):
                    reason = "collapsed below best" if collapsed else "no eval improvement"
                    print(
                        f"       early stop   {reason} for {evals_without_improvement} checkpoints"
                        f"   (best at ep {best_eval_episode})"
                    )
                    break

    print("\n\nAfterstate Continuation Complete\n")
    print(f"  Time      {time.time() - start_time:.0f}s")
    print(f"  Epsilon   {agent.epsilon:.4f}")
    print(f"  Memory    online {len(agent.memory)}   demo {len(agent.demo_memory)}")
    if best_eval_episode:
        print(f"  Best eval at ep {best_eval_episode}")
    print(f"  Challenger best   {CHALLENGER_BEST_MODEL}")

    print(f"\nFinal test vs Greedy   {EVAL_GAMES} games\n")
    candidate_path = CHALLENGER_BEST_MODEL if os.path.exists(CHALLENGER_BEST_MODEL) else str(AFTERSTATE_TRAINED_MODEL)
    agent.load_model(candidate_path)
    agent.epsilon = 0.0
    final_wins, final_pins = evaluate(agent, greedy_opp, games=EVAL_GAMES, max_turns=MAX_TURNS)
    print(f"  Result   {final_wins:.1f}% wins   pins={final_pins:.1f}/10")
    print(f"\nFinal test vs Minimax(depth={MINIMAX_DEPTH})   {EVAL_GAMES} games\n")
    minimax_wins, minimax_pins = evaluate(agent, minimax_opp, games=EVAL_GAMES, max_turns=MAX_TURNS)
    print(f"  Result   {minimax_wins:.1f}% wins   pins={minimax_pins:.1f}/10")

    should_promote = final_wins >= 70.0 and final_pins >= 9.0 and minimax_wins >= 50.0
    if should_promote:
        shutil.copy2(candidate_path, AFTERSTATE_BEST_MODEL)
        shutil.copy2(candidate_path, AFTERSTATE_TRAINED_MODEL)
        print(f"\nPromoted challenger to tournament afterstate model: {AFTERSTATE_BEST_MODEL}")
    else:
        print("\nKept existing tournament afterstate model; challenger did not pass promotion gate.")

if __name__ == "__main__":
    main()
