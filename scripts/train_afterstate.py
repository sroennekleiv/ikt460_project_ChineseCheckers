# Train the Afterstate value agent from scratch.

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.afterstate import AfterstateValueAgent
from src.agents import GreedyAgent, HomeFirstRandomAgent
from src.env import ChineseCheckersEnv
from src.paths import (
    AFTERSTATE_BEST_MODEL,
    AFTERSTATE_CHECKPOINT_DIR,
    AFTERSTATE_FINAL_MODEL,
    AFTERSTATE_LEARNING_CURVE,
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

PHASE1_EPISODES = 300
PHASE2_EPISODES = 900
TOTAL_EPISODES = PHASE1_EPISODES + PHASE2_EPISODES

WARMUP_EPISODES = 200
MAX_TURNS = 400
EVAL_FREQ = 100
EVAL_GAMES = 10
CHECKPOINT_FREQ = 500
TARGET_UPDATE_STEPS  = 250
# Replay is relatively expensive on CPU, so we batch several moves between
# updates instead of learning after every single action.
REPLAY_FREQUENCY     = 4

BEST_MODEL = str(AFTERSTATE_BEST_MODEL)
FINAL_MODEL = str(AFTERSTATE_FINAL_MODEL)
COMPAT_MODEL = str(AFTERSTATE_TRAINED_MODEL)
CURVE_PNG = str(AFTERSTATE_LEARNING_CURVE)

shaped_reward = afterstate_shaped_reward
evaluate = evaluate_afterstate

def greedy_warmup(agent, episodes=WARMUP_EPISODES):
    # Warm-start uses a stronger teacher before full online learning begins.
    print(f"\nGreedy warm-start   {episodes} games")
    teacher = GreedyAgent(name="WarmupTeacher")
    opponent = HomeFirstRandomAgent(name="WarmupEasyOpp")
    wins = 0

    for episode in range(episodes):
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

            action = teacher.choose_action(env, valid_actions)
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

        if (episode + 1) % 20 == 0:
            print(f"  {episode + 1:3d}/{episodes}   demo {len(agent.demo_memory)}   wins {wins}/{episode + 1}")

    pretrain_steps = min(250, len(agent.demo_memory) // max(1, agent.batch_size))
    for _ in range(pretrain_steps):
        agent.replay()
    agent.update_target_network()
    print(f"  Done   {len(agent.demo_memory)} experiences   {wins}/{episodes} wins   {pretrain_steps} pretrain steps\n")

def save_curve(history):
    if not PLOT_AVAILABLE or not history["eval_ep"]:
        return

    # The report only needs a few clear curves, so this plot stays compact.
    figure, axes = plt.subplots(2, 1, figsize=(10, 7))
    figure.suptitle("Afterstate Value Training — Chinese Checkers", fontsize=13)

    axes[0].plot(history["eval_ep"], history["eval_wins_easy"], "c-o", ms=3, label="Win % vs EasyRandom")
    axes[0].plot(history["eval_ep"], history["eval_wins_greedy"], "g-o", ms=3, label="Win % vs Greedy")
    axes[0].axhline(50, color="orange", ls="--", alpha=0.4)
    axes[0].axvline(PHASE1_EPISODES, color="gray", ls=":", alpha=0.5, label="Phase 2 start")
    axes[0].set_ylabel("Win rate (%)")
    axes[0].set_ylim(-5, 105)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["eval_ep"], history["eval_pins_easy"], "c-o", ms=3, label="Pins vs EasyRandom")
    axes[1].plot(history["eval_ep"], history["eval_pins_greedy"], "b-o", ms=3, label="Pins vs Greedy")
    axes[1].axhline(10, color="green", ls="--", alpha=0.4)
    axes[1].axvline(PHASE1_EPISODES, color="gray", ls=":", alpha=0.5)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Avg pins / 10")
    axes[1].set_ylim(-0.5, 11)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(CURVE_PNG, dpi=100)
    plt.close(figure)
    print(f"       curve -> {CURVE_PNG}")

def main():
    requested = int(sys.argv[1]) if len(sys.argv) > 1 else None
    phase1 = min(PHASE1_EPISODES, requested) if requested else PHASE1_EPISODES
    phase2 = max(0, (requested or TOTAL_EPISODES) - phase1)
    episodes = phase1 + phase2

    probe_env = ChineseCheckersEnv(num_players=2, player_colors=["yellow", "purple"])
    state_size = len(probe_env.reset())

    agent = AfterstateValueAgent(state_size=state_size, player_color="yellow", name="AfterstateAgent")
    agent.demo_fraction = 0.50

    easy_opponent = HomeFirstRandomAgent(name="EasyOpp")
    greedy_opponent = GreedyAgent(name="GreedyOpp")

    print("\nChinese Checkers   Afterstate Training")
    print()
    greedy_warmup(agent, episodes=WARMUP_EPISODES)
    print("Training Setup")
    print()
    print(f"  Episodes   {episodes}   ({phase1} vs EasyRandom   {phase2} vs Greedy)")
    print(f"  State      {state_size} features")
    print(f"  Color lanes {', '.join(f'{a}/{b}' for a, b in PLAYABLE_COLOR_PAIRS)}")
    print(f"  Epsilon    {agent.epsilon:.2f} -> {agent.epsilon_min:.2f}   decay {agent.epsilon_decay}")
    print("  Reward     win +100   timeout/draw -30   loss -50   + clipped progress + PBRS")
    print(f"  Replay     online {agent.memory.maxlen}   demo {agent.demo_memory.maxlen}   demo frac {agent.demo_fraction:.2f}")
    print(f"  Target     sync every {TARGET_UPDATE_STEPS} replay updates")
    print(f"  Eval       every {EVAL_FREQ} episodes with epsilon=0")
    print()
    print("Training")
    print()

    history = {
        "eval_ep": [],
        "eval_wins_easy": [],
        "eval_pins_easy": [],
        "eval_wins_greedy": [],
        "eval_pins_greedy": [],
    }

    best_score = float("-inf")
    best_episode = 0
    no_improve = 0
    phase1_best_score = float("-inf")
    phase2_best_score = float("-inf")
    start_time = time.time()

    recent_wins = []
    recent_pins = []
    recent_rewards = []
    print_frequency = max(1, min(20, episodes // 25))

    for episode in range(episodes):
        in_phase1 = episode < phase1
        opponent = easy_opponent if in_phase1 else greedy_opponent
        opp_tag = "EasyRandom" if in_phase1 else "Greedy"
        player_colors = color_pair_for_game(episode)
        agent_color = player_colors[0]
        env = ChineseCheckersEnv(num_players=2, player_colors=player_colors, max_turns=MAX_TURNS)

        if episode == phase1 and phase2 > 0:
            # Phase 2 is harder, so we lift exploration a little. We do not
            # reset it all the way up, because that would throw away the good
            # policy the warmup just taught the network.
            agent.epsilon = max(agent.epsilon, 0.10)
            no_improve = 0
            best_score = float("-inf")
            print(f"\n--- Phase 2 start (ep {episode})  ε set to {agent.epsilon:.2f} ---\n")

        env.reset()
        done = False
        info = {}
        steps = 0
        total_reward = 0.0
        won = False

        while not done and steps < MAX_TURNS:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            if env.get_current_player() == agent_color:
                # The agent move is the one we store as an afterstate example.
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
                agent.remember(afterstate, reward, next_position_state, done)
                total_reward += reward

                if steps % REPLAY_FREQUENCY == 0:
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
        recent_pins.append(env.count_player_pins_in_target(agent_color))
        recent_rewards.append(total_reward)

        if (episode + 1) % CHECKPOINT_FREQ == 0:
            agent.save_model(str(AFTERSTATE_CHECKPOINT_DIR / f"phase1_ep{episode + 1}.pth"))

        if (episode + 1) % print_frequency == 0:
            sample_count = min(len(recent_wins), print_frequency)
            window_wins = recent_wins[-sample_count:]
            window_pins = recent_pins[-sample_count:]
            window_rewards = recent_rewards[-sample_count:]
            win_rate = sum(window_wins) / sample_count * 100.0
            avg_pins = sum(window_pins) / sample_count
            avg_reward = sum(window_rewards) / sample_count
            print(
                f"  ep {episode + 1:4d}/{episodes}  [{opp_tag}]   "
                f"e={agent.epsilon:.3f}   steps={steps:3d}   wins={win_rate:4.1f}%   "
                f"avgR={avg_reward:+.2f}   pins={avg_pins:.1f}/10   t={time.time() - start_time:.0f}s"
            )

        if (episode + 1) % EVAL_FREQ == 0:
            easy_wins, easy_pins = evaluate(agent, easy_opponent, games=EVAL_GAMES, max_turns=MAX_TURNS)
            greedy_wins, greedy_pins = evaluate(agent, greedy_opponent, games=EVAL_GAMES, max_turns=MAX_TURNS)
            print(f"       eval EasyRandom   wins={easy_wins:.1f}%   pins={easy_pins:.1f}/10")
            print(f"       eval Greedy      wins={greedy_wins:.1f}%   pins={greedy_pins:.1f}/10")

            history["eval_ep"].append(episode + 1)
            history["eval_wins_easy"].append(easy_wins)
            history["eval_pins_easy"].append(easy_pins)
            history["eval_wins_greedy"].append(greedy_wins)
            history["eval_pins_greedy"].append(greedy_pins)
            save_curve(history)

            easy_score = model_selection_score(easy_wins, easy_pins)
            greedy_score = model_selection_score(greedy_wins, greedy_pins)

            if in_phase1:
                eval_score = easy_score + 0.10 * greedy_score
            else:
                eval_score = greedy_score + 0.10 * easy_score

            if in_phase1:
                phase1_best_score = max(phase1_best_score, eval_score)
            else:
                phase2_best_score = max(phase2_best_score, eval_score)

            if eval_score > best_score:
                best_score = eval_score
                best_episode = episode + 1
                no_improve = 0
                agent.save_model(BEST_MODEL)
                agent.save_model(COMPAT_MODEL)
                print("       *** New best model saved ***")
            else:
                no_improve += 1
                if (not in_phase1) and no_improve >= 8:
                    print(
                        f"       early stop   no eval improvement for {no_improve} checkpoints   "
                        f"(best at ep {best_episode})"
                    )
                    break

    elapsed = time.time() - start_time
    print()
    print("Training Complete")
    print()
    print(f"  Time      {elapsed:.0f}s")
    print(f"  Epsilon   {agent.epsilon:.4f}")
    print(f"  Wins      {sum(recent_wins[-200:]) / max(1, len(recent_wins[-200:])) * 100:.1f}%   (last 200 eps)")
    print(f"  Pins      {sum(recent_pins[-200:]) / max(1, len(recent_pins[-200:])):.1f}/10")
    print(f"  Memory    online {len(agent.memory)}   demo {len(agent.demo_memory)}")
    if best_episode:
        print(f"  Best eval at ep {best_episode}")

    agent.save_model(FINAL_MODEL)
    if best_episode:
        agent.load_model(BEST_MODEL)
        agent.save_model(COMPAT_MODEL)
        print()
        print(f"  Best model saved to {COMPAT_MODEL}")

    print()
    print(f"Test vs Greedy   {EVAL_GAMES} games")
    print()
    final_wins, final_pins = evaluate(agent, greedy_opponent, games=EVAL_GAMES, max_turns=MAX_TURNS)
    print(f"  Result   {final_wins:.0f}% wins   avg pins {final_pins:.1f}/10")

if __name__ == "__main__":
    main()
