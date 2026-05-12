# Chinese Checkers RL Project

This repo now keeps the active learning code focused on two agents:

- `Afterstate`
- `AlphaZero`

Old DQN and PPO training code has been removed from the active project layout.

## Main Commands

- `python3 player.py`: tournament/server player.
- `python3 main.py`: local GUI.
- `python3 scripts/evaluate.py 6`: evaluate Afterstate and baseline opponents.
- `python3 scripts/evaluate.py 6 --alphazero`: include a fast AlphaZero check.

## Training Commands

- `python3 scripts/train_afterstate.py`
- `python3 scripts/continue_afterstate.py`
- `python3 scripts/train_alphazero.py`
- `python3 scripts/continue_alphazero.py`

## Structure

- `src/board.py`: board geometry and pins.
- `src/game.py`: rules, move handling, and turn management.
- `src/env.py`: reinforcement-learning environment.
- `src/gui.py`: local GUI.
- `src/agents.py`: random, greedy, and minimax opponents.
- `src/afterstate.py`: Afterstate agent and search logic.
- `src/alphazero.py`: AlphaZero agent and MCTS logic.
- `src/network.py`: PyTorch network definitions.
- `src/paths.py`: project output paths and folder helpers.
- `src/perspective.py`: rotates every color lane into the shared training view.
- `src/rewards.py`: reward shaping and evaluation scoring helpers.
- `src/training.py`: shared AlphaZero training helpers.
- `scripts/`: runnable training and evaluation commands.
- `server/`: tournament server support files. This folder stays untouched.
- `outputs/`: generated models, plots, and logs.
