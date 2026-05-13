# AlphaZero-style search agent for two-player Chinese Checkers.

import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.board import HEX_DIRECTIONS
from src.network import PolicyValueNetwork
from src.perspective import (
    REFERENCE_COLOR,
    index_to_reference_perspective,
    indices_to_reference_perspective,
)

SEARCH_DIRS = HEX_DIRECTIONS

_AZ_REFERENCE_COLOR = REFERENCE_COLOR

def _colour_list(board, player_order=None):
    if player_order is not None:
        return [str(colour) for colour in player_order]

    # Without an explicit order we still build one stable colour list so state
    # packing stays deterministic.
    colours = set()
    for colour, opposite in board.colour_opposites.items():
        colours.add(str(colour))
        colours.add(str(opposite))
    return sorted(colours)

def _positions_from_pins(pins_on_board, colours):
    # Positions are stored by pin id so the policy head has a stable move index.
    positions = {colour: [None] * 10 for colour in colours}

    for pin in pins_on_board:
        colour = str(pin.color)
        if colour not in positions:
            positions[colour] = [None] * 10
        pin_id = int(pin.id)
        if 0 <= pin_id < len(positions[colour]):
            positions[colour][pin_id] = int(pin.axialindex)

    for colour in positions:
        fallback = []
        for pin in pins_on_board:
            if str(pin.color) == colour:
                fallback.append((int(pin.id), int(pin.axialindex)))
        if fallback:
            fallback.sort()
            for pin_id, axialindex in fallback:
                if positions[colour][pin_id] is None:
                    positions[colour][pin_id] = axialindex
        positions[colour] = tuple(
            int(axialindex) if axialindex is not None else 0
            for axialindex in positions[colour]
        )

    return positions

def _axial_distance(board, idx1, idx2):
    cell_1 = board.cells[int(idx1)]
    cell_2 = board.cells[int(idx2)]
    q_diff = cell_1.q - cell_2.q
    r_diff = cell_1.r - cell_2.r
    s_diff = (-cell_1.q - cell_1.r) - (-cell_2.q - cell_2.r)
    return max(abs(q_diff), abs(r_diff), abs(s_diff))

def _count_pins_in_target(positions, target_cells):
    return sum(1 for position in positions if position in target_cells)

def _total_distance_to_target(board, positions, target_cells):
    if not positions or not target_cells:
        return 0

    in_target = {position for position in positions if position in target_cells}
    outside = [position for position in positions if position not in target_cells]

    if not outside:
        return 0

    available = [cell for cell in target_cells if cell not in in_target]
    if not available:
        return 0

    # Matching the most constrained pins first gives a more realistic endgame
    # distance than letting every pin claim the same nearest target cell.
    outside_sorted = sorted(
        outside,
        key=lambda position: min(_axial_distance(board, position, target) for target in available),
        reverse=True,
    )

    total = 0
    remaining = list(available)
    for position in outside_sorted:
        if not remaining:
            break
        best_target = min(remaining, key=lambda target: _axial_distance(board, position, target))
        total += _axial_distance(board, position, best_target)
        remaining.remove(best_target)

    return total

def _repetition_positions_key(positions_by_colour):
    # Repetition should care about the occupied cells, not which same-colour
    # pin id happens to sit on each one.
    return tuple(
        sorted(
            (str(colour), tuple(sorted(int(position) for position in positions)))
            for colour, positions in positions_by_colour.items()
        )
    )

class AlphaZeroBoardState:
    def __init__(
        self,
        board,
        positions,
        current_player,
        player_order=None,
        move_count=0,
        max_turns=400,
        max_repetitions=6,
        state_counts=None,
    ):
        self.board = board
        self.player_order = tuple(_colour_list(board, player_order))
        self.positions = {
            str(colour): tuple(int(position) for position in positions.get(str(colour), ()))
            for colour in self.player_order
        }
        self.current_player = str(current_player)
        self.move_count = int(move_count)
        self.max_turns = int(max_turns)
        self.max_repetitions = int(max_repetitions)
        self.state_counts = dict(state_counts or {})

    @classmethod
    def from_env(cls, env):
        # MCTS works on immutable board states, so we snapshot the environment
        # before search starts.
        colours = [str(colour) for colour in env.player_colors]
        positions = _positions_from_pins(env.pins_on_board, colours)
        return cls(
            board=env.board,
            positions=positions,
            current_player=env.get_current_player(),
            player_order=colours,
            move_count=env.turn_count,
            max_turns=env.max_turns,
            max_repetitions=env.max_repetitions,
            state_counts=env.state_counts,
        )

    @classmethod
    def from_board(
        cls,
        pins_on_board,
        current_player,
        board,
        player_order=None,
        move_count=0,
        max_turns=400,
        max_repetitions=6,
        state_counts=None,
    ):
        colours = _colour_list(board, player_order or [current_player, board.colour_opposites.get(current_player, "")])
        positions = _positions_from_pins(pins_on_board, colours)
        return cls(
            board=board,
            positions=positions,
            current_player=current_player,
            player_order=colours,
            move_count=move_count,
            max_turns=max_turns,
            max_repetitions=max_repetitions,
            state_counts=state_counts,
        )

    def opponent_of(self, colour):
        for player in self.player_order:
            if player != str(colour):
                return player
        return str(colour)

    def target_cells(self, colour):
        target_colour = self.board.colour_opposites.get(str(colour), "")
        return set(self.board.axial_of_colour(target_colour)) if target_colour else set()

    def pins_in_goal(self, colour):
        return _count_pins_in_target(self.positions.get(str(colour), ()), self.target_cells(colour))

    def total_distance_to_target(self, colour):
        return _total_distance_to_target(
            self.board,
            self.positions.get(str(colour), ()),
            self.target_cells(colour),
        )

    def race_margin(self, player_color):
        colour = str(player_color)
        opponent = self.opponent_of(colour)
        my_pins = self.pins_in_goal(colour)
        opp_pins = self.pins_in_goal(opponent)
        my_dist = self.total_distance_to_target(colour)
        opp_dist = self.total_distance_to_target(opponent)
        return 18.0 * (my_pins - opp_pins) + 0.35 * (opp_dist - my_dist)

    def heuristic_value(self, player_color):
        margin = self.race_margin(player_color)
        return float(math.tanh(margin / 12.0))

    def state_vector(self, player_color=None):
        colour = str(player_color or self.current_player)
        own_positions = set(self.positions.get(colour, ()))
        opponent_positions = set()

        for other_colour, positions in self.positions.items():
            if other_colour == colour:
                continue
            opponent_positions.update(positions)

        # Earlier versions only mirrored yellow and purple. The current code
        # rotates all supported two-player lanes into the same reference view.
        if colour != _AZ_REFERENCE_COLOR:
            own_positions = set(indices_to_reference_perspective(self.board, colour, own_positions))
            opponent_positions = set(indices_to_reference_perspective(self.board, colour, opponent_positions))
            colour = _AZ_REFERENCE_COLOR

        target_positions = self.target_cells(colour)
        board_size = len(self.board.cells)
        # The network sees the board as three binary layers: mine, theirs, target.
        own_layer = [1 if idx in own_positions else 0 for idx in range(board_size)]
        opp_layer = [1 if idx in opponent_positions else 0 for idx in range(board_size)]
        target_layer = [1 if idx in target_positions else 0 for idx in range(board_size)]
        return own_layer + opp_layer + target_layer

    def reference_destination(self, action, player_color=None):
        _, destination = action
        colour = str(player_color or self.current_player)
        return index_to_reference_perspective(self.board, colour, destination, _AZ_REFERENCE_COLOR)

    def policy_index(self, action, player_color=None):
        pin_id, _ = action
        return int(pin_id) * 121 + self.reference_destination(action, player_color)

    def repetition_key(self, current_player=None):
        # Search should use the same repetition identity as the live environment.
        return str(current_player or self.current_player), _repetition_positions_key(self.positions)

    def valid_actions(self, player_color=None):
        colour = str(player_color or self.current_player)
        positions = self.positions.get(colour, ())
        occupied = set()
        for colour_positions in self.positions.values():
            occupied.update(int(position) for position in colour_positions)

        valid = []
        for pin_id, start_idx in enumerate(positions):
            start_cell = self.board.cells[int(start_idx)]
            possible = set()

            # Search uses the same step-and-hop rule as the real board.
            for dq, dr in SEARCH_DIRS:
                neighbour = self.board.hole_index_of.get((start_cell.q + dq, start_cell.r + dr))
                if neighbour is not None and neighbour not in occupied:
                    possible.add(neighbour)

            visited = {int(start_idx)}
            stack = [int(start_idx)]
            while stack:
                current = stack.pop()
                current_cell = self.board.cells[int(current)]

                for dq, dr in SEARCH_DIRS:
                    adjacent = self.board.hole_index_of.get((current_cell.q + dq, current_cell.r + dr))
                    landing = self.board.hole_index_of.get((current_cell.q + 2 * dq, current_cell.r + 2 * dr))

                    if adjacent is None or landing is None:
                        continue
                    if adjacent not in occupied or landing in occupied or landing in visited:
                        continue

                    possible.add(landing)
                    visited.add(landing)
                    stack.append(landing)

            for destination in possible:
                if start_cell.postype != colour and self.board.cells[int(destination)].postype == colour:
                    continue
                valid.append((pin_id, int(destination)))

        return valid

    def progress_score(self, action, colour):
        # This is a local move prior, not the final decision.
        pin_id, destination = action
        positions = list(self.positions.get(str(colour), ()))
        if not 0 <= int(pin_id) < len(positions):
            return -1.0

        old_position = positions[int(pin_id)]
        old_distance = _total_distance_to_target(self.board, positions, self.target_cells(colour))
        positions[int(pin_id)] = int(destination)
        new_distance = _total_distance_to_target(self.board, positions, self.target_cells(colour))

        score = float(old_distance - new_distance)
        target_positions = self.target_cells(colour)
        old_in_target = int(old_position) in target_positions
        new_in_target = int(destination) in target_positions
        if new_in_target and not old_in_target:
            score += 20.0
        if old_in_target and not new_in_target:
            score -= 80.0
        if old_in_target and new_in_target:
            score -= 1.0
        return score

    def endgame_progress_score(self, action, colour):
        score = self.progress_score(action, colour)
        pin_id, destination = action
        positions = self.positions.get(str(colour), ())
        if not 0 <= int(pin_id) < len(positions):
            return score

        target_positions = self.target_cells(colour)
        old_position = int(positions[int(pin_id)])
        old_in_target = old_position in target_positions
        new_in_target = int(destination) in target_positions
        unfinished_pins = sum(1 for position in positions if int(position) not in target_positions)

        # Endgames get noisy if already-finished pins keep shuffling inside the
        # target triangle, so we bias the scorer toward leaving them settled.
        if unfinished_pins > 1 and old_in_target:
            score -= 12.0
        elif unfinished_pins == 1 and old_in_target and new_in_target:
            score -= 2.0
        if not old_in_target:
            score += 4.0
            if new_in_target:
                score += 12.0
        return score

    def apply_action(self, action):
        # Every action returns a fresh next state so search branches stay isolated.
        pin_id, destination = action
        mover = self.current_player
        next_player = self.opponent_of(mover)
        new_positions = {colour: list(positions) for colour, positions in self.positions.items()}
        new_positions[mover][int(pin_id)] = int(destination)
        next_state = AlphaZeroBoardState(
            board=self.board,
            positions=new_positions,
            current_player=next_player,
            player_order=self.player_order,
            move_count=self.move_count + 1,
            max_turns=self.max_turns,
            max_repetitions=self.max_repetitions,
        )

        if next_state.pins_in_goal(mover) == len(next_state.positions.get(mover, ())):
            return next_state, True, {"winner": mover, "message": f"{mover} wins"}

        if next_state.move_count >= self.max_turns:
            return next_state, True, {"winner": None, "message": "draw"}

        next_counts = dict(self.state_counts)
        repetition_key = next_state.repetition_key()
        next_counts[repetition_key] = next_counts.get(repetition_key, 0) + 1
        next_state.state_counts = next_counts
        if next_state.max_repetitions > 0 and next_counts[repetition_key] >= next_state.max_repetitions:
            return next_state, True, {"winner": None, "message": "draw"}

        return next_state, False, {"winner": None, "message": ""}

    def terminal_value_for(self, player_color, info):
        winner = info.get("winner")
        if winner is None:
            return 0.0
        return 1.0 if str(winner) == str(player_color) else -1.0

class MCTSNode:
    def __init__(self, prior=0.0):
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

class AlphaZeroAgent:

    def __init__(self, state_size, name="AlphaZeroAgent"):
        self.name = name
        self.state_size = state_size
        self.action_size = 10 * 121

        self.learning_rate = 0.0003
        self.weight_decay = 0.0001
        self.value_coef = 1.0
        self.num_simulations = 100
        self.c_puct = 1.5
        self.dirichlet_alpha = 0.30
        self.dirichlet_epsilon = 0.25
        self.temperature = 1.0
        self.batch_size = 64
        self.heuristic_leaf_blend = 0.60
        self.progress_prior_weight = 0.85
        self.root_progress_weight = 1.25
        self.root_race_weight = 0.0
        self.afterstate_guide_weight = 12.0
        # Fresh training runs can use a slightly larger network, but older
        # checkpoints still reload with their original architecture.
        self.network_hidden_size = 384
        self.network_num_blocks = 6
        self.afterstate_guide_network = None
        self.afterstate_guide_encoder = None
        self.afterstate_search_agent = None
        self.afterstate_guide_color = None
        self.afterstate_search_override = True
        self.endgame_override_pins = 5
        self.endgame_override_distance = 45
        self.recent_position_keys = deque(maxlen=12)
        self.training_buffer = deque(maxlen=50000)

        # One optimizer update trains the policy head and value head together.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_network_and_optimizer()
        self.train_updates = 0

    def _build_network_and_optimizer(self):
        self.policy_value_network = PolicyValueNetwork(
            self.state_size,
            self.action_size,
            hidden_size=self.network_hidden_size,
            num_blocks=self.network_num_blocks,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy_value_network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def copy_weights_from(self, other):
        if (
            self.network_hidden_size != int(other.network_hidden_size)
            or self.network_num_blocks != int(other.network_num_blocks)
        ):
            self.network_hidden_size = int(other.network_hidden_size)
            self.network_num_blocks = int(other.network_num_blocks)
            self._build_network_and_optimizer()
        self.policy_value_network.load_state_dict(other.policy_value_network.state_dict())
        self.optimizer.load_state_dict(other.optimizer.state_dict())
        self.num_simulations = int(other.num_simulations)
        self.c_puct = float(other.c_puct)
        self.network_hidden_size = int(other.network_hidden_size)
        self.network_num_blocks = int(other.network_num_blocks)
        self.heuristic_leaf_blend = float(other.heuristic_leaf_blend)
        self.progress_prior_weight = float(other.progress_prior_weight)
        self.root_progress_weight = float(other.root_progress_weight)
        self.root_race_weight = float(other.root_race_weight)
        self.afterstate_guide_weight = float(other.afterstate_guide_weight)
        self.afterstate_guide_network = other.afterstate_guide_network
        self.afterstate_guide_encoder = other.afterstate_guide_encoder
        self.afterstate_search_agent = other.afterstate_search_agent
        self.afterstate_search_override = bool(other.afterstate_search_override)
        self.endgame_override_pins = int(other.endgame_override_pins)
        self.endgame_override_distance = int(other.endgame_override_distance)
        self.train_updates = int(other.train_updates)

    def _current_progress_prior_weight(self):
        if self.train_updates >= 1500:
            return min(self.progress_prior_weight, 0.25)
        if self.train_updates >= 600:
            return min(self.progress_prior_weight, 0.50)
        return self.progress_prior_weight

    def action_to_index(self, action):
        pin_id, destination = action
        return int(pin_id) * 121 + int(destination)

    def action_to_policy_index(self, action, state=None, player_color=None):
        if state is not None:
            return state.policy_index(action, player_color)
        return self.action_to_index(action)

    def index_to_action(self, index):
        index = int(index)
        return index // 121, index % 121

    def build_mask(self, valid_actions, state=None, player_color=None):
        # Illegal actions are masked before softmax so they get no probability.
        mask = torch.zeros(self.action_size, dtype=torch.bool)
        for action in valid_actions:
            action_index = self.action_to_policy_index(action, state, player_color)
            if 0 <= action_index < self.action_size:
                mask[action_index] = True
        return mask

    def _masked_logits(self, logits, mask):
        fill = torch.full_like(logits, -1e9)
        return torch.where(mask, logits, fill)

    def _masked_logits_batch(self, logits, masks):
        fill = torch.full_like(logits, -1e9)
        return torch.where(masks, logits, fill)

    def _uses_endgame_scoring(self, state):
        # Near the finish line we care more about locking finished pins in place.
        pins = state.pins_in_goal(state.current_player)
        distance = state.total_distance_to_target(state.current_player)
        return pins >= self.endgame_override_pins or distance <= self.endgame_override_distance

    def _position_key_after(self, state, action):
        pin_id, destination = action
        positions = list(state.positions.get(state.current_player, ()))
        if not 0 <= int(pin_id) < len(positions):
            return (state.current_player, tuple(positions))
        positions[int(pin_id)] = int(destination)
        return (state.current_player, tuple(int(position) for position in positions))

    def _action_starts_in_target(self, state, action):
        pin_id, _ = action
        positions = state.positions.get(state.current_player, ())
        if not 0 <= int(pin_id) < len(positions):
            return False
        return int(positions[int(pin_id)]) in state.target_cells(state.current_player)

    def _remember_deterministic_choice(self, state, action, temp):
        if temp > 1e-6 or action is None:
            return
        if state.move_count <= 1:
            self.recent_position_keys.clear()
        self.recent_position_keys.append(self._position_key_after(state, action))

    def _selection_progress_score(self, state, action, use_endgame_score):
        if use_endgame_score:
            score = state.endgame_progress_score(action, state.current_player)
            repeated_position = self._position_key_after(state, action) in self.recent_position_keys
            no_progress = state.progress_score(action, state.current_player) <= 0.0
            # Repeating a no-progress endgame loop is usually just wasted motion.
            if repeated_position and (no_progress or self._action_starts_in_target(state, action)):
                score -= 25.0
            return score
        return state.progress_score(action, state.current_player)

    def _race_score_after(self, state, action):
        next_state, done, info = state.apply_action(action)
        if done and info.get("winner") == state.current_player:
            return 100.0
        if done:
            return next_state.race_margin(state.current_player)
        return next_state.race_margin(state.current_player)

    def enable_afterstate_guide(self, filepath, weight=None, verbose=True):
        try:
            from src.afterstate import AfterstateValueNetwork, _encode_from_positions
            from src.afterstate import AfterstateSearchAgent

            checkpoint = torch.load(filepath, map_location="cpu")
            if checkpoint.get("state_size") != self.state_size:
                return False

            guide = AfterstateValueNetwork(self.state_size).to(self.device)
            guide.load_state_dict(checkpoint["value_network_state_dict"])
            guide.eval()
            self.afterstate_guide_network = guide
            self.afterstate_guide_encoder = _encode_from_positions
            search_agent = AfterstateSearchAgent(self.state_size, player_color="yellow", name="AlphaZeroGuideSearch")
            if search_agent.load_model(filepath, verbose=False):
                search_agent.epsilon = 0.0
                self.afterstate_search_agent = search_agent
                # The guide is lane-specific in practice, so we remember which
                # colour it was trained for and avoid using it in the wrong frame.
                self.afterstate_guide_color = search_agent.player_color
            if weight is not None:
                self.afterstate_guide_weight = float(weight)
            if verbose:
                print(f"AlphaZero afterstate guide loaded from {filepath} (trained as {self.afterstate_guide_color})")
            return True
        except Exception as exc:
            print(f"Warning: could not load AlphaZero afterstate guide ({exc}).")
            self.afterstate_guide_network = None
            self.afterstate_guide_encoder = None
            self.afterstate_search_agent = None
            self.afterstate_guide_color = None
            return False

    def _afterstate_search_action(self, state, valid_actions):
        search_agent = self.afterstate_search_agent
        if search_agent is None or not valid_actions:
            return None

        root_colour = state.current_player
        best_action = None
        best_score = float("-inf")
        ordered_actions = search_agent._ordered_actions(state, root_colour, valid_actions, maximizing=True)

        for action in ordered_actions:
            next_state, done, info = state.apply_action(action)
            if done and info.get("winner") == root_colour:
                return action

            immediate = search_agent._afterstate_value(state, root_colour, action)
            score = immediate + search_agent.response_weight * search_agent._search(
                next_state,
                root_colour,
                search_agent.search_depth - 1,
            )
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _afterstate_feature(self, state, action):
        if self.afterstate_guide_encoder is None:
            return None

        colour = state.current_player
        opponent = state.opponent_of(colour)
        own_positions = list(state.positions.get(colour, ()))
        if not own_positions:
            return None

        pin_id, destination = action
        if not 0 <= int(pin_id) < len(own_positions):
            return None

        own_positions[int(pin_id)] = int(destination)
        own_positions.sort()
        opponent_positions = tuple(sorted(int(pos) for pos in state.positions.get(opponent, ())))
        return self.afterstate_guide_encoder(
            tuple(int(pos) for pos in own_positions),
            opponent_positions,
            state.board,
            colour,
        )

    def _afterstate_guide_scores(self, state, actions):
        if self.afterstate_guide_network is None or not actions:
            return np.zeros(len(actions), dtype=np.float64)

        features = []
        usable_actions = []
        for action in actions:
            feature = self._afterstate_feature(state, action)
            if feature is None:
                continue
            features.append(feature)
            usable_actions.append(action)

        if not features:
            return np.zeros(len(actions), dtype=np.float64)

        with torch.no_grad():
            tensor = torch.FloatTensor(features).to(self.device)
            values = self.afterstate_guide_network(tensor).squeeze(1).detach().cpu().numpy()

        score_by_action = {action: float(value) for action, value in zip(usable_actions, values)}
        scores = np.array([score_by_action.get(action, 0.0) for action in actions], dtype=np.float64)
        std = float(np.std(scores))
        # Normalizing keeps the guide on a predictable scale before blending.
        if std > 1e-6:
            scores = (scores - float(np.mean(scores))) / std
        return scores

    def _predict(self, state):
        state_tensor = torch.FloatTensor(state.state_vector(state.current_player)).unsqueeze(0).to(self.device)
        valid_actions = state.valid_actions()
        if not valid_actions:
            return {}, 0.0

        mask = self.build_mask(valid_actions, state=state).to(self.device)
        decayed_progress_weight = self._current_progress_prior_weight()
        with torch.no_grad():
            logits, value = self.policy_value_network(state_tensor)
            masked_logits = self._masked_logits(logits.squeeze(0), mask)
            if decayed_progress_weight > 0.0:
                # Early on, the progress bias gives the raw network a saner prior.
                progress_bias = torch.zeros_like(masked_logits)
                use_endgame_score = self._uses_endgame_scoring(state)
                for action in valid_actions:
                    if use_endgame_score:
                        progress = state.endgame_progress_score(action, state.current_player)
                    else:
                        progress = state.progress_score(action, state.current_player)
                    progress_bias[self.action_to_policy_index(action, state)] = max(-4.0, min(4.0, progress / 6.0))
                masked_logits = masked_logits + decayed_progress_weight * progress_bias
            probabilities = torch.softmax(masked_logits, dim=0)

        priors = {}
        total = 0.0
        for action in valid_actions:
            probability = float(probabilities[self.action_to_policy_index(action, state)].item())
            priors[action] = probability
            total += probability

        if total <= 0.0:
            uniform = 1.0 / len(valid_actions)
            priors = {action: uniform for action in valid_actions}
        else:
            priors = {action: probability / total for action, probability in priors.items()}

        return priors, float(value.item())

    def _expand(self, node, state):
        priors, network_value = self._predict(state)
        for action, prior in priors.items():
            node.children[action] = MCTSNode(prior=prior)

        # A small heuristic blend steadies the value target while the network is young.
        heuristic_value = state.heuristic_value(state.current_player)
        if self.train_updates >= 1500:
            blend = 0.15
        elif self.train_updates >= 600:
            blend = 0.30
        else:
            blend = self.heuristic_leaf_blend

        return (1.0 - blend) * float(network_value) + blend * heuristic_value

    def _select_child(self, node):
        # Standard PUCT: balance the current value against visit pressure.
        best_score = None
        best_action = None
        best_child = None
        sqrt_parent = math.sqrt(max(1, node.visit_count))

        for action, child in node.children.items():
            q_value = -child.value()
            u_value = self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q_value + u_value
            if best_score is None or score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _backpropagate(self, search_path, value):
        # Values flip sign every ply because players want opposite outcomes.
        current_value = float(value)
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += current_value
            current_value = -current_value

    def _add_root_noise(self, root):
        if not root.children:
            return

        actions = list(root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        for action, noise_value in zip(actions, noise):
            child = root.children[action]
            child.prior = (1.0 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * float(noise_value)

    def run_mcts(self, state, add_noise=False):
        # Each simulation walks to a leaf, expands it, then backs the value up.
        root = MCTSNode(prior=1.0)
        root_value = self._expand(root, state)
        root.visit_count = 1
        root.value_sum = float(root_value)

        if add_noise:
            self._add_root_noise(root)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            rollout_state = state
            done = False
            info = {"winner": None, "message": ""}

            while node.children:
                action, node = self._select_child(node)
                rollout_state, done, info = rollout_state.apply_action(action)
                search_path.append(node)
                if done:
                    break

            if done:
                leaf_value = rollout_state.terminal_value_for(rollout_state.current_player, info)
            else:
                leaf_value = self._expand(node, rollout_state)

            self._backpropagate(search_path, leaf_value)

        return root

    def search_policy(self, state, temperature=None, add_noise=False):
        valid_actions = state.valid_actions()
        temp = self.temperature if temperature is None else float(temperature)
        if state.move_count <= 1:
            self.recent_position_keys.clear()

        for action in valid_actions:
            _, done, info = state.apply_action(action)
            if done and info.get("winner") == state.current_player:
                policy_target = torch.zeros(self.action_size, dtype=torch.float32)
                policy_target[self.action_to_policy_index(action, state)] = 1.0
                self._remember_deterministic_choice(state, action, temp)
                return action, policy_target

        if temp <= 1e-6 and self.afterstate_search_override:
            # Deterministic play is allowed to use the stronger Afterstate search override.
            guide_action = self._afterstate_search_action(state, valid_actions)
            if guide_action is not None:
                policy_target = torch.zeros(self.action_size, dtype=torch.float32)
                policy_target[self.action_to_policy_index(guide_action, state)] = 1.0
                self._remember_deterministic_choice(state, guide_action, temp)
                return guide_action, policy_target

        if temp <= 1e-6:
            pins = state.pins_in_goal(state.current_player)
            distance = state.total_distance_to_target(state.current_player)
            if pins >= self.endgame_override_pins or distance <= self.endgame_override_distance:
                # Late endgames can skip full MCTS and choose directly from the
                # progress-plus-guide score.
                guide_scores = self._afterstate_guide_scores(state, valid_actions)
                scored_actions = []
                for index, action in enumerate(valid_actions):
                    score = self._selection_progress_score(state, action, True)
                    score += self.afterstate_guide_weight * guide_scores[index]
                    scored_actions.append((score, action))
                best_action = max(scored_actions, key=lambda item: item[0])[1]
                policy_target = torch.zeros(self.action_size, dtype=torch.float32)
                policy_target[self.action_to_policy_index(best_action, state)] = 1.0
                self._remember_deterministic_choice(state, best_action, temp)
                return best_action, policy_target

        root = self.run_mcts(state, add_noise=add_noise)
        if not root.children:
            return None, torch.zeros(self.action_size, dtype=torch.float32)

        actions = list(root.children.keys())
        visit_counts = np.array([root.children[action].visit_count for action in actions], dtype=np.float64)
        if temp <= 1e-6:
            probabilities = np.zeros_like(visit_counts)
            use_endgame_score = self._uses_endgame_scoring(state)
            progress_scores = np.array(
                [
                    self._selection_progress_score(state, action, use_endgame_score)
                    for action in actions
                ],
                dtype=np.float64,
            )
            race_scores = np.array(
                [self._race_score_after(state, action) / 10.0 for action in actions],
                dtype=np.float64,
            )
            guide_scores = self._afterstate_guide_scores(state, actions)
            root_scores = (
                visit_counts
                + self.root_progress_weight * progress_scores
                + self.root_race_weight * race_scores
                + self.afterstate_guide_weight * guide_scores
            )
            probabilities[int(np.argmax(root_scores))] = 1.0
        else:
            scaled = np.power(visit_counts, 1.0 / temp)
            total = float(np.sum(scaled))
            if total <= 0.0:
                probabilities = np.ones_like(visit_counts) / len(visit_counts)
            else:
                probabilities = scaled / total

        policy_target = torch.zeros(self.action_size, dtype=torch.float32)
        for action, probability in zip(actions, probabilities):
            policy_target[self.action_to_policy_index(action, state)] += float(probability)

        selected_index = int(np.random.choice(len(actions), p=probabilities))
        selected_action = actions[selected_index]
        self._remember_deterministic_choice(state, selected_action, temp)
        return selected_action, policy_target

    def choose_action_from_state(self, state, deterministic=True):
        action, policy_target = self.search_policy(
            state,
            temperature=0.0 if deterministic else 1.0,
            add_noise=not deterministic,
        )
        return action, policy_target

    def state_from_env(self, env):
        return AlphaZeroBoardState.from_env(env)

    def choose_action(self, env, valid_actions=None, deterministic=True):
        state = self.state_from_env(env)
        action, _ = self.choose_action_from_state(state, deterministic=deterministic)
        if valid_actions is None or action is None or action in valid_actions:
            return action
        return random.choice(valid_actions) if valid_actions else None

    def choose_action_from_board(self, pins_on_board, player_color, valid_actions, board, deterministic=True):
        state = AlphaZeroBoardState.from_board(
            pins_on_board,
            player_color,
            board,
            player_order=[player_color, board.colour_opposites.get(player_color, "")],
        )
        action, _ = self.choose_action_from_state(state, deterministic=deterministic)
        if action in valid_actions:
            return action
        return valid_actions[0] if valid_actions else None

    def add_examples(self, examples):
        self.training_buffer.extend(examples)

    def _sample_training_batch(self):
        # Plain random sampling can drift toward whichever result type is most
        # common in the buffer. A small bucket balance keeps wins, losses, and
        # draw-ish positions visible during training.
        examples = list(self.training_buffer)
        if len(examples) <= self.batch_size:
            return list(examples)

        positive = [example for example in examples if float(example[2]) > 0.5]
        negative = [example for example in examples if float(example[2]) < -0.5]
        neutral = [
            example
            for example in examples
            if -0.5 <= float(example[2]) <= 0.5
        ]

        buckets = [bucket for bucket in (positive, negative, neutral) if bucket]
        if len(buckets) <= 1:
            return random.sample(examples, self.batch_size)

        target_per_bucket = max(1, self.batch_size // len(buckets))
        batch = []
        used_ids = set()

        for bucket in buckets:
            take = min(len(bucket), target_per_bucket)
            for example in random.sample(bucket, take):
                batch.append(example)
                used_ids.add(id(example))

        if len(batch) < self.batch_size:
            remainder = [example for example in examples if id(example) not in used_ids]
            if len(remainder) >= self.batch_size - len(batch):
                batch.extend(random.sample(remainder, self.batch_size - len(batch)))
            else:
                batch.extend(remainder)

        if len(batch) > self.batch_size:
            batch = random.sample(batch, self.batch_size)

        return batch

    def supervised_update(self, states, policy_targets, masks, value_targets=None, epochs=3):
        if not states:
            return None

        states_tensor = torch.FloatTensor(states).to(self.device)
        targets_tensor = torch.stack(policy_targets).to(self.device)
        masks_tensor = torch.stack(masks).to(self.device)
        values_tensor = torch.FloatTensor(value_targets).to(self.device) if value_targets is not None else None
        batch_size = states_tensor.size(0)
        last_loss = 0.0

        for _ in range(epochs):
            permutation = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, self.batch_size):
                end = start + self.batch_size
                batch_index = permutation[start:end]
                batch_states = states_tensor[batch_index]
                batch_targets = targets_tensor[batch_index]
                batch_masks = masks_tensor[batch_index]

                logits, values = self.policy_value_network(batch_states)
                masked_logits = self._masked_logits_batch(logits, batch_masks)
                log_probabilities = torch.log_softmax(masked_logits, dim=1)
                policy_loss = -(batch_targets * log_probabilities).sum(dim=1).mean()

                if values_tensor is not None:
                    batch_values = values_tensor[batch_index]
                    value_loss = F.smooth_l1_loss(values.squeeze(1), batch_values)
                    loss = policy_loss + self.value_coef * value_loss
                else:
                    loss = policy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_value_network.parameters(), max_norm=1.0)
                self.optimizer.step()

                last_loss = float(policy_loss.item())

        return last_loss

    def train_step(self):
        if len(self.training_buffer) < self.batch_size:
            return None

        batch = self._sample_training_batch()
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        target_policies = torch.stack([e[1] for e in batch]).to(self.device)
        target_values = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        masks = torch.stack([e[3] for e in batch]).to(self.device)

        logits, values = self.policy_value_network(states)
        values = values.squeeze(1)
        masked_logits = self._masked_logits_batch(logits, masks)
        log_probabilities = torch.log_softmax(masked_logits, dim=1)

        policy_loss = -(target_policies * log_probabilities).sum(dim=1).mean()
        value_loss = F.smooth_l1_loss(values, target_values)
        loss = policy_loss + self.value_coef * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_value_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_updates += 1
        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "loss": float(loss.item()),
        }

    def save_model(self, filepath):
        torch.save(
            {
                "policy_value_state_dict": self.policy_value_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "state_size": self.state_size,
                "action_size": self.action_size,
                "board_perspective_policy": "six_lane_destination_v1",
                "num_simulations": self.num_simulations,
                "c_puct": self.c_puct,
                "network_hidden_size": self.network_hidden_size,
                "network_num_blocks": self.network_num_blocks,
                "heuristic_leaf_blend": self.heuristic_leaf_blend,
                "progress_prior_weight": self.progress_prior_weight,
                "root_progress_weight": self.root_progress_weight,
                "root_race_weight": self.root_race_weight,
                "afterstate_guide_weight": self.afterstate_guide_weight,
                "afterstate_search_override": self.afterstate_search_override,
                "endgame_override_pins": self.endgame_override_pins,
                "endgame_override_distance": self.endgame_override_distance,
                "train_updates": self.train_updates,
            },
            filepath,
        )
    def load_model(self, filepath, verbose=True):
        checkpoint = torch.load(filepath, map_location="cpu")
        if checkpoint.get("state_size") != self.state_size or checkpoint.get("action_size") != self.action_size:
            print("Warning: saved AlphaZero model shape does not match current code.")
            return False

        target_hidden_size = int(checkpoint.get("network_hidden_size", 256))
        target_num_blocks = int(checkpoint.get("network_num_blocks", 4))
        if (
            target_hidden_size != self.network_hidden_size
            or target_num_blocks != self.network_num_blocks
        ):
            self.network_hidden_size = target_hidden_size
            self.network_num_blocks = target_num_blocks
            self._build_network_and_optimizer()

        try:
            self.policy_value_network.load_state_dict(checkpoint["policy_value_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.num_simulations = int(checkpoint.get("num_simulations", self.num_simulations))
            self.c_puct = float(checkpoint.get("c_puct", self.c_puct))
            self.network_hidden_size = int(checkpoint.get("network_hidden_size", self.network_hidden_size))
            self.network_num_blocks = int(checkpoint.get("network_num_blocks", self.network_num_blocks))
            self.heuristic_leaf_blend = float(checkpoint.get("heuristic_leaf_blend", self.heuristic_leaf_blend))
            self.progress_prior_weight = float(checkpoint.get("progress_prior_weight", self.progress_prior_weight))
            self.root_progress_weight = float(checkpoint.get("root_progress_weight", self.root_progress_weight))
            self.root_race_weight = float(checkpoint.get("root_race_weight", self.root_race_weight))
            self.afterstate_guide_weight = float(checkpoint.get("afterstate_guide_weight", self.afterstate_guide_weight))
            self.afterstate_search_override = bool(checkpoint.get("afterstate_search_override", self.afterstate_search_override))
            self.endgame_override_pins = int(checkpoint.get("endgame_override_pins", self.endgame_override_pins))
            self.endgame_override_distance = int(checkpoint.get("endgame_override_distance", self.endgame_override_distance))
            self.train_updates = int(checkpoint.get("train_updates", 0))
            if verbose:
                print(f"Model loaded from {filepath}")
                print(
                    f"Saved sims: {self.num_simulations}  "
                    f"net: {self.network_hidden_size}x{self.network_num_blocks}  "
                    f"heuristic blend: {self.heuristic_leaf_blend:.2f}  "
                    f"progress prior: {self.progress_prior_weight:.2f}  "
                    f"root progress: {self.root_progress_weight:.2f}  "
                    f"root race: {self.root_race_weight:.2f}  "
                    f"guide: {self.afterstate_guide_weight:.2f}  "
                    f"train updates: {self.train_updates}"
                )
            return True
        except RuntimeError as exc:
            print(f"Warning: could not load AlphaZero weights ({exc}).")
            return False
