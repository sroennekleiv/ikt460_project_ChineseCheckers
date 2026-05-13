# Afterstate agent and its small search extension.

import random
from collections import deque

import torch
import torch.nn as nn

from src.alphazero import AlphaZeroBoardState
from src.board import HexBoard, HEX_DIRECTIONS
from src.network import AfterstateValueNetwork
from src.perspective import REFERENCE_COLOR, indices_to_reference_perspective

DIRECTIONS = HEX_DIRECTIONS

_REFERENCE_COLOR = REFERENCE_COLOR

def _choose_from_top_actions(scored_actions, top_k=3):
    if not scored_actions:
        return None

    # Exploration stays close to the best few moves instead of wandering over
    # the whole action list.
    ranked = sorted(scored_actions, key=lambda item: item[0], reverse=True)
    frontier = ranked[: max(1, min(int(top_k), len(ranked)))]
    best_score = frontier[0][0]
    close_actions = [action for score, action in frontier if score >= best_score - 0.15]
    pool = close_actions or [action for _, action in frontier]
    return random.choice(pool)

def _normalise_colour(colour):
    return str(colour)

def _target_positions(board, player_color):
    target_colour = board.colour_opposites.get(_normalise_colour(player_color), "")
    return set(board.axial_of_colour(target_colour))

def _position_state_from_pins(pins_on_board, player_color):
    player = _normalise_colour(player_color)
    # Replay only needs board indices for "my pins" and "their pins".
    own_positions = sorted(
        pin.axialindex for pin in pins_on_board if _normalise_colour(pin.color) == player
    )
    opp_positions = sorted(
        pin.axialindex for pin in pins_on_board if _normalise_colour(pin.color) != player
    )
    return tuple(own_positions), tuple(opp_positions)

def _reference_position_state(own_positions, opp_positions, board, player_color):
    colour = _normalise_colour(player_color)
    if colour == _REFERENCE_COLOR:
        return tuple(int(p) for p in own_positions), tuple(int(p) for p in opp_positions)
    return (
        indices_to_reference_perspective(board, colour, own_positions),
        indices_to_reference_perspective(board, colour, opp_positions),
    )

def _encode_from_positions(own_positions, opp_positions, board, player_color):
    # The network only learns one lane orientation. Every other lane is rotated
    # into that reference frame before features are built.
    if _normalise_colour(player_color) != _REFERENCE_COLOR:
        own_positions, opp_positions = _reference_position_state(
            own_positions,
            opp_positions,
            board,
            player_color,
        )
        player_color = _REFERENCE_COLOR

    board_size = len(board.cells)
    own_layer = [0] * board_size
    opp_layer = [0] * board_size

    for idx in own_positions:
        if 0 <= int(idx) < board_size:
            own_layer[int(idx)] = 1

    for idx in opp_positions:
        if 0 <= int(idx) < board_size:
            opp_layer[int(idx)] = 1

    targets = _target_positions(board, player_color)
    target_layer = [1 if idx in targets else 0 for idx in range(board_size)]
    return own_layer + opp_layer + target_layer

def encode_afterstate(pins_on_board, player_color, board, action):
    # An afterstate is the position right after our move lands.
    own_positions, opp_positions = _position_state_from_pins(pins_on_board, player_color)
    move_pin_id, destination = action
    destination = int(destination)

    moved_positions = list(own_positions)
    origin = None

    for pin in pins_on_board:
        if _normalise_colour(pin.color) != _normalise_colour(player_color):
            continue
        if int(pin.id) == int(move_pin_id):
            origin = int(pin.axialindex)
            break

    if origin is None:
        return _encode_from_positions(own_positions, opp_positions, board, player_color)

    for index, position in enumerate(moved_positions):
        if int(position) == origin:
            moved_positions[index] = destination
            break

    moved_positions.sort()
    return _encode_from_positions(tuple(moved_positions), opp_positions, board, player_color)

def _legal_destinations(start_idx, occupied, board, player_color):
    # This rebuilds legal moves from plain position data, which is why replay
    # does not need to store whole environments.
    current_cell = board.cells[int(start_idx)]
    q0, r0 = current_cell.q, current_cell.r
    possible = set()

    for dq, dr in DIRECTIONS:
        neighbour = board.hole_index_of.get((q0 + dq, r0 + dr))
        if neighbour is None or neighbour in occupied:
            continue
        destination = board.cells[neighbour]
        if current_cell.postype != _normalise_colour(player_color) and destination.postype == _normalise_colour(player_color):
            continue
        possible.add(neighbour)

    visited = {int(start_idx)}
    stack = [int(start_idx)]

    while stack:
        current = stack.pop()
        cell = board.cells[current]
        for dq, dr in DIRECTIONS:
            adjacent = board.hole_index_of.get((cell.q + dq, cell.r + dr))
            landing = board.hole_index_of.get((cell.q + 2 * dq, cell.r + 2 * dr))
            if adjacent is None or landing is None:
                continue
            if adjacent not in occupied or landing in occupied or landing in visited:
                continue
            destination = board.cells[landing]
            if current_cell.postype != _normalise_colour(player_color) and destination.postype == _normalise_colour(player_color):
                continue
            visited.add(landing)
            possible.add(landing)
            stack.append(landing)

    return sorted(possible)

def generate_afterstates_from_position_state(position_state, board, player_color):
    own_positions, opp_positions = position_state
    own_positions = tuple(sorted(int(pos) for pos in own_positions))
    opp_positions = tuple(sorted(int(pos) for pos in opp_positions))
    occupied = set(own_positions) | set(opp_positions)

    afterstates = []
    seen = set()

    # Different moves can land in the same afterstate, so we keep one copy.
    for index, start_idx in enumerate(own_positions):
        legal_moves = _legal_destinations(start_idx, occupied, board, player_color)
        for destination in legal_moves:
            moved_positions = list(own_positions)
            moved_positions[index] = int(destination)
            moved_positions.sort()
            moved_key = tuple(moved_positions)
            if moved_key in seen:
                continue
            seen.add(moved_key)
            afterstates.append(
                _encode_from_positions(moved_key, opp_positions, board, player_color)
            )

    return afterstates

class AfterstateValueAgent:

    def __init__(self, state_size, player_color="yellow", name="AfterstateValueAgent"):
        self.name = name
        self.state_size = state_size
        self.player_color = _normalise_colour(player_color)

        self.learning_rate = 0.0001
        self.gamma = 0.99
        self.epsilon = 0.15
        self.epsilon_decay = 0.9997
        self.epsilon_min = 0.01
        self.batch_size = 64

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.value_network = AfterstateValueNetwork(self.state_size).to(self.device)
        self.target_network = AfterstateValueNetwork(self.state_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.value_network.parameters(), lr=self.learning_rate)

        self.memory = deque(maxlen=30000)
        self.demo_memory = deque(maxlen=30000)
        self.demo_fraction = 0.50
        self.train_updates = 0

        # Replay only stores compact position data. A fixed helper board lets us
        # rebuild the legal follow-up afterstates later without storing full envs.
        self.replay_board = HexBoard(R=4, hole_radius=16, spacing=34)
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.value_network.state_dict())

    def afterstate_from_env_action(self, env, action, player_color=None):
        colour = _normalise_colour(player_color or env.get_current_player())
        return encode_afterstate(env.pins_on_board, colour, env.board, action)

    def position_state_from_env(self, env, player_color=None):
        colour = _normalise_colour(player_color or env.get_current_player())
        own_positions, opp_positions = _position_state_from_pins(env.pins_on_board, colour)
        return _reference_position_state(own_positions, opp_positions, env.board, colour)

    def choose_action(self, env, valid_actions, explore=False):
        if not valid_actions:
            return None

        current_player = _normalise_colour(env.get_current_player())
        if random.random() <= self.epsilon:
            return random.choice(valid_actions)

        # The value network scores every candidate afterstate in one batch.
        features = [
            encode_afterstate(env.pins_on_board, current_player, env.board, action)
            for action in valid_actions
        ]
        feature_tensor = torch.FloatTensor(features).to(self.device)

        with torch.no_grad():
            values = self.value_network(feature_tensor).squeeze(1)

        if explore:
            scored_actions = [
                (float(values[index].item()), action)
                for index, action in enumerate(valid_actions)
            ]
            return _choose_from_top_actions(scored_actions)

        best_index = int(torch.argmax(values).item())
        return valid_actions[best_index]

    def choose_action_from_board(self, pins_on_board, player_color, valid_actions, board, explore=False):
        if not valid_actions:
            return None

        if random.random() <= self.epsilon:
            return random.choice(valid_actions)

        features = [
            encode_afterstate(pins_on_board, player_color, board, action)
            for action in valid_actions
        ]
        feature_tensor = torch.FloatTensor(features).to(self.device)

        with torch.no_grad():
            values = self.value_network(feature_tensor).squeeze(1)

        if explore:
            scored_actions = [
                (float(values[index].item()), action)
                for index, action in enumerate(valid_actions)
            ]
            return _choose_from_top_actions(scored_actions)

        best_index = int(torch.argmax(values).item())
        return valid_actions[best_index]

    def remember(self, afterstate, reward, next_position_state, done, demo=False):
        # Stored experiences stay lightweight so longer runs fit comfortably in RAM.
        if next_position_state is not None:
            own_positions, opp_positions = next_position_state
            stored_next = (
                tuple(int(pos) for pos in own_positions),
                tuple(int(pos) for pos in opp_positions),
            )
        else:
            stored_next = None

        experience = (list(afterstate), float(reward), stored_next, bool(done))
        if demo:
            self.demo_memory.append(experience)
        else:
            self.memory.append(experience)

    def _sample_batch(self):
        # Demo memory keeps the agent grounded while online memory lets it adapt.
        combined_size = len(self.memory) + len(self.demo_memory)
        if combined_size < self.batch_size:
            return None

        demo_target = min(len(self.demo_memory), int(self.batch_size * self.demo_fraction))
        online_target = min(len(self.memory), self.batch_size - demo_target)

        if demo_target + online_target < self.batch_size:
            combined = list(self.demo_memory) + list(self.memory)
            return random.sample(combined, self.batch_size)

        batch = []
        if online_target:
            batch.extend(random.sample(self.memory, online_target))
        if demo_target:
            batch.extend(random.sample(self.demo_memory, demo_target))
        random.shuffle(batch)
        return batch

    def replay(self):
        batch = self._sample_batch()
        if batch is None:
            return False

        states = torch.FloatTensor([sample[0] for sample in batch]).to(self.device)
        current_values = self.value_network(states).squeeze(1)

        rewards_dones      = []
        groups             = []
        flat_afterstates   = []
        group_offsets      = []

        for _, reward, next_pos, done in batch:
            rewards_dones.append((float(reward), bool(done)))
            if not done and next_pos is not None:
                g = generate_afterstates_from_position_state(
                        next_pos, self.replay_board, self.player_color)
                groups.append(g)
                start = len(flat_afterstates)
                flat_afterstates.extend(g)
                group_offsets.append((start, len(flat_afterstates)))
            else:
                groups.append(None)
                group_offsets.append(None)

        # Instead of running the target network once per example, we flatten all
        # generated next afterstates into one batch and score them together.
        flat_values = None
        if flat_afterstates:
            with torch.no_grad():
                flat_t = torch.FloatTensor(flat_afterstates).to(self.device)
                flat_values = self.target_network(flat_t).squeeze(1)

        targets = []
        for i, (reward, done) in enumerate(rewards_dones):
            if group_offsets[i] is None or flat_values is None:
                targets.append(reward)
            else:
                s, e = group_offsets[i]
                if e > s:
                    targets.append(reward + self.gamma * flat_values[s:e].max().item())
                else:
                    targets.append(reward)

        target_tensor = torch.FloatTensor(targets).to(self.device)
        loss = nn.SmoothL1Loss()(current_values, target_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.train_updates += 1
        return True

    def save_model(self, filepath):
        torch.save({
            "value_network_state_dict": self.value_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "state_size": self.state_size,
            "memory_size": len(self.memory),
            "demo_memory_size": len(self.demo_memory),
            "train_updates": self.train_updates,
            "player_color": self.player_color,
            "board_perspective": "six_lane_rotation_v1",
        }, filepath)
    def load_model(self, filepath, verbose=True):
        checkpoint = torch.load(filepath, map_location="cpu")
        if checkpoint.get("state_size") != self.state_size:
            print("Warning: saved afterstate model shape does not match current code.")
            return False

        try:
            self.value_network.load_state_dict(checkpoint["value_network_state_dict"])
            self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epsilon = checkpoint.get("epsilon", self.epsilon)
            self.train_updates = checkpoint.get("train_updates", 0)
            # Replay regenerates afterstates in the shared reference frame every
            # time. For that reason we keep the runtime colour fixed even if an
            # older checkpoint stores a more specific lane label.
            self.player_color = _REFERENCE_COLOR
            demo_size = checkpoint.get("demo_memory_size", 0)
            if verbose:
                print(f"Model loaded from {filepath}")
                print(
                    f"Saved epsilon: {self.epsilon:.3f}  memory: {checkpoint['memory_size']}  "
                    f"demo: {demo_size}"
                )
            return True
        except RuntimeError as exc:
            print(f"Warning: could not load afterstate weights ({exc}).")
            return False

class AfterstateSearchAgent(AfterstateValueAgent):

    def __init__(self, state_size, player_color="yellow", name="AfterstateSearchAgent"):
        super().__init__(state_size=state_size, player_color=player_color, name=name)
        self.search_depth = 2
        self.search_width = 10
        self.response_width = 6
        self.response_weight = 0.85

    def _encode_state_action(self, state, player_color, action):
        colour = _normalise_colour(player_color)
        opponent = state.opponent_of(colour)
        own_positions = tuple(state.positions.get(colour, ()))
        opp_positions = tuple(state.positions.get(opponent, ()))

        pin_id, destination = action
        moved_positions = list(own_positions)
        if 0 <= int(pin_id) < len(moved_positions):
            moved_positions[int(pin_id)] = int(destination)
        moved_positions.sort()
        return _encode_from_positions(tuple(moved_positions), opp_positions, state.board, colour)

    def _afterstate_value(self, state, player_color, action):
        feature = self._encode_state_action(state, player_color, action)
        feature_tensor = torch.FloatTensor(feature).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.value_network(feature_tensor).squeeze(1)
        return float(value.item())

    def _terminal_score(self, state, root_colour):
        root = _normalise_colour(root_colour)
        opponent = state.opponent_of(root)
        if state.pins_in_goal(root) == 10:
            return 1000.0
        if state.pins_in_goal(opponent) == 10:
            return -1000.0
        return None

    def _leaf_score(self, state, root_colour):
        root = _normalise_colour(root_colour)
        current = _normalise_colour(state.current_player)
        heuristic = 12.0 * state.heuristic_value(root)

        if current == root:
            actions = state.valid_actions()
            if not actions:
                return heuristic
            scored = [self._afterstate_value(state, root, action) for action in actions]
            return max(scored) + heuristic

        return heuristic

    def _ordered_actions(self, state, player_color, actions, maximizing):
        colour = _normalise_colour(player_color)
        if maximizing:
            # On our own turn we trust the value network and search the best
            # looking afterstates first.
            scored = [
                (self._afterstate_value(state, colour, action), action)
                for action in actions
            ]
            scored.sort(key=lambda item: item[0], reverse=True)
            return [action for _, action in scored[:self.search_width]]

        # Opponent replies use the cheaper progress score to keep search light.
        scored = [
            (state.progress_score(action, colour), action)
            for action in actions
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [action for _, action in scored[:self.response_width]]

    def _search(self, state, root_colour, depth):
        terminal = self._terminal_score(state, root_colour)
        if terminal is not None:
            return terminal
        if depth <= 0:
            return self._leaf_score(state, root_colour)

        current = _normalise_colour(state.current_player)
        root = _normalise_colour(root_colour)
        maximizing = current == root
        actions = state.valid_actions()
        if not actions:
            return self._leaf_score(state, root_colour)

        ordered_actions = self._ordered_actions(state, current, actions, maximizing=maximizing)

        if maximizing:
            best_score = float("-inf")
            for action in ordered_actions:
                immediate = self._afterstate_value(state, current, action)
                next_state, _, _ = state.apply_action(action)
                score = immediate + self.response_weight * self._search(next_state, root_colour, depth - 1)
                if score > best_score:
                    best_score = score
            return best_score

        worst_score = float("inf")
        for action in ordered_actions:
            next_state, _, _ = state.apply_action(action)
            score = self._search(next_state, root_colour, depth - 1)
            if score < worst_score:
                worst_score = score
        return worst_score

    def choose_action(self, env, valid_actions, explore=False):
        if not valid_actions:
            return None
        return self.choose_action_from_board(
            env.pins_on_board,
            env.get_current_player(),
            valid_actions,
            env.board,
            explore=explore,
        )

    def choose_action_from_board(self, pins_on_board, player_color, valid_actions, board, explore=False):
        if not valid_actions:
            return None

        if random.random() <= self.epsilon:
            return random.choice(valid_actions)

        root_colour = _normalise_colour(player_color)
        state = AlphaZeroBoardState.from_board(
            pins_on_board,
            root_colour,
            board,
            player_order=[root_colour, board.colour_opposites.get(root_colour, "")],
        )

        best_action = None
        best_score = float("-inf")
        scored_actions = []
        ordered_actions = self._ordered_actions(state, root_colour, valid_actions, maximizing=True)

        # Search scores our move, then subtracts the strength of the best reply.
        for action in ordered_actions:
            immediate = self._afterstate_value(state, root_colour, action)
            next_state, _, _ = state.apply_action(action)
            score = immediate + self.response_weight * self._search(next_state, root_colour, self.search_depth - 1)
            scored_actions.append((score, action))
            if score > best_score:
                best_score = score
                best_action = action

        if explore:
            exploratory_action = _choose_from_top_actions(scored_actions)
            if exploratory_action is not None:
                return exploratory_action

        return best_action if best_action is not None else valid_actions[0]
