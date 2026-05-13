# Environment wrapper used by the learning code.

from src.board import HexBoard
from src.game import GameManager

class ChineseCheckersEnv:

    def __init__(self, num_players=2, player_colors=None, max_turns=500, max_repetitions=6):
        self.num_players = num_players
        self.player_colors = player_colors
        self.max_turns = max_turns
        self.max_repetitions = max_repetitions

        self.board = None
        self.game = None
        self.pins_on_board = []
        self.done = False
        self.turn_count = 0
        self.last_actions = {}
        self.move_history = []
        self.state_counts = {}

    def reset(self):
        # Reset always builds a fresh board so no occupied flags leak between games.
        self.board = HexBoard(R=4, hole_radius=16, spacing=34)
        self.game = GameManager(self.board, num_players=self.num_players)

        if self.player_colors is None:
            self.player_colors = self.game.assign_players_colors(self.num_players)
        else:
            self.player_colors = [str(color) for color in self.player_colors]
            self.game.sync_player_state(self.player_colors)

        self.pins_on_board = self.game.place_pins_to_board(self.player_colors)

        self.done = False
        self.turn_count = 0
        self.last_actions = {color: None for color in self.player_colors}
        self.move_history = []
        self.state_counts = {}

        return self.get_state()

    def get_current_player(self):
        return self.game.move_manager.get_current_player()

    def get_player_pins(self, player_color):
        return [pin for pin in self.pins_on_board if str(pin.color) == str(player_color)]

    def total_distance_to_target(self, player_color):
        # We match each unfinished pin to its own target cell instead of letting
        # several pins "share" the same nearest hole. That makes endgame scoring
        # much more realistic when only a few target cells remain.
        player_pins = self.get_player_pins(player_color)
        target_set = set(self.get_target_positions(player_color))

        if not player_pins or not target_set:
            return 0

        in_target_cells = {p.axialindex for p in player_pins if p.axialindex in target_set}
        outside_pins = [p for p in player_pins if p.axialindex not in target_set]

        if not outside_pins:
            return 0

        available = [t for t in target_set if t not in in_target_cells]

        if not available:
            return 0

        def min_dist_to_available(pin):
            return min(self.axial_distance(pin.axialindex, t) for t in available)

        outside_sorted = sorted(outside_pins, key=min_dist_to_available, reverse=True)

        total = 0
        remaining = list(available)
        for pin in outside_sorted:
            if not remaining:
                break
            best_t = min(remaining, key=lambda t: self.axial_distance(pin.axialindex, t))
            total += self.axial_distance(pin.axialindex, best_t)
            remaining.remove(best_t)

        return total

    def get_state(self):
        # The learned agents always see the board as three binary layers:
        # where my pins are, where the opponent pins are, and which cells form
        # my target triangle.
        current_player   = self.get_current_player()
        target_positions = set(self.get_target_positions(current_player))

        own_layer = []
        opp_layer = []

        for idx, cell in enumerate(self.board.cells):
            if not cell.occupied:
                own_layer.append(0)
                opp_layer.append(0)
            else:
                pin = self.get_pin_at_position(idx)
                if pin is None:
                    own_layer.append(0)
                    opp_layer.append(0)
                elif str(pin.color) == str(current_player):
                    own_layer.append(1)
                    opp_layer.append(0)
                else:
                    own_layer.append(0)
                    opp_layer.append(1)

        target_layer = [1 if idx in target_positions else 0
                        for idx in range(len(self.board.cells))]
        return own_layer + opp_layer + target_layer

    def get_pin_at_position(self, board_index):
        for pin in self.pins_on_board:
            if pin.axialindex == board_index:
                return pin
        return None

    def get_valid_actions(self):
        current_player = self.get_current_player()
        actions = []

        # The environment uses the live pin objects as the source of truth for legality.
        for pin in self.pins_on_board:
            if str(pin.color) != str(current_player):
                continue

            for move in pin.get_legal_moves():
                actions.append((pin.id, move))

        return actions

    def _board_state_key(self):
        # Repetition depends on both the occupied cells and whose turn it is.
        # Two identical layouts are not the same state if the side to move flipped.
        positions_by_color = {}
        for pin in self.pins_on_board:
            colour = str(pin.color)
            positions_by_color.setdefault(colour, []).append(int(pin.axialindex))

        packed_positions = tuple(
            sorted(
                (colour, tuple(sorted(positions)))
                for colour, positions in positions_by_color.items()
            )
        )
        return str(self.get_current_player()), packed_positions

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True, {"message": "Game already finished"}

        pin_id, destination_cell = action
        current_player = self.get_current_player()

        pin = self.game.get_pin_id_of_player(self.pins_on_board, current_player, pin_id)

        if pin is None:
            return self.get_state(), -1, False, {"message": "Invalid pin"}

        action_score = self.evaluate_action_progress(action, current_player)
        old_position = pin.axialindex
        move_success = pin.place_pin(destination_cell)

        if not move_success:
            return self.get_state(), -1, False, {"message": "Illegal move"}

        self.turn_count += 1
        self.game.move_manager.log_move(current_player, action, pin, destination_cell, 0.0)
        self.last_actions[current_player] = (pin.id, old_position, destination_cell)

        self.move_history.append((current_player, pin_id, old_position, destination_cell))
        if len(self.move_history) > 30:
            self.move_history.pop(0)

        # The immediate reward starts with local move progress before win or draw bonuses.
        reward = action_score

        if self.has_player_won(current_player):
            self.done = True
            return self.get_state(), reward + 100, True, {"message": f"{current_player} wins"}

        self.game.move_manager.next_player_turn()

        # Repetition draws stop endless back-and-forth cycles from producing
        # unbounded games and strange training targets.
        board_key = self._board_state_key()
        self.state_counts[board_key] = self.state_counts.get(board_key, 0) + 1
        if self.state_counts[board_key] >= self.max_repetitions:
            self.done = True
            return self.get_state(), reward, True, {"message": "Draw by repetition"}

        if self.turn_count >= self.max_turns:
            self.done = True
            return self.get_state(), reward, True, {"message": "Max turns reached"}

        return self.get_state(), reward, False, {}

    def get_state_for_player(self, player_color):
        # This is the fixed-player version of get_state(). It is useful when
        # training code wants a view for a specific colour instead of the side
        # whose turn it currently is.
        target_positions = set(self.get_target_positions(player_color))

        own_layer = []
        opp_layer = []

        for idx, cell in enumerate(self.board.cells):
            if not cell.occupied:
                own_layer.append(0)
                opp_layer.append(0)
            else:
                pin = self.get_pin_at_position(idx)
                if pin is None:
                    own_layer.append(0)
                    opp_layer.append(0)
                elif str(pin.color) == str(player_color):
                    own_layer.append(1)
                    opp_layer.append(0)
                else:
                    own_layer.append(0)
                    opp_layer.append(1)

        target_layer = [1 if idx in target_positions else 0
                        for idx in range(len(self.board.cells))]
        return own_layer + opp_layer + target_layer

    def get_valid_actions_for_player(self, player_color):
        actions = []
        for pin in self.pins_on_board:
            if str(pin.color) != str(player_color):
                continue
            for move in pin.get_legal_moves():
                actions.append((pin.id, move))
        return actions

    def axial_distance(self, idx1, idx2):
        cell_1 = self.board.cells[idx1]
        cell_2 = self.board.cells[idx2]

        q_diff = cell_1.q - cell_2.q
        r_diff = cell_1.r - cell_2.r
        s_diff = (-cell_1.q - cell_1.r) - (-cell_2.q - cell_2.r)

        return max(abs(q_diff), abs(r_diff), abs(s_diff))

    def get_target_positions(self, player_color):
        # A player's goal is always the opposite home triangle.
        target_color = self.board.colour_opposites[player_color]
        return self.board.axial_of_colour(target_color)

    def evaluate_action_progress(self, action, player_color):
        # This scorer is intentionally simple: reward moves that reduce the
        # assignment distance, strongly reward entering the goal, and penalize
        # moves that pull finished pins back out or repeat obvious cycles.
        pin_id, destination_cell = action
        pin = self.game.get_pin_id_of_player(self.pins_on_board, player_color, pin_id)

        if pin is None:
            return -1

        old_position = pin.axialindex
        old_distance = self.total_distance_to_target(player_color)

        target_positions = set(self.get_target_positions(player_color))

        pin.axialindex = destination_cell
        new_distance = self.total_distance_to_target(player_color)
        pin.axialindex = old_position

        score = old_distance - new_distance

        if destination_cell in target_positions and old_position not in target_positions:
            score += 20

        if old_position in target_positions and destination_cell not in target_positions:
            score -= 30

        recent_own = [(m[1], m[2], m[3]) for m in self.move_history[-10:] if m[0] == player_color]
        for move_pin_id, from_pos, to_pos in recent_own:
            if pin_id == move_pin_id and old_position == to_pos and destination_cell == from_pos:
                score -= 8
                break

        return score

    def count_player_pins_in_target(self, player_color):
        target_color = self.board.colour_opposites[player_color]
        target_positions = set(self.board.axial_of_colour(target_color))

        count = 0

        for pin in self.pins_on_board:
            if str(pin.color) == str(player_color) and pin.axialindex in target_positions:
                count += 1

        return count

    def has_player_won(self, player_color):
        return self.count_player_pins_in_target(player_color) == 10
