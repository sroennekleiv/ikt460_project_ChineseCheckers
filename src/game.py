# Game helpers shared by the GUI and the training environment.

import time

import numpy as np

from src.board import Pin

BASE_COLORS = ["red", "lawn green", "yellow"]
OPPOSITE_COLORS = ["blue", "gray0", "purple"]
ALL_COLORS = BASE_COLORS + OPPOSITE_COLORS

RUNNING = True
ASSIGNED_COLORS = []

class MoveManager:

    def __init__(self, player_colors=None):
        # The move manager only tracks turn order and move history. It does not
        # know the rules of the board itself.
        self.player_colors = [str(color) for color in player_colors] if player_colors is not None else []
        self.current_player_index = 0
        self.move_history = {color: [] for color in self.player_colors}
        self.turn_counter = 0

    def get_current_player(self):
        if not self.player_colors:
            return None
        self.current_player_index %= len(self.player_colors)
        return self.player_colors[self.current_player_index]

    def next_player_turn(self):
        self.current_player_index = (self.current_player_index + 1) % len(self.player_colors)
        return self.get_current_player()

    def log_move(self, player_color, move, pin, destination, time_used):
        player_color = str(player_color)
        self.turn_counter += 1
        # Keeping a compact move record makes it easy to score a finished game
        # or inspect what happened in the GUI.
        move_record = {
            "turn": self.turn_counter,
            "move": move,
            "pin_id": pin.id if pin is not None else None,
            "destination": destination,
            "time_used": round(time_used, 2),
        }
        self.move_history.setdefault(player_color, []).append(move_record)

    def get_player_moves(self, player_color):
        return self.move_history.get(str(player_color), [])

    def get_total_time(self, player_color):
        moves = self.get_player_moves(player_color)
        return sum(move["time_used"] for move in moves)

class TimeoutManager:

    def __init__(self, players, turn_time_limit=60, game_time_limit=600):
        # The GUI and tournament tools can reuse the same timeout bookkeeping.
        self.turn_time_limit = turn_time_limit
        self.game_time_limit = game_time_limit
        self.game_time = None
        self.turn_start_time = None
        self.total_time = {str(player): 0.0 for player in players}

    def start_turn(self):
        self.turn_start_time = time.time()

    def start_game_timer(self):
        self.game_start_time = time.time()

    def end_game_timer(self):
        self.game_time = time.time() - self.game_start_time

    def end_turn(self, player_color):
        player_color = str(player_color)
        if self.turn_start_time is None:
            return 0.0

        elapsed = time.time() - self.turn_start_time
        self.total_time[player_color] = self.total_time.get(player_color, 0.0) + elapsed
        self.turn_start_time = None
        return elapsed

    def is_turn_timeout(self):
        if self.turn_start_time is None:
            return False
        return (time.time() - self.turn_start_time) >= self.turn_time_limit

    def is_game_timeout(self, player_color):
        player_color = str(player_color)
        return self.total_time.get(player_color, 0.0) >= self.game_time_limit

    def get_total_time(self, player_color):
        return self.total_time.get(str(player_color), 0.0)

class GameLogic:

    def __init__(self, board, players, move_manager=None, timeout_manager=None):
        self.board = board
        self.players = players
        self.move_manager = move_manager or MoveManager(players)
        self.timeout_manager = timeout_manager

    def count_pins_in_goal(self, player_color):
        # Finishing all 10 pins in the opposite triangle is the real win
        # condition, so many scores build from this one count.
        goal_color = self.board.colour_opposites[player_color]
        goal_positions = {
            index
            for index, cell in enumerate(self.board.cells)
            if cell.postype == goal_color
        }

        count = 0
        for pin in getattr(self.board, "pins_on_board", []):
            if pin.color == player_color and pin.axialindex in goal_positions:
                count += 1
        return count

    def check_status_winning(self, player_color):
        return self.count_pins_in_goal(player_color) == 10

    def check_win_condition(self, player_color):
        pins_in_goal = self.count_pins_in_goal(player_color)
        player_total_moves = self.move_manager.get_player_moves(player_color)
        player_total_time = self.move_manager.get_total_time(player_color)

        if self.check_status_winning(player_color) and not self.reached_game_timeout(player_color):
            print(f"{player_color.upper()} wins the game with {pins_in_goal} pins in the goal area!")

        score = pins_in_goal * 100 - len(player_total_moves) - int(player_total_time)
        return {
            "player": player_color,
            "pins_in_goal": pins_in_goal,
            "total_moves": len(player_total_moves),
            "total_time": player_total_time,
            "score": score,
            "won": self.check_status_winning(player_color),
        }

    def get_scoring_table(self):
        # This table is mostly for end-of-game summaries, not for training.
        table = [self.check_win_condition(player) for player in self.players]
        table.sort(key=lambda row: row["score"], reverse=True)
        return table

    def reached_game_timeout(self, player_color=None):
        if self.timeout_manager is None or player_color is None:
            return False
        return self.timeout_manager.is_game_timeout(player_color)

class GameManager:

    def __init__(self, board, num_players, player_colors=None):
        self.board = board
        self.num_players = num_players
        self.player_colors = list(player_colors) if player_colors is not None else []
        self.pins_on_board = []

        self.move_manager = None
        self.timeout_manager = None
        self.game_logic = None
        self.sync_player_state(self.player_colors)

    def sync_player_state(self, player_colors=None):
        # Any time the colour order changes, the move manager and timeout
        # manager need to be rebuilt around that new order.
        if player_colors is not None:
            self.player_colors = [str(color) for color in player_colors]
        else:
            self.player_colors = [str(color) for color in self.player_colors]

        self.move_manager = MoveManager(self.player_colors)
        self.timeout_manager = TimeoutManager(
            self.player_colors,
            turn_time_limit=30,
            game_time_limit=1800,
        )
        self.game_logic = GameLogic(
            self.board,
            self.player_colors,
            move_manager=self.move_manager,
            timeout_manager=self.timeout_manager,
        )

    def assign_players_colors(self, number_of_players):
        # Two and four players must use opposite pairs. Three and six players
        # use the natural colour sets directly.
        if number_of_players == 2:
            base_colour = np.random.choice(BASE_COLORS)
            opposite_colour = self.board.colour_opposites[base_colour]
            self.player_colors = [base_colour, opposite_colour]
        elif number_of_players == 3:
            self.player_colors = list(np.random.choice(BASE_COLORS, size=3, replace=False))
        elif number_of_players == 4:
            self.player_colors = []
            for colour in np.random.choice(BASE_COLORS, size=2, replace=False):
                self.player_colors.append(colour)
                self.player_colors.append(self.board.colour_opposites[colour])
        elif number_of_players == 6:
            self.player_colors = BASE_COLORS.copy() + OPPOSITE_COLORS.copy()
        else:
            raise ValueError("Supported player counts are 2, 3, 4, and 6.")

        print("Assigned colors:")
        for index, colour in enumerate(self.player_colors, start=1):
            print(f"Player {index}: {colour}")

        self.sync_player_state(self.player_colors)
        return self.player_colors

    def place_pins_to_board(self, pins_color):
        # Reset occupancy first so calling reset twice does not leave stale
        # occupied flags behind on the board.
        for pin in self.pins_on_board:
            self.board.cells[pin.axialindex].occupied = False

        self.pins_on_board = []
        for colour in pins_color:
            home = self.board.axial_of_colour(colour)
            self.pins_on_board.extend(
                Pin(self.board, home[index], id=index, color=colour)
                for index in range(10)
            )

        self.board.pins_on_board = self.pins_on_board
        return self.pins_on_board

    def get_pin_id_of_player(self, pins_on_board, player_color, pin_id):
        # Training and GUI code both ask for pins by colour and id, so the
        # lookup lives here instead of being duplicated everywhere.
        matches = [
            pin
            for pin in pins_on_board
            if pin.color == player_color and str(pin.id) == str(pin_id)
        ]
        if not matches:
            print("No matching pin found.")
            return None
        return matches[0]
