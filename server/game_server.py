import time
import numpy as np

from src.board import Pin
from src.game import BASE_COLORS, GameLogic, MoveManager, OPPOSITE_COLORS, TimeoutManager


class GameManager:
    def __init__(self, board, num_players, player_colors=None):
        self.board = board
        self.player_colors = list(
            player_colors) if player_colors is not None else []  # List of player colors in turn order
        self.num_players = num_players

        self.pins_on_board = []

        self.move_manager = None
        self.timeout_manager = None
        self.game_logic = None
        self.sync_player_state(self.player_colors)

    def sync_player_state(self, player_colors=None):
        if player_colors is not None:
            self.player_colors = [str(color) for color in player_colors]
        else:
            self.player_colors = [str(color) for color in self.player_colors]

        self.move_manager = MoveManager(self.player_colors)
        self.timeout_manager = TimeoutManager(self.player_colors, turn_time_limit=30, game_time_limit=1800)
        self.game_logic = GameLogic(self.board, self.player_colors)
        self.game_logic.move_manager = self.move_manager

    def assign_players_colors(self, number_of_players):
        if number_of_players == 2:
            p1 = np.random.choice(BASE_COLORS)
            p2 = self.board.colour_opposites[p1]
            self.player_colors = [p1, p2]

        elif number_of_players == 3:
            pair_choices = BASE_COLORS.copy()
            chosen = np.random.choice(pair_choices, size=3, replace=False)
            self.player_colors = list(chosen)

        elif number_of_players == 4:
            pairs = []
            chosen = np.random.choice(BASE_COLORS, size=2, replace=False)
            for c in chosen:
                pairs.append(c)
                pairs.append(self.board.colour_opposites[c])
            self.player_colors = pairs

        elif number_of_players == 6:
            self.player_colors = BASE_COLORS.copy() + OPPOSITE_COLORS.copy()

        print("Assigned colors:")
        for i, c in enumerate(self.player_colors):
            print(f"Player {i + 1}: {c}")

        self.sync_player_state(self.player_colors)
        return self.player_colors

    def place_pins_to_board(self, pins_color):
        for pin in self.pins_on_board:
            self.board.cells[pin.axialindex].occupied = False
        self.pins_on_board = []

        for color in pins_color:
            home = self.board.axial_of_colour(color)
            self.pins_on_board += [
                Pin(self.board, home[i], id=i, color=color)
                for i in range(10)
            ]

        self.board.pins_on_board = self.pins_on_board

        return self.pins_on_board

    def get_pin_id_of_player(self, pins_on_board, player_color, pin_id) -> Pin:
        matching_pins = [pin for pin in pins_on_board if pin.color == player_color and str(pin.id) == str(pin_id)]
        # Check if a matching pin was found
        if not matching_pins:
            print("No matching pin found.")
            return None

        players_pin_id = matching_pins[0]  # Get the matching pin
        return players_pin_id
