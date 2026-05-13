# Local entry point for playing games against the trained agents.

import os
import random
import select
import sys
import time

import numpy as np

from src.afterstate import AfterstateSearchAgent, AfterstateValueAgent
from src.agents import GreedyAgent, MinimaxAgent, RandomAgent
from src.alphazero import AlphaZeroAgent
from src.board import HexBoard
from src.game import GameManager, RUNNING, TimeoutManager
from src.gui import BoardGUI
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

np.random.seed()
random.seed()

SUPPORTED_PLAYER_COUNTS = {2, 3, 4, 6}
OPENING_EXPLORATION_TURNS = 12
SELFPLAY_STYLE_MODES = {"asvr", "asvg", "asvm", "azvr", "azvg", "azvm", "azvas"}
TWO_PLAYER_BASE_COLORS = ["red", "lawn green", "yellow"]
RANDOM_TWO_PLAYER_MODES = {
    "hvh", "hvrandom", "hvgreedy", "gvr", "rvr", "gvg",
    "hvminimax", "mvm", "hvafter", "asvr", "asvg", "asvm",
    "hvalpha", "azvr", "azvg", "azvm", "azvas",
}

def load_afterstate_agent():
    # Local play always uses this fixed state size, so we can load the model
    # without spinning up a training probe first.
    state_size = 363
    model_path = first_existing(
        AFTERSTATE_BEST_MODEL,
        AFTERSTATE_TRAINED_MODEL,
        AFTERSTATE_FINAL_MODEL,
    )
    if os.path.exists(model_path):
        search_agent = AfterstateSearchAgent(
            state_size=state_size,
            player_color="yellow",
            name="AfterstateSearchAgent",
        )
        if search_agent.load_model(model_path):
            search_agent.epsilon = 0.0
            print("Afterstate+Search model loaded.")
            return search_agent

        agent = AfterstateValueAgent(
            state_size=state_size,
            player_color="yellow",
            name="AfterstateAgent",
        )
        if agent.load_model(model_path):
            agent.epsilon = 0.0
            print("Afterstate model loaded.")
            return agent

        print("Afterstate model is old. Retrain before using Afterstate mode.")
    else:
        print("No trained afterstate model found. Afterstate agent will play randomly.")
    return AfterstateValueAgent(state_size=state_size, player_color="yellow", name="AfterstateAgent")

def load_alphazero_agent():
    state_size = 363
    agent = AlphaZeroAgent(state_size=state_size, name="AlphaZeroAgent")
    model_path = first_existing(
        ALPHAZERO_BEST_MODEL,
        ALPHAZERO_EXTERNAL_MODEL,
        ALPHAZERO_TRAINED_MODEL,
        ALPHAZERO_FINAL_MODEL,
    )
    if os.path.exists(model_path):
        if agent.load_model(model_path):
            guide_path = first_existing(AFTERSTATE_BEST_MODEL, AFTERSTATE_TRAINED_MODEL, AFTERSTATE_FINAL_MODEL)
            if os.path.exists(guide_path):
                agent.enable_afterstate_guide(guide_path)
            print("AlphaZero model loaded.")
        else:
            print("AlphaZero model is old. Retrain before using AlphaZero mode.")
    else:
        print("No trained AlphaZero model found. AlphaZero agent will play randomly.")
    return agent

def check_win(pins_on_board, player_color, board):
    opposite_color = board.colour_opposites.get(player_color)
    if not opposite_color:
        return False
    target_cells = {i for i, cell in enumerate(board.cells) if cell.postype == opposite_color}
    return sum(1 for p in pins_on_board if p.color == player_color and p.axialindex in target_cells) == 10

def is_reverse_action(action, current_player, pins_on_board, recent_moves):
    pin_id, dest = action
    pin = next((p for p in pins_on_board if p.color == current_player and p.id == pin_id), None)
    if pin is None:
        return False

    old_pos = pin.axialindex
    for colour, old_pin_id, old_cell, new_cell in recent_moves[-12:]:
        if colour == current_player and old_pin_id == pin_id and old_cell == dest and new_cell == old_pos:
            return True
    return False

def remember_move(recent_moves, current_player, pin_id, old_cell, new_cell):
    recent_moves.append((current_player, int(pin_id), old_cell, new_cell))
    if len(recent_moves) > 30:
        recent_moves.pop(0)

def _mode_player_labels(game_mode):
    labels = {
        "hvh": ("Human 1", "Human 2"),
        "hvrandom": ("Human", "Random"),
        "hvgreedy": ("Human", "Greedy"),
        "gvr": ("Greedy", "Random"),
        "rvr": ("Random", "Random"),
        "gvg": ("Greedy", "Greedy"),
        "hvminimax": ("Human", "Minimax"),
        "mvm": ("Minimax", "Minimax"),
        "hvafter": ("Afterstate", "Human"),
        "asvr": ("Afterstate", "Random"),
        "asvg": ("Afterstate", "Greedy"),
        "asvm": ("Afterstate", "Minimax"),
        "hvalpha": ("AlphaZero", "Human"),
        "azvr": ("AlphaZero", "Random"),
        "azvg": ("AlphaZero", "Greedy"),
        "azvm": ("AlphaZero", "Minimax"),
        "azvas": ("Afterstate", "AlphaZero"),
    }
    return labels.get(game_mode)

def build_player_roles(game_mode, player_colors):
    if len(player_colors) != 2:
        return {}
    labels = _mode_player_labels(game_mode)
    if labels is None:
        return {}
    return {
        str(player_colors[index]): labels[index]
        for index in range(min(len(player_colors), len(labels)))
    }

def _sample_two_player_colours(game):
    base_colour = random.choice(TWO_PLAYER_BASE_COLORS)
    colours = [base_colour, game.board.colour_opposites[base_colour]]
    random.shuffle(colours)
    return colours

def _announce_side_assignment(game_mode, colours):
    labels = _mode_player_labels(game_mode)
    if labels is None:
        return
    first_label, second_label = labels
    print(f"Controllers this game: {first_label.upper()}={colours[0].upper()}, {second_label.upper()}={colours[1].upper()}.")
    print(f"Turn order this game: {colours[0].upper()} moves first, {colours[1].upper()} moves second.")

def configure_player_colors_for_mode(game, game_mode, player_colors):
    # Local two-player demos reshuffle both the active opposite-colour lane and
    # which side moves first so it is easy to spot colour-pair quirks by eye.
    if game.num_players == 2 and game_mode in RANDOM_TWO_PLAYER_MODES:
        colours = _sample_two_player_colours(game)
        game.sync_player_state(colours)
        print("This two-player mode uses a random opposite-colour lane each game.")
        print("Both the lane and the side assignment are reshuffled on every start.")
        print(f"Active lane this game: {colours[0].upper()} vs {colours[1].upper()}.")
        _announce_side_assignment(game_mode, colours)
        return colours

    return player_colors

def _get_ai_type(game_mode, current_player, player_colors):
    # The mode string decides which side, if any, should be controlled by an AI.
    p0, p1 = player_colors[0], player_colors[1]
    rules = {
        "hvafter":  {p0: "afterstate"},
        "asvr":     {p0: "afterstate", p1: "random"},
        "asvg":     {p0: "afterstate", p1: "greedy"},
        "asvm":     {p0: "afterstate", p1: "minimax"},
        "hvalpha":  {p0: "alphazero"},
        "azvr":     {p0: "alphazero", p1: "random"},
        "azvg":     {p0: "alphazero", p1: "greedy"},
        "azvm":     {p0: "alphazero", p1: "minimax"},
        "azvas":    {p0: "afterstate", p1: "alphazero"},
        "hvrandom": {p1: "random"},
        "hvgreedy": {p1: "greedy"},
        "hvminimax":{p1: "minimax"},
        "gvr":      {p0: "greedy",    p1: "random"},
        "rvr":      {p0: "random",    p1: "random"},
        "gvg":      {p0: "greedy",    p1: "greedy"},
        "mvm":      {p0: "minimax",   p1: "minimax"},
    }
    return rules.get(game_mode, {}).get(current_player)

def _select_ai_action(ai_type, agents, pins_on_board, current_player, remaining, board, explore_opening=False):
    # Only the learned agents use extra opening exploration to vary their demos.
    if ai_type == "afterstate":
        return agents["afterstate"].choose_action_from_board(
            pins_on_board,
            current_player,
            remaining,
            board,
            explore=explore_opening,
        )
    if ai_type == "alphazero":
        return agents["alphazero"].choose_action_from_board(
            pins_on_board,
            current_player,
            remaining,
            board,
            deterministic=not explore_opening,
        )
    if ai_type == "greedy":
        return agents["greedy"].choose_action_from_board(pins_on_board, current_player, remaining, board)
    if ai_type == "minimax":
        return agents["minimax"].choose_action_from_board(pins_on_board, current_player, remaining, board)
    if ai_type == "random":
        return agents["random"].choose_action(None, remaining)
    return None

if __name__ == "__main__":
    print("Chinese Checkers")
    print()

    while RUNNING:
        command = input("Type 'start' to begin, or 'exit' to quit: ")

        if command == 'exit':
            break

        if command != 'start':
            continue

        board     = HexBoard(R=4, hole_radius=16, spacing=34)
        num_turns = 0

        num_players = int(input("Enter number of players (2, 3, 4, or 6): "))
        if num_players not in SUPPORTED_PLAYER_COUNTS:
            print("Invalid number of players.")
            continue

        game          = GameManager(board, num_players=num_players)
        player_colors = game.assign_players_colors(num_players)

        # The menu stays flat on purpose so it is easy to launch quick demos.
        print("Choose game mode:")
        print("  1. Human vs Human")
        print("  2. Human vs Random")
        print("  3. Human vs Greedy")
        print("  4. Greedy vs Random")
        print("  5. Random vs Random")
        print("  6. Greedy vs Greedy")
        print("  7. Human vs Minimax")
        print("  8. Minimax vs Minimax")
        print("  9. Human vs Afterstate")
        print("  10. Afterstate vs Random")
        print("  11. Afterstate vs Greedy")
        print("  12. Afterstate vs Minimax")
        print("  13. Human vs AlphaZero")
        print("  14. AlphaZero vs Random")
        print("  15. AlphaZero vs Greedy")
        print("  16. AlphaZero vs Minimax")
        print("  17. Afterstate vs AlphaZero")
        mode_input = input("Enter number (1-17): ").strip()
        game_mode  = {'1': 'hvh', '2': 'hvrandom',
                      '3': 'hvgreedy', '4': 'gvr',
                      '5': 'rvr', '6': 'gvg', '7': 'hvminimax',
                      '8': 'mvm', '9': 'hvafter', '10': 'asvr',
                      '11': 'asvg', '12': 'asvm', '13': 'hvalpha',
                      '14': 'azvr', '15': 'azvg', '16': 'azvm',
                      '17': 'azvas'}.get(mode_input, 'hvh')
        mode_labels = {'hvh': 'Human vs Human',
                       'hvrandom': 'Human vs Random',
                       'hvgreedy': 'Human vs Greedy', 'gvr': 'Greedy vs Random',
                       'rvr': 'Random vs Random', 'gvg': 'Greedy vs Greedy',
                       'hvminimax': 'Human vs Minimax', 'mvm': 'Minimax vs Minimax',
                       'hvafter': 'Human vs Afterstate', 'asvr': 'Afterstate vs Random',
                       'asvg': 'Afterstate vs Greedy', 'asvm': 'Afterstate vs Minimax',
                       'hvalpha': 'Human vs AlphaZero', 'azvr': 'AlphaZero vs Random',
                       'azvg': 'AlphaZero vs Greedy', 'azvm': 'AlphaZero vs Minimax',
                       'azvas': 'Afterstate vs AlphaZero'}
        print(f"Mode: {mode_labels.get(game_mode, game_mode)}")

        player_colors = configure_player_colors_for_mode(game, game_mode, player_colors)
        player_roles = build_player_roles(game_mode, player_colors)
        pins_on_board = game.place_pins_to_board(player_colors)

        print("\nPlayers: " + ", ".join(c.upper() for c in player_colors))
        if player_roles:
            print("Controllers: " + ", ".join(f"{colour.upper()}={role}" for colour, role in player_roles.items()))
        print()
        board.print_ascii(pins=pins_on_board, empty="·")

        agents = {
            "afterstate":load_afterstate_agent()  if game_mode in ("hvafter", "asvr", "asvg", "asvm", "azvas") else None,
            "alphazero": load_alphazero_agent()   if game_mode in ("hvalpha", "azvr", "azvg", "azvm", "azvas") else None,
            "random":    RandomAgent()            if game_mode in ("hvrandom", "gvr", "rvr", "azvr") else None,
            "greedy":    GreedyAgent()            if game_mode in ("hvgreedy", "gvr", "gvg", "asvg", "azvg") else None,
            "minimax":   MinimaxAgent(depth=2)    if game_mode in ("hvminimax", "mvm", "asvm", "azvm") else None,
        }

        gui = BoardGUI(board, pins_on_board, player_roles=player_roles)
        gui.window.update()
        game.timeout_manager = TimeoutManager(game.player_colors, turn_time_limit=60, game_time_limit=1800)
        game.timeout_manager.start_game_timer()

        MAX_TURNS           = 500
        board_state_history = {}
        recent_moves        = []

        while True:
            current_player = game.move_manager.get_current_player()
            game.timeout_manager.start_turn()
            gui.set_turn(num_turns + 1)
            gui._update_active_dot(current_player)

            print(f"\nTurn {num_turns + 1}: {current_player.upper()}")

            if num_turns >= MAX_TURNS:
                print(f"Max turns ({MAX_TURNS}) reached. Game over.")
                break

            # Local play uses a shorter repetition rule than training. The goal
            # here is to stop visible loops quickly instead of analysing them.
            state_key = tuple(sorted((p.color, p.axialindex) for p in pins_on_board))
            board_state_history[state_key] = board_state_history.get(state_key, 0) + 1
            if board_state_history[state_key] >= 3:
                print("Board state repeated 3 times. Draw.")
                break

            if game.timeout_manager.is_turn_timeout():
                print(f"{current_player.upper()} timed out. Turn skipped.")
                game.timeout_manager.end_turn(current_player)
                game.move_manager.next_player_turn()
                break

            if game.timeout_manager.is_game_timeout(current_player):
                print(f"{current_player.upper()} exceeded total game time. Game over.")
                game.timeout_manager.end_game_timer()
                break

            # AI turns all share the same flow: gather legal moves, optionally
            # filter obvious reversals, then let the selected agent choose.
            ai_type = _get_ai_type(game_mode, current_player, player_colors)

            if ai_type is not None:
                gui.set_status(f"{current_player.upper()} is thinking...")
                gui.window.update()
                if game_mode in ("gvr", "rvr", "gvg", "mvm", "asvr", "asvg", "asvm", "azvr", "azvg", "azvm", "azvas"):
                    time.sleep(0.7)

                occupied      = {pin.axialindex for pin in pins_on_board}
                valid_actions = [(pin.id, dest)
                                 for pin in pins_on_board if pin.color == current_player
                                 for dest in pin.get_legal_moves() if dest not in occupied]

                if not valid_actions:
                    print(f"{current_player.upper()} has no valid moves. Skipping.")
                    game.timeout_manager.end_turn(current_player)
                    game.move_manager.next_player_turn()
                    continue

                move_pin_success = False
                attempted = set()
                while not move_pin_success and len(attempted) < len(valid_actions):
                    remaining = [a for a in valid_actions if a not in attempted]
                    safe_remaining = [a for a in remaining if not is_reverse_action(a, current_player, pins_on_board, recent_moves)]
                    if safe_remaining:
                        remaining = safe_remaining

                    explore_opening = (
                        game_mode in SELFPLAY_STYLE_MODES
                        and ai_type in {"afterstate", "alphazero"}
                        and num_turns < OPENING_EXPLORATION_TURNS
                    )
                    action = _select_ai_action(
                        ai_type,
                        agents,
                        pins_on_board,
                        current_player,
                        remaining,
                        board,
                        explore_opening=explore_opening,
                    )

                    if action is None:
                        break
                    attempted.add(action)
                    pin_id, dest_id  = action
                    players_pin      = game.get_pin_id_of_player(pins_on_board, current_player, str(pin_id))
                    if players_pin is None:
                        continue
                    old_cell         = players_pin.axialindex
                    move_path        = players_pin.get_move_path(dest_id)
                    move_pin_success = players_pin.place_pin(dest_id)

                if move_pin_success:
                    remember_move(recent_moves, current_player, pin_id, old_cell, dest_id)
                    print(f"  {current_player.upper()} moved pin {pin_id} -> cell {dest_id}")
                    board.print_ascii(pins=pins_on_board, empty='·')
                    gui.animate_move(
                        pins_on_board,
                        pin_id,
                        current_player,
                        move_path,
                        status_msg=f"{current_player.upper()} moved pin {pin_id} -> {dest_id}",
                    )
                    time_used = game.timeout_manager.end_turn(current_player)
                    game.move_manager.log_move(current_player, f"({pin_id},{dest_id})", players_pin, dest_id, time_used)
                    if check_win(pins_on_board, current_player, board):
                        print(f"\n{current_player.upper()} WINS after {num_turns + 1} turns!")
                        gui.show_winner(current_player)
                        break
                    game.move_manager.next_player_turn()
                    num_turns += 1
                else:
                    print(f"{current_player.upper()} could not complete a move. Skipping.")
                    game.timeout_manager.end_turn(current_player)
                    game.move_manager.next_player_turn()
                continue

            # Human turns can come either from typed input or GUI clicks. Both
            # paths end up as the same `(pin_id, destination)` action tuple.
            gui.enable_click(current_player)
            print("Click a pin, or type: pin_id,dest_id  |  pass  |  exit")
            print("> ", end='', flush=True)

            move_input    = None
            pin_id_click  = None
            dest_id_click = None

            while move_input is None and pin_id_click is None:
                gui.window.update()
                if gui._pending_action is not None:
                    pin_id_click, dest_id_click = gui._pending_action
                    gui._pending_action = None
                    break
                try:
                    if select.select([sys.stdin], [], [], 0)[0]:
                        move_input = sys.stdin.readline().strip()
                except Exception:
                    pass
            gui.disable_click()

            if pin_id_click is not None:
                print(f"({pin_id_click},{dest_id_click})")
                players_pin      = game.get_pin_id_of_player(pins_on_board, current_player, str(pin_id_click))
                if players_pin is None:
                    print("Invalid pin.")
                    continue
                old_cell         = players_pin.axialindex
                move_path        = players_pin.get_move_path(dest_id_click)
                move_pin_success = players_pin.place_pin(dest_id_click)
                if move_pin_success:
                    remember_move(recent_moves, current_player, pin_id_click, old_cell, dest_id_click)
                    print(f"  {current_player.upper()} moved pin {pin_id_click} -> cell {dest_id_click}")
                    board.print_ascii(pins=pins_on_board, empty='·')
                    gui.animate_move(
                        pins_on_board,
                        pin_id_click,
                        current_player,
                        move_path,
                        status_msg=f"You moved pin {pin_id_click} -> {dest_id_click}",
                    )
                    time_used = game.timeout_manager.end_turn(current_player)
                    game.move_manager.log_move(current_player, f"({pin_id_click},{dest_id_click})", players_pin, dest_id_click, time_used)
                    if check_win(pins_on_board, current_player, board):
                        print(f"\n{current_player.upper()} WINS after {num_turns + 1} turns!")
                        gui.show_winner(current_player)
                        break
                    game.move_manager.next_player_turn()
                    num_turns += 1
                else:
                    print("Invalid move, please try again.")
                continue

            if not move_input:
                move_input = ''

            if move_input.lower() == 'exit':
                break

            if move_input.lower() == 'pass':
                print(f"  {current_player.upper()} passes.")
                time_used = game.timeout_manager.end_turn(current_player)
                game.move_manager.log_move(current_player, "pass", None, None, time_used)
                game.move_manager.next_player_turn()
                continue

            if ',' not in move_input or len(move_input.split(',')) != 2:
                print("Invalid format. Use: pin_id,dest_id")
                continue

            if game.timeout_manager.is_turn_timeout():
                print(f"{current_player.upper()} timed out. Turn skipped.")
                time_used = game.timeout_manager.end_turn(current_player)
                game.move_manager.log_move(current_player, "timeout", None, None, time_used)
                game.move_manager.next_player_turn()
                continue

            pin_num = move_input.split(',')[0].replace('(', '')
            try:
                dest_id = int(move_input.split(',')[1].replace(')', ''))
            except ValueError:
                print("Invalid destination.")
                continue

            players_pin      = game.get_pin_id_of_player(pins_on_board, current_player, pin_num)
            if players_pin is None:
                print("Invalid pin.")
                continue
            old_cell         = players_pin.axialindex
            move_path        = players_pin.get_move_path(dest_id)
            move_pin_success = players_pin.place_pin(dest_id)

            if move_pin_success:
                remember_move(recent_moves, current_player, players_pin.id, old_cell, dest_id)
                print(f"  {current_player.upper()} moved pin {pin_num} -> cell {dest_id}")
                board.print_ascii(pins=pins_on_board, empty='·')
                gui.animate_move(pins_on_board, players_pin.id, current_player, move_path)
                time_used = game.timeout_manager.end_turn(current_player)
                game.move_manager.log_move(current_player, move_input, players_pin, dest_id, time_used)
                if check_win(pins_on_board, current_player, board):
                    print(f"\n{current_player.upper()} WINS after {num_turns + 1} turns!")
                    gui.show_winner(current_player)
                    break
                game.move_manager.next_player_turn()
                num_turns += 1
            else:
                print("Invalid move. Your pins:", [(p.id, p.axialindex) for p in pins_on_board if p.color == current_player])

        print(f"\nGame over. {num_turns} turns played.")
        print("\nMove history:")
        for color, moves in game.move_manager.move_history.items():
            print(f"\n{color.upper()} ({len(moves)} moves):")
            for move in moves:
                print(f"  {move}")
