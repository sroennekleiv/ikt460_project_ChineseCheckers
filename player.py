# Tournament client for the external game server.

import os
import json
import random
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.board import HexBoard
from src.paths import (
    AFTERSTATE_BEST_MODEL,
    AFTERSTATE_FINAL_MODEL,
    AFTERSTATE_TRAINED_MODEL,
    ALPHAZERO_BEST_MODEL,
    ALPHAZERO_EXTERNAL_MODEL,
    first_existing,
)

# The tournament server address lives here so it is easy to change without
# digging through the RPC code further down.
HOST = "10.245.30.129"
PORT = 50555

# The learned two-player agents all use the same 363-feature board view:
# own occupancy, opponent occupancy, and target cells.
STATE_SIZE = 363
PREFERRED_TOURNAMENT_AGENT = "alphazero"
AFTERSTATE_TOURNAMENT_COLORS = {"yellow", "purple", "red", "blue", "lawn green", "gray0"}
ALPHAZERO_TOURNAMENT_COLORS = {"yellow", "purple", "red", "blue", "lawn green", "gray0"}
TWO_PLAYER_FALLBACK_DEPTH = 3
MULTI_PLAYER_FALLBACK_DEPTH = 2
TWO_PLAYER_FALLBACK_TIME_BUDGET = 1.2
MULTI_PLAYER_FALLBACK_TIME_BUDGET = 0.8

_board = HexBoard(R=4, hole_radius=16, spacing=34)
_DIRS  = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

# The board geometry never changes during a tournament game, so we precompute
# each colour's target cells once and reuse them for all fallback evaluation.
_target_cells = {
    c: set(_board.axial_of_colour(_board.colour_opposites[c]))
    for c in _board.colour_opposites
}

def rpc(payload):
    # The tournament server expects one request per short-lived socket.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(10.0)
    try:
        s.connect((HOST, PORT))
    except Exception as e:
        return {"ok": False, "error": f"connect-failed: {e}"}
    s.sendall(json.dumps(payload).encode("utf-8"))
    chunks = []
    while True:
        chunk = s.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
    data = b"".join(chunks)
    s.close()
    if not data:
        return {"ok": False, "error": "no-response"}
    try:
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        return {"ok": False, "error": f"bad-json: {e}"}

def _axial_dist(idx1, idx2):
    c1, c2 = _board.cells[idx1], _board.cells[idx2]
    return max(abs(c1.q - c2.q),
               abs(c1.r - c2.r),
               abs((-c1.q - c1.r) - (-c2.q - c2.r)))

def _total_dist(positions, target_cells):
    # This distance is intentionally assignment-based. Once a pin claims a goal
    # cell, another unfinished pin should not pretend that the same cell is
    # still available, or the fallback search becomes far too optimistic.
    if not target_cells or not positions:
        return 0

    in_target  = set(p for p in positions if p in target_cells)
    outside    = [p for p in positions if p not in target_cells]

    if not outside:
        return 0

    available = [t for t in target_cells if t not in in_target]
    if not available:
        return 0

    # We match the most constrained pins first, which usually gives a better
    # estimate of how much real work is left in crowded endgames.
    outside_sorted = sorted(outside,
                            key=lambda p: min(_axial_dist(p, t) for t in available),
                            reverse=True)
    total     = 0
    remaining = list(available)
    for pin in outside_sorted:
        if not remaining:
            break
        best_t = min(remaining, key=lambda t: _axial_dist(pin, t))
        total += _axial_dist(pin, best_t)
        remaining.remove(best_t)

    return total

def _get_moves(pos_dict, colour):
    occupied = set()
    for positions in pos_dict.values():
        occupied.update(positions)

    valid = []
    for pin_id, start_idx in enumerate(pos_dict.get(colour, [])):
        start_cell = _board.cells[start_idx]
        q0, r0 = start_cell.q, start_cell.r
        possible = set()

        for dq, dr in _DIRS:
            ni = _board.hole_index_of.get((q0 + dq, r0 + dr))
            if ni is not None and ni not in occupied:
                possible.add(ni)

        visited = {start_idx}
        stack   = [start_idx]
        while stack:
            curr = stack.pop()
            cq, cr = _board.cells[curr].q, _board.cells[curr].r
            for dq, dr in _DIRS:
                adj  = _board.hole_index_of.get((cq + dq,     cr + dr))
                land = _board.hole_index_of.get((cq + 2 * dq, cr + 2 * dr))
                if adj is None or land is None:
                    continue
                if adj in occupied and land not in occupied and land not in visited:
                    possible.add(land)
                    visited.add(land)
                    stack.append(land)

        for dest in possible:
            if start_cell.postype != colour and _board.cells[dest].postype == colour:
                continue
            valid.append((pin_id, dest))

    return valid

def _sim(pos_dict, colour, pin_id, dest):
    new = {c: list(p) for c, p in pos_dict.items()}
    new[colour][pin_id] = dest
    return new

def _evaluate(pos_dict, my_colour):
    my_pos    = pos_dict.get(my_colour, [])
    my_target = _target_cells[my_colour]
    my_dist   = _total_dist(my_pos, my_target)
    my_in     = sum(1 for p in my_pos if p in my_target)

    # Blocking matters in multi-player positions too, so the fallback scorer
    # lightly penalizes opponents sitting inside our target triangle.
    opp_blocking = sum(
        1 for c, positions in pos_dict.items()
        if c != my_colour
        for p in positions if p in my_target
    )

    return -my_dist + 15 * my_in - 5 * opp_blocking

def _move_score(pos_dict, colour, pin_id, dest):
    # Alpha-beta gets much more useful when promising moves are searched first.
    # This quick score is only for ordering, not for the final decision itself.
    positions = pos_dict.get(colour, [])
    target    = _target_cells[colour]
    old_dist  = _total_dist(positions, target)
    simulated = list(positions)
    simulated[pin_id] = dest
    score = old_dist - _total_dist(simulated, target)
    if dest in target and positions[pin_id] not in target:
        score += 20
    if positions[pin_id] in target and dest not in target:
        score -= 30
    return score

# The fallback search clears this cache every move. Reusing scores within a
# single decision saves a lot of repeated work without risking stale positions.
_tt: dict = {}

def _pos_key(pos_dict, current_colour, depth, maximizing):
    return (
        hash(tuple(sorted((c, tuple(sorted(p))) for c, p in pos_dict.items()))),
        current_colour, depth, maximizing
    )

def _minimax(pos_dict, my_colour, player_order, turn_idx, depth, alpha, beta, deadline):
    # This is a paranoid-style search: we maximise our own outcome and treat
    # all other players as a combined opposing force.
    my_pos = pos_dict.get(my_colour, [])

    # Any fully completed home triangle is treated as a terminal result.
    for colour, positions in pos_dict.items():
        if len(positions) == 10 and all(p in _target_cells[colour] for p in positions):
            return 10000 if colour == my_colour else -10000

    if depth == 0 or time.time() > deadline:
        return _evaluate(pos_dict, my_colour)

    current_colour = player_order[turn_idx]
    maximizing     = (current_colour == my_colour)
    key            = _pos_key(pos_dict, current_colour, depth, maximizing)
    if key in _tt:
        return _tt[key]

    moves = _get_moves(pos_dict, current_colour)
    next_idx = (turn_idx + 1) % len(player_order)

    if not moves:
        result = _minimax(pos_dict, my_colour, player_order, next_idx, depth - 1, alpha, beta, deadline)
        _tt[key] = result
        return result

    moves.sort(key=lambda m: _move_score(pos_dict, current_colour, m[0], m[1]),
               reverse=maximizing)

    if maximizing:
        best = float('-inf')
        for pin_id, dest in moves:
            score = _minimax(_sim(pos_dict, current_colour, pin_id, dest),
                             my_colour, player_order, next_idx, depth - 1, alpha, beta, deadline)
            best  = max(best, score)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
    else:
        best = float('inf')
        for pin_id, dest in moves:
            score = _minimax(_sim(pos_dict, current_colour, pin_id, dest),
                             my_colour, player_order, next_idx, depth - 1, alpha, beta, deadline)
            best  = min(best, score)
            beta  = min(beta, best)
            if beta <= alpha:
                break

    _tt[key] = best
    return best

def pick_best_action(server_state, my_colour, legal_moves, depth=4, time_budget=1.2):
    global _tt
    _tt = {}

    pos_dict      = {c: list(p) for c, p in server_state.get("pins", {}).items()}
    valid_actions = [(int(pid), dest) for pid, dests in legal_moves.items() for dest in dests]
    if not valid_actions:
        return None

    # Server turn order is the real authority. The two-player fallback only
    # exists so local testing still works if that information is missing.
    players      = server_state.get("players", [])
    all_colours  = [p["colour"] for p in players if "colour" in p]
    if not all_colours:
        all_colours = [my_colour, _board.colour_opposites.get(my_colour, my_colour)]

    # Rotating the order so our colour comes first makes the recursive search
    # code easier to reason about.
    if my_colour in all_colours:
        idx = all_colours.index(my_colour)
        all_colours = all_colours[idx:] + all_colours[:idx]

    # Once we simulate our own move, search continues with the next colour.
    next_idx = 1 % len(all_colours)

    best_action = None
    best_score  = float('-inf')
    search_start = time.time()
    deadline    = search_start + float(time_budget)

    for d in range(1, depth + 1):
        if time.time() > deadline:
            break

        candidate_action = None
        candidate_score  = float('-inf')
        alpha = float('-inf')
        beta  = float('inf')

        sorted_actions = sorted(valid_actions,
                                key=lambda m: _move_score(pos_dict, my_colour, m[0], m[1]),
                                reverse=True)

        for pin_id, dest in sorted_actions:
            if time.time() > deadline:
                break
            score = _minimax(_sim(pos_dict, my_colour, pin_id, dest),
                             my_colour, all_colours, next_idx, d - 1, alpha, beta, deadline)
            score += random.uniform(0, 1e-4)
            if score > candidate_score:
                candidate_score  = score
                candidate_action = (pin_id, dest)
            alpha = max(alpha, candidate_score)

        if candidate_action is not None:
            best_action = candidate_action
            best_score  = candidate_score

        elapsed = time.time() - search_start
        print(f"  depth={d}: score={best_score:.1f}  action={best_action}  t={elapsed:.2f}s"
              f"  players={len(all_colours)}")

    return best_action if best_action else valid_actions[0]

def should_use_learned_agent(server_state, my_colour):
    # The learned models were trained for two-player play, so bigger games use
    # the handcrafted fallback search instead.
    players = server_state.get("players", [])
    player_count = len(players) if players else len(server_state.get("pins", {}))
    opposite = _board.colour_opposites.get(my_colour, "")
    learned_colours = (
        ALPHAZERO_TOURNAMENT_COLORS
        if PREFERRED_TOURNAMENT_AGENT == "alphazero"
        else AFTERSTATE_TOURNAMENT_COLORS
    )
    return (
        player_count == 2
        and my_colour in learned_colours
        and opposite in learned_colours
    )

def is_reverse_action(server_state, current_player, action, recent_moves, lookback=8):
    pin_id, dest = action
    # We only care about this pin's recent history, because that is what tells
    # us whether the proposed destination is part of a small cycle.
    pin_history = [
        (old_cell, new_cell)
        for colour, old_pin_id, old_cell, new_cell in recent_moves
        if colour == current_player and int(old_pin_id) == int(pin_id)
    ]
    for old_cell, _ in pin_history[-lookback:]:
        if int(old_cell) == int(dest):
            return True
    return False

def remember_move(recent_moves, current_player, pin_id, old_cell, new_cell):
    recent_moves.append((str(current_player), int(pin_id), int(old_cell), int(new_cell)))
    if len(recent_moves) > 30:
        recent_moves.pop(0)

class _PinProxy:
    __slots__ = ("id", "color", "axialindex")
    def __init__(self, pin_id, color, axialindex):
        self.id        = pin_id
        self.color     = color
        self.axialindex = axialindex

def _make_pins_on_board(server_state):
    proxies = []
    for colour, indices in server_state.get("pins", {}).items():
        for pin_id, idx in enumerate(indices):
            proxies.append(_PinProxy(pin_id, colour, idx))
    return proxies

def _pins_in_goal(server_state, colour):
    positions = server_state.get("pins", {}).get(colour, [])
    targets = _target_cells.get(colour, set())
    return sum(1 for pos in positions if pos in targets)

def _distance_to_goal(server_state, colour):
    positions = list(server_state.get("pins", {}).get(colour, []))
    return _total_dist(positions, _target_cells.get(colour, set()))

def _progress_bar(count, total=10):
    count = max(0, min(int(count), total))
    return "#" * count + "." * (total - count)

def _state_players(server_state):
    players = server_state.get("players", [])
    if players:
        return players
    return [
        {"name": colour, "colour": colour}
        for colour in server_state.get("pins", {}).keys()
    ]

def _render_terminal_dashboard(server_state, my_colour, my_name, rl_agent_desc, game_id):
    # This keeps the tournament client readable in a plain terminal without
    # needing the local Tk GUI from main.py.
    current_turn = server_state.get("current_turn_colour", "-")
    status = server_state.get("status", "-")
    move_count = server_state.get("move_count", 0)
    last_move = server_state.get("last_move")

    print("\033[2J\033[H", end="")
    print("=" * 76)
    print(f"CHINESE CHECKERS TOURNAMENT  |  Game {game_id}  |  Status {status}  |  Move {move_count}")
    print(f"You: {my_name} ({my_colour.upper()})  |  Client: {rl_agent_desc}")
    print(f"Current turn: {str(current_turn).upper()}")
    print("-" * 76)
    print("Players:")

    for player in _state_players(server_state):
        colour = str(player.get("colour", ""))
        name = str(player.get("name", colour))
        home = _pins_in_goal(server_state, colour)
        dist = _distance_to_goal(server_state, colour)
        marker = "YOU" if colour == my_colour else ""
        turn_marker = "*" if colour == current_turn else " "
        print(
            f" {turn_marker} {colour.upper():11} {name[:18]:18} "
            f"[{_progress_bar(home)}] home={home}/10 dist={dist:>3} {marker}"
        )

    if last_move:
        print("-" * 76)
        print(
            f"Last move: {last_move['by']} ({last_move['colour']}) "
            f"{last_move['from']}->{last_move['to']}  [{last_move['move_ms']:.1f}ms]"
        )

    print("-" * 76)
    print("Board:")
    _board.print_ascii(pins=_make_pins_on_board(server_state), empty=".")
    print("=" * 76)
    sys.stdout.flush()

def try_load_afterstate():
    try:
        from src.afterstate import AfterstateSearchAgent, AfterstateValueAgent
        model_path = first_existing(
            AFTERSTATE_BEST_MODEL,
            AFTERSTATE_TRAINED_MODEL,
            AFTERSTATE_FINAL_MODEL,
        )
        if os.path.exists(model_path):
            search_agent = AfterstateSearchAgent(
                state_size=STATE_SIZE,
                player_color="yellow",
                name="TournamentAfterstateSearch",
            )
            if search_agent.load_model(model_path):
                search_agent.epsilon = 0.0
                return search_agent

            agent = AfterstateValueAgent(
                state_size=STATE_SIZE,
                player_color="yellow",
                name="TournamentAfterstate",
            )
            if agent.load_model(model_path):
                agent.epsilon = 0.0
                return agent
        return None
    except Exception as e:
        print(f"Afterstate agent not available: {e}")
        return None

def try_load_alphazero():
    try:
        from src.alphazero import AlphaZeroAgent
        agent = AlphaZeroAgent(state_size=STATE_SIZE, name="TournamentAlphaZero")
        # We deliberately prefer the same "best" checkpoint used elsewhere in
        # the project so local testing and tournament play stay aligned.
        loaded = False
        for candidate in (ALPHAZERO_BEST_MODEL, ALPHAZERO_EXTERNAL_MODEL):
            if os.path.exists(candidate) and agent.load_model(str(candidate)):
                print(f"AlphaZero loaded from {os.path.basename(str(candidate))}")
                loaded = True
                break
        if not loaded:
            return None
        # The guide is still helpful in tournament mode because it sharpens the
        # root priors without replacing MCTS altogether.
        guide_path = first_existing(AFTERSTATE_BEST_MODEL, AFTERSTATE_TRAINED_MODEL, AFTERSTATE_FINAL_MODEL)
        if os.path.exists(guide_path):
            agent.enable_afterstate_guide(guide_path)
        # Tournament moves have a hard deadline, so we trim simulations a little
        # compared with heavier offline evaluation settings.
        agent.num_simulations = 80
        return agent
    except Exception as e:
        print(f"AlphaZero not available: {e}")
        return None

def main():
    print("Tournament Player starting...")
    name = input("Enter name: ").strip()
    if not name:
        return

    afterstate_agent = try_load_afterstate()
    alphazero_agent  = try_load_alphazero()

    if PREFERRED_TOURNAMENT_AGENT == "afterstate" and afterstate_agent is not None:
        rl_agent_desc = f"afterstate ({afterstate_agent.name})"
    elif alphazero_agent is not None:
        rl_agent_desc = f"AlphaZero ({alphazero_agent.name}, {alphazero_agent.num_simulations} sims)"
    elif afterstate_agent is not None:
        rl_agent_desc = f"afterstate ({afterstate_agent.name})"
    else:
        rl_agent_desc = "none (minimax fallback)"

    r = rpc({"op": "join", "player_name": name})
    if not r.get("ok"):
        print("JOIN ERROR:", r.get("error"))
        return

    game_id   = r["game_id"]
    player_id = r["player_id"]
    colour    = r["colour"]
    print(f"Joined game {game_id} as {colour}  |  RL agent: {rl_agent_desc}")

    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        if st.get("state", {}).get("status") in ("READY_TO_START", "PLAYING"):
            break
        print("Waiting for players...")
        time.sleep(0.5)

    print("Waiting for admin to start the game...")
    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        if st.get("state", {}).get("status") == "PLAYING":
            break
        time.sleep(0.5)

    print("Game started\n")

    timeoutnotice_move = -1
    recent_moves       = []
    last_dashboard_key = None

    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        if not st.get("ok"):
            print("Error:", st.get("error"))
            return

        state = st["state"]
        dashboard_key = (
            state.get("status"),
            state.get("move_count", 0),
            state.get("current_turn_colour"),
            state.get("turn_timeout_notice"),
        )
        if dashboard_key != last_dashboard_key:
            _render_terminal_dashboard(state, colour, name, rl_agent_desc, game_id)
            last_dashboard_key = dashboard_key

        if state.get("turn_timeout_notice") and timeoutnotice_move < state.get("move_count", 0):
            print("WARNING TIMEOUT:", state["turn_timeout_notice"])
            timeoutnotice_move = state.get("move_count", 0)

        if state["status"] == "FINISHED":
            print("\nGame finished")
            for pl in state["players"]:
                sc = pl.get("score")
                if sc:
                    print(f"  {pl['name']} ({pl['colour']}): "
                          f"{sc['final_score']:.1f} "
                          f"[time={sc['time_score']:.1f}, "
                          f"moves({sc['moves']})={sc['move_score']:.1f}, "
                          f"pins={sc['pin_goal_score']:.1f}, "
                          f"dist={sc['distance_score']:.1f}]")
            break

        if state.get("current_turn_colour") == colour and state["status"] == "PLAYING":
            print("\nMy turn")

            legal_req = rpc({"op": "get_legal_moves", "game_id": game_id, "player_id": player_id})
            if not legal_req.get("ok"):
                print("Error getting legal moves:", legal_req.get("error"))
                time.sleep(0.5)
                continue

            legal_moves = legal_req.get("legal_moves", {})
            if not legal_moves:
                print("No legal moves.")
                time.sleep(0.5)
                continue

            valid_actions = [(int(p), d) for p, dests in legal_moves.items() for d in dests]
            safe_actions = [
                action for action in valid_actions
                if not is_reverse_action(state, colour, action, recent_moves)
            ]
            candidate_actions = safe_actions if safe_actions else valid_actions

            filtered_legal_moves = {}
            for pin_id, dest in candidate_actions:
                filtered_legal_moves.setdefault(str(pin_id), []).append(dest)

            use_learned_agent = should_use_learned_agent(state, colour)

            # The learned agents get first chance in two-player lanes they were
            # actually trained for. If anything goes wrong, the scripted search
            # still gives us a legal fallback move.
            action = None
            if use_learned_agent and afterstate_agent is not None:
                try:
                    pins_proxy = _make_pins_on_board(state)
                    action = afterstate_agent.choose_action_from_board(
                        pins_proxy, colour, candidate_actions, _board)
                except Exception as e:
                    print(f"Afterstate error: {e}")

            if action is None and use_learned_agent and alphazero_agent is not None:
                try:
                    t0 = time.time()
                    pins_proxy = _make_pins_on_board(state)
                    action = alphazero_agent.choose_action_from_board(
                        pins_proxy, colour, candidate_actions, _board, deterministic=True)
                    print(f"  AlphaZero fallback: {time.time() - t0:.2f}s")
                except Exception as e:
                    print(f"AlphaZero error: {e}")

            if action is None:
                players = state.get("players", [])
                player_count = len(players) if players else len(state.get("pins", {}))
                fallback_depth = TWO_PLAYER_FALLBACK_DEPTH if player_count == 2 else MULTI_PLAYER_FALLBACK_DEPTH
                fallback_budget = (
                    TWO_PLAYER_FALLBACK_TIME_BUDGET
                    if player_count == 2
                    else MULTI_PLAYER_FALLBACK_TIME_BUDGET
                )
                action = pick_best_action(
                    state,
                    colour,
                    filtered_legal_moves if filtered_legal_moves else legal_moves,
                    depth=fallback_depth,
                    time_budget=fallback_budget,
                )

            # At this point the priority is simply "submit something legal".
            if action is None:
                fallback_moves = filtered_legal_moves if filtered_legal_moves else legal_moves
                pid_str = next(iter(fallback_moves))
                action  = (int(pid_str), fallback_moves[pid_str][0])

            pin_id, to_index = action
            print(f"Playing pin {pin_id} -> cell {to_index}")
            old_cell = state.get("pins", {}).get(colour, [])[int(pin_id)]

            mv = rpc({"op": "move", "game_id": game_id, "player_id": player_id,
                      "pin_id": pin_id, "to_index": to_index})

            if not mv.get("ok"):
                print("Move rejected:", mv.get("error"))
            elif mv.get("status") == "WIN":
                remember_move(recent_moves, colour, pin_id, old_cell, to_index)
                print("YOU WIN!", mv.get("msg"))
            elif mv.get("status") == "DRAW":
                remember_move(recent_moves, colour, pin_id, old_cell, to_index)
                print("DRAW", mv.get("msg"))
            else:
                remember_move(recent_moves, colour, pin_id, old_cell, to_index)

        time.sleep(0.5)

if __name__ == "__main__":
    main()
