# Simple hand-written agents used for training and testing.

import random

from src.board import HEX_DIRECTIONS

class GreedyAgent:

    def __init__(self, name="GreedyAgent"):
        self.name = name

    def choose_action(self, env, valid_actions):
        if not valid_actions:
            return None

        best_score = None
        best_actions = []
        current_player = env.get_current_player()

        # Greedy only asks one question: which move improves progress the most
        # right now?
        for action in valid_actions:
            score = env.evaluate_action_progress(action, current_player)
            if best_score is None or score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        # Several moves can look equally good to a greedy scorer. Random tie
        # breaking keeps the agent from falling into the exact same sequence.
        return random.choice(best_actions)

    def choose_action_from_board(self, pins_on_board, player_color, valid_actions, board):
        if not valid_actions:
            return None

        target_colour = board.colour_opposites.get(player_color, "")
        target_indices = set(board.axial_of_colour(target_colour))

        def axial_dist(idx1, idx2):
            c1 = board.cells[idx1]
            c2 = board.cells[idx2]
            return max(abs(c1.q - c2.q), abs(c1.r - c2.r),
                       abs((-c1.q - c1.r) - (-c2.q - c2.r)))

        def total_dist(pin_positions):
            return sum(min(axial_dist(idx, t) for t in target_indices)
                       for idx in pin_positions if target_indices)

        my_pins = [p.axialindex for p in pins_on_board if str(p.color) == str(player_color)]
        pin_map = {p.id: p.axialindex for p in pins_on_board if str(p.color) == str(player_color)}
        old_dist = total_dist(my_pins)

        best_score = None
        best_actions = []

        for action in valid_actions:
            pin_id, dest = action
            origin = pin_map.get(pin_id)
            if origin is None:
                continue

            simulated = [dest if idx == origin else idx for idx in my_pins]
            score = old_dist - total_dist(simulated)

            if dest in target_indices and origin not in target_indices:
                score += 20
            if origin in target_indices and dest not in target_indices:
                score -= 30

            if best_score is None or score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        # The board-only version uses the same tie-breaking idea as choose_action.
        return random.choice(best_actions) if best_actions else random.choice(valid_actions)

class MinimaxAgent:

    def __init__(self, name="MinimaxAgent", depth=2):
        self.name = name
        self.depth = depth
        self._dirs = HEX_DIRECTIONS

    def choose_action(self, env, valid_actions):
        if not valid_actions:
            return None

        return self.choose_action_from_board(
            env.pins_on_board,
            env.get_current_player(),
            valid_actions,
            env.board,
        )

    def _target_cells(self, colour, board):
        target_colour = board.colour_opposites.get(colour, "")
        return set(board.axial_of_colour(target_colour))

    def _axial_dist(self, idx1, idx2, board):
        c1, c2 = board.cells[idx1], board.cells[idx2]
        return max(abs(c1.q - c2.q), abs(c1.r - c2.r),
                   abs((-c1.q - c1.r) - (-c2.q - c2.r)))

    def _total_dist(self, positions, targets, board):
        if not targets or not positions:
            return 0
        return sum(min(self._axial_dist(p, t, board) for t in targets) for p in positions)

    def _evaluate(self, pos_dict, my_colour, board):
        opp_colour = board.colour_opposites.get(my_colour, "")
        my_dist = self._total_dist(pos_dict.get(my_colour, []),
                                   self._target_cells(my_colour, board), board)
        opp_dist = self._total_dist(pos_dict.get(opp_colour, []),
                                    self._target_cells(opp_colour, board), board)
        return opp_dist - my_dist

    def _get_moves(self, pos_dict, colour, board):
        # This mirrors the board's legal-move logic, but works on plain copied
        # position dictionaries so minimax can simulate ahead cheaply.
        occupied = set(idx for positions in pos_dict.values() for idx in positions)
        valid = []
        for pin_id, start_idx in enumerate(pos_dict.get(colour, [])):
            sc = board.cells[start_idx]
            q0, r0 = sc.q, sc.r
            possible = set()
            for dq, dr in self._dirs:
                ni = board.hole_index_of.get((q0 + dq, r0 + dr))
                if ni is not None and ni not in occupied:
                    possible.add(ni)
            visited, stack = {start_idx}, [start_idx]
            while stack:
                curr = stack.pop()
                cq, cr = board.cells[curr].q, board.cells[curr].r
                for dq, dr in self._dirs:
                    adj  = board.hole_index_of.get((cq + dq,     cr + dr))
                    land = board.hole_index_of.get((cq + 2 * dq, cr + 2 * dr))
                    if adj is None or land is None:
                        continue
                    if adj in occupied and land not in occupied and land not in visited:
                        possible.add(land)
                        visited.add(land)
                        stack.append(land)
            for dest in possible:
                if sc.postype != colour and board.cells[dest].postype == colour:
                    continue
                valid.append((pin_id, dest))
        return valid

    def _sim(self, pos_dict, colour, pin_id, dest):
        new = {c: list(p) for c, p in pos_dict.items()}
        new[colour][pin_id] = dest
        return new

    def _move_score(self, pos_dict, colour, pin_id, dest, board):
        targets = self._target_cells(colour, board)
        positions = pos_dict.get(colour, [])
        old_dist = self._total_dist(positions, targets, board)
        simulated = list(positions)
        simulated[pin_id] = dest
        score = old_dist - self._total_dist(simulated, targets, board)
        if dest in targets and positions[pin_id] not in targets:
            score += 20
        if positions[pin_id] in targets and dest not in targets:
            score -= 30
        return score

    def _minimax(self, pos_dict, my_colour, cur_colour, depth, maximizing, alpha, beta, board):
        opp_colour = board.colour_opposites.get(my_colour, "")
        my_pos  = pos_dict.get(my_colour, [])
        opp_pos = pos_dict.get(opp_colour, [])
        my_t  = self._target_cells(my_colour, board)
        opp_t = self._target_cells(opp_colour, board)
        if len(my_pos)  == 10 and all(p in my_t  for p in my_pos):  return  10000
        if len(opp_pos) == 10 and all(p in opp_t for p in opp_pos): return -10000
        if depth == 0:
            return self._evaluate(pos_dict, my_colour, board)
        moves = self._get_moves(pos_dict, cur_colour, board)
        if not moves:
            return self._evaluate(pos_dict, my_colour, board)
        # Ordering the most promising moves first gives alpha-beta many more
        # chances to prune bad branches early.
        moves.sort(key=lambda m: self._move_score(pos_dict, cur_colour, m[0], m[1], board),
                   reverse=maximizing)
        next_colour = opp_colour if cur_colour == my_colour else my_colour
        if maximizing:
            best = float('-inf')
            for pin_id, dest in moves:
                score = self._minimax(self._sim(pos_dict, cur_colour, pin_id, dest),
                                      my_colour, next_colour, depth - 1, False, alpha, beta, board)
                best = max(best, score)
                alpha = max(alpha, best)
                if beta <= alpha:
                    break
            return best
        else:
            best = float('inf')
            for pin_id, dest in moves:
                score = self._minimax(self._sim(pos_dict, cur_colour, pin_id, dest),
                                      my_colour, next_colour, depth - 1, True, alpha, beta, board)
                best = min(best, score)
                beta = min(beta, best)
                if beta <= alpha:
                    break
            return best

    def choose_action_from_board(self, pins_on_board, player_color, valid_actions, board):
        if not valid_actions:
            return None

        pos_dict = {}
        for pin in pins_on_board:
            pos_dict.setdefault(pin.color, [])
        for pin in pins_on_board:
            pos_dict[pin.color].append(pin.axialindex)

        opp_colour = board.colour_opposites.get(player_color, "")
        best_action = None
        best_score  = float('-inf')
        alpha, beta = float('-inf'), float('inf')

        sorted_actions = sorted(
            valid_actions,
            key=lambda m: self._move_score(pos_dict, player_color, m[0], m[1], board),
            reverse=True
        )

        for pin_id, dest in sorted_actions:
            new_state = self._sim(pos_dict, player_color, pin_id, dest)
            score = self._minimax(new_state, player_color, opp_colour,
                                  self.depth - 1, False, alpha, beta, board)
            if score > best_score:
                best_score  = score
                best_action = (pin_id, dest)
            alpha = max(alpha, best_score)

        return best_action if best_action else valid_actions[0]

class RandomAgent:

    def __init__(self, name="RandomAgent"):
        self.name = name

    def choose_action(self, env, valid_actions):
        if not valid_actions:
            return None
        return random.choice(valid_actions)

class HomeFirstRandomAgent:

    def __init__(self, name="HomeFirstRandomAgent"):
        self.name = name

    def choose_action(self, env, valid_actions):
        if not valid_actions:
            return None

        current_player = env.get_current_player()
        pin_pos = {p.id: p.axialindex for p in env.pins_on_board
                   if str(p.color) == str(current_player)}
        own_home = {idx for idx, cell in enumerate(env.board.cells)
                    if cell.postype == str(current_player)}

        # This bias makes the random teacher slightly more useful than a purely
        # uniform agent, because it at least starts the race in the right direction.
        exit_actions = [(pid, dest) for pid, dest in valid_actions
                        if pin_pos.get(pid) in own_home and dest not in own_home]

        return random.choice(exit_actions) if exit_actions else random.choice(valid_actions)
