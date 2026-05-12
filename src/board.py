# Board geometry and move rules for Chinese Checkers.

import math

# Axial coordinates have six natural neighbours. We keep the direction list in
# one shared constant so every agent searches the board the same way.
HEX_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

def axial_distance(board, idx1: int, idx2: int) -> int:
    c1, c2 = board.cells[idx1], board.cells[idx2]
    return max(abs(c1.q - c2.q),
               abs(c1.r - c2.r),
               abs((-c1.q - c1.r) - (-c2.q - c2.r)))

class BoardPosition:

    def __init__(self, q, r, spacing, postype='board'):
        self.q = q
        self.r = r

        # The board uses axial coordinates for rules, but the GUI needs pixel
        # coordinates. We precompute both once so rendering stays cheap.
        self.x = spacing * (math.sqrt(3) * q + math.sqrt(3) / 2 * r)
        self.y = spacing * (3 / 2 * r)

        self.postype = postype
        self.occupied = False

class HexBoard:

    def __init__(self, R=4, hole_radius=18, spacing=34):
        self.R = R
        self.hole_radius = hole_radius
        self.spacing = spacing

        # Every colour races toward the opposite home triangle. Keeping that
        # mapping on the board object makes it easy for agents and env code to
        # ask "where is this colour trying to go?"
        self.colour_opposites = {'red': 'blue', 'lawn green': 'gray0', 'blue': 'red', 'yellow': 'purple',
                                 'purple': 'yellow', 'gray0': 'lawn green'}

        self.cells = []
        self.hole_index_of = {}
        self.cartesian = []
        self._rows = []

        self._generate_hexagon()
        self._project_to_pixels()
        self._build_rows_for_ascii()

    def _generate_hexagon(self):
        R = self.R
        cells = []

        # The center of the board is a regular hexagon in axial coordinates.
        for q in range(-R, R + 1):
            for r in range(-R, R + 1):
                s = -q - r
                if max(abs(q), abs(r), abs(s)) <= R:
                    newcell = BoardPosition(q, r, self.spacing)
                    cells.append(newcell)

        # Chinese Checkers is the center hexagon plus six 10-cell home triangles.
        base_blue = [(1, -5), (2, -5), (3, -5), (4, -5), (2, -6), (3, -6), (4, -6), (3, -7), (4, -7), (4, -8)]
        base_red = [(-1, 5), (-2, 5), (-3, 5), (-4, 5), (-2, 6), (-3, 6), (-4, 6), (-3, 7), (-4, 7), (-4, 8)]
        base_yellow = [(-1, -4), (-2, -3), (-3, -2), (-4, -1), (-2, -4), (-3, -3), (-4, -2), (-3, -4), (-4, -3),
                       (-4, -4)]
        base_green = [(5, -4), (5, -3), (5, -2), (5, -1), (6, -4), (6, -3), (6, -2), (7, -4), (7, -3), (8, -4)]
        base_purple = [(1, 4), (2, 3), (3, 2), (4, 1), (2, 4), (3, 3), (4, 2), (3, 4), (4, 3), (4, 4)]
        base_gray0 = [(-5, 1), (-5, 2), (-5, 3), (-5, 4), (-6, 2), (-6, 3), (-6, 4), (-7, 3), (-7, 4), (-8, 4)]

        for (q, r) in base_blue:
            newcell = BoardPosition(q, r, self.spacing, postype='blue')
            cells.append(newcell)
        for (q, r) in base_red:
            newcell = BoardPosition(q, r, self.spacing, postype='red')
            cells.append(newcell)
        for (q, r) in base_yellow:
            newcell = BoardPosition(q, r, self.spacing, postype='yellow')
            cells.append(newcell)
        for (q, r) in base_green:
            newcell = BoardPosition(q, r, self.spacing, postype='lawn green')
            cells.append(newcell)
        for (q, r) in base_purple:
            newcell = BoardPosition(q, r, self.spacing, postype='purple')
            cells.append(newcell)
        for (q, r) in base_gray0:
            newcell = BoardPosition(q, r, self.spacing, postype='gray0')
            cells.append(newcell)

        # Stable sorting gives us stable indices, which is important because
        # the RL state vectors and saved models depend on this ordering.
        cells.sort(key=lambda t: (t.r, t.q))
        self.cells = cells
        self.hole_index_of = {(ax.q, ax.r): i for i, ax in enumerate(cells)}

    def _project_to_pixels(self):
        cart = []

        for t in self.cells:
            x = t.x
            y = t.y
            cart.append((x, y))

        self.cartesian = cart

    def _build_rows_for_ascii(self):
        rows = {}

        for t in self.cells:
            rows.setdefault(t.r, []).append((t.q, t.r, t.postype))

        # Sorting by row then column makes the ASCII board stable and readable.
        ordered = []
        for rr in sorted(rows.keys()):
            ordered.append(sorted(rows[rr], key=lambda x: x[0]))

        self._rows = ordered

    def print_ascii(self, pins=None, empty='·'):
        pin_map = {}

        if pins:
            for p in pins:
                q = self.cells[p.axialindex].q
                r = self.cells[p.axialindex].r
                pin_map[(q, r)] = (p.color[:1].upper() if p.color else 'X')

        max_width = max(len(row) for row in self._rows)

        for row in self._rows:
            pad = " " * (max_width - len(row))
            parts = []
            for (q, r, t) in row:
                parts.append(pin_map.get((q, r), empty if t == 'board' else t[:1].lower()))
            print(pad + " ".join(parts))

    def axial_index(self, q, r):
        return self.hole_index_of[(q, r)]

    def axial_of_index(self, idx):
        return self.cells[idx]

    def axial_of_colour(self, colour):
        l = [(cell.q, cell.r) for cell in self.cells if cell.postype == colour]
        return [self.hole_index_of[(q, r)] for (q, r) in l]

class Pin:

    def __init__(self, board: HexBoard, axialindex: int, id: int, color="red"):
        self.board = board
        self.axialindex = axialindex
        self.color = color
        self.id = id
        self.board.cells[axialindex].occupied = True

    @property
    def position(self):
        return self.board.cartesian[self.axialindex]

    def get_possible_moves(self):
        board = self.board
        start_hole_index = self.axialindex

        neighbour_directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

        def idx_of(q, r):
            return board.hole_index_of.get((q, r), None)

        start_cell = board.cells[start_hole_index]
        q0, r0 = start_cell.q, start_cell.r
        possible = set()

        for dq, dr in neighbour_directions:
            ni = idx_of(q0 + dq, r0 + dr)
            if ni is not None and not board.cells[ni].occupied:
                possible.add(ni)

        # Hop search is where most of the rule complexity lives. We explore all
        # reachable landing cells, but each landing cell is only expanded once
        # so we do not loop forever through the same jump cycle.
        visited = {start_hole_index}
        stack = [start_hole_index]

        while stack:
            current_cell_index = stack.pop()
            cq, cr = board.cells[current_cell_index].q, board.cells[current_cell_index].r

            for dq, dr in neighbour_directions:
                aq, ar = cq + dq, cr + dr
                bq, br = cq + 2 * dq, cr + 2 * dr

                adj_idx = idx_of(aq, ar)
                land_idx = idx_of(bq, br)

                if adj_idx is None or land_idx is None:
                    continue

                if board.cells[adj_idx].occupied and not board.cells[land_idx].occupied:
                    if land_idx not in visited:
                        possible.add(land_idx)
                        visited.add(land_idx)
                        stack.append(land_idx)

        return sorted(possible)

    def get_legal_moves(self):
        legal = []
        current_cell = self.board.cells[self.axialindex]

        for destination_cell in self.get_possible_moves():
            destination = self.board.cells[destination_cell]

            if destination.occupied:
                continue

            if current_cell.postype != self.color and destination.postype == self.color:
                continue

            legal.append(destination_cell)

        return legal

    def get_move_path(self, destination_cell):
        destination_cell = int(destination_cell)
        start_idx = int(self.axialindex)

        if destination_cell == start_idx:
            return [start_idx]

        board = self.board
        current_cell = board.cells[start_idx]
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

        def idx_of(q, r):
            return board.hole_index_of.get((q, r), None)

        def allowed(idx):
            destination = board.cells[int(idx)]
            if current_cell.postype != self.color and destination.postype == self.color:
                return False
            return True

        if destination_cell not in self.get_possible_moves():
            return [start_idx, destination_cell]

        # Single-step moves can return immediately. Only hop chains need a
        # parent map to rebuild the path.
        q0 = current_cell.q
        r0 = current_cell.r
        for dq, dr in directions:
            neighbour = idx_of(q0 + dq, r0 + dr)
            if neighbour == destination_cell and not board.cells[neighbour].occupied and allowed(neighbour):
                return [start_idx, destination_cell]

        visited = {start_idx}
        previous = {}
        stack = [start_idx]

        while stack:
            current = stack.pop()
            cell = board.cells[current]

            for dq, dr in directions:
                adjacent = idx_of(cell.q + dq, cell.r + dr)
                landing = idx_of(cell.q + 2 * dq, cell.r + 2 * dr)

                if adjacent is None or landing is None:
                    continue
                if not board.cells[adjacent].occupied or board.cells[landing].occupied:
                    continue
                if landing in visited or not allowed(landing):
                    continue

                visited.add(landing)
                previous[landing] = current

                if landing == destination_cell:
                    # Walk backwards through the parent map to rebuild the full hop chain.
                    path = [landing]
                    while path[-1] != start_idx:
                        path.append(previous[path[-1]])
                    path.reverse()
                    return path

                stack.append(landing)

        return [start_idx, destination_cell]

    def place_pin(self, destination_cell: int):
        if int(destination_cell) < 0 or int(destination_cell) >= len(self.board.cells):
            print("Pin index out of bounds for this board.")
            return False

        if int(destination_cell) not in self.get_possible_moves():
            print("Illegal move. Destination cell is not a valid move for this pin.")
            return False

        # Once a pin leaves its own starting triangle, we do not allow it to
        # retreat back in. That matches the usual tournament rule set and keeps
        # agents from gaming the score with backwards shuffling.
        if self.board.cells[self.axialindex].postype != self.color and self.board.cells[
            int(destination_cell)].postype == self.color:
            print("Cannot place pin here; Cannot move back to own home.")
            return False

        if self.board.cells[int(destination_cell)].occupied == True:
            print("Cannot place pin here; position occupied.")
            return False

        # The actual move is just three state updates: free the old cell, move
        # the pin index, then mark the new cell as occupied.
        self.board.cells[self.axialindex].occupied = False
        self.axialindex = int(destination_cell)
        self.board.cells[int(destination_cell)].occupied = True

        return True
