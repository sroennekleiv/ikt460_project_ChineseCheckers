# Helpers for rotating every lane into one shared training view.

# Yellow is the reference lane the saved models were trained to understand.
REFERENCE_COLOR = "yellow"

# These are the three opposite-colour pairs that matter in two-player games.
PLAYABLE_COLOR_PAIRS = (
    ("yellow", "purple"),
    ("red", "blue"),
    ("lawn green", "gray0"),
)

# Rotations and index maps only depend on the board object and the colour pair,
# so caching them saves a lot of repeated work during training.
_rotation_cache: dict[tuple[int, str, str], int] = {}
_index_map_cache: dict[tuple[int, str, str], dict[int, int]] = {}

def rotate_axial(q, r, steps):
    q = int(q)
    r = int(r)
    for _ in range(int(steps) % 6):
        q, r = -r, q + r
    return q, r

def _colour_coords(board, colour):
    # The home triangle for one colour is enough to tell us how that lane
    # should be rotated into the shared reference view.
    return {
        (board.cells[index].q, board.cells[index].r)
        for index in board.axial_of_colour(str(colour))
    }

def rotation_steps_to_reference(board, player_color, reference_color=REFERENCE_COLOR):
    key = (id(board), str(player_color), str(reference_color))
    if key in _rotation_cache:
        return _rotation_cache[key]

    player_home = _colour_coords(board, player_color)
    reference_home = _colour_coords(board, reference_color)
    for steps in range(6):
        if {rotate_axial(q, r, steps) for q, r in player_home} == reference_home:
            _rotation_cache[key] = steps
            return steps

    _rotation_cache[key] = 0
    return 0

def index_map_to_reference(board, player_color, reference_color=REFERENCE_COLOR):
    key = (id(board), str(player_color), str(reference_color))
    if key in _index_map_cache:
        return _index_map_cache[key]

    # Once we know how many 60-degree turns a lane needs, we can translate any
    # board index from that lane into the reference lane.
    steps = rotation_steps_to_reference(board, player_color, reference_color)
    mapping: dict[int, int] = {}
    for index, cell in enumerate(board.cells):
        rotated = rotate_axial(cell.q, cell.r, steps)
        mapped = board.hole_index_of.get(rotated)
        if mapped is not None:
            mapping[index] = mapped

    _index_map_cache[key] = mapping
    return mapping

def index_to_reference_perspective(board, player_color, index, reference_color=REFERENCE_COLOR):
    mapping = index_map_to_reference(board, player_color, reference_color)
    return mapping.get(int(index), int(index))

def indices_to_reference_perspective(board, player_color, indices, reference_color=REFERENCE_COLOR):
    mapping = index_map_to_reference(board, player_color, reference_color)
    return tuple(mapping.get(int(index), int(index)) for index in indices)

def color_pair_for_game(game_index):
    # Training cycles through all supported lanes and flips the starting side
    # every full pass so one model sees both roles.
    pair = list(PLAYABLE_COLOR_PAIRS[int(game_index) % len(PLAYABLE_COLOR_PAIRS)])
    if (int(game_index) // len(PLAYABLE_COLOR_PAIRS)) % 2 == 1:
        pair.reverse()
    return pair
