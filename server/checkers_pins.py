try:
    from .checkers_board import HexBoard
except ImportError:  # Allow direct execution from the server/ directory.
    from checkers_board import HexBoard

from src.board import Pin as CorePin


class Pin(CorePin):
    def __init__(self, board: HexBoard, axialindex: int, id: int, color="red"):
        super().__init__(board, axialindex, id, color=color)
