# Tk GUI used for local play and debugging.

import tkinter as tk
import time
from src.board import HexBoard

# This palette gives the local board a clear visual hierarchy: dark background,
# muted empty cells, and brighter pin colours so movement is easy to follow.
_BG             = "#0a0a18"
_PANEL_BG       = "#11112a"
_CELL_BOARD     = "#1a1a2e"
_CELL_OUTLINE   = "#28284a"
_CELL_VALID     = "#00c896"
_CELL_VALID_OUT = "#00ffbe"
_SELECTED_GLOW  = "#ffffff"
_TEXT_DIM       = "#5a6a7a"
_TEXT_MAIN      = "#dfe6e9"
_TEXT_ACCENT    = "#a29bfe"

# Active home zones stay visible enough to show lane ownership during play.
_HOME_FILL = {
    'red':        "#3b1212", 'lawn green': "#0f3b12",
    'blue':       "#0f2a3b", 'yellow':     "#3b2d0a",
    'purple':     "#26103a", 'gray0':      "#1c1c1c",
}
_HOME_OUT = {
    'red':        "#e74c3c", 'lawn green': "#2ecc71",
    'blue':       "#3498db", 'yellow':     "#f1c40f",
    'purple':     "#9b59b6", 'gray0':      "#95a5a6",
}
# Unused home zones are still drawn, but very softly, so the full board shape
# stays visible without competing for attention.
_HOME_FILL_GHOST = {k: "#121220" for k in _HOME_FILL}
_HOME_OUT_GHOST  = {k: "#1e1e30" for k in _HOME_OUT}

_PIN = {
    'red':        ("#e74c3c", "#ff9f9f", "#922b21"),
    'lawn green': ("#2ecc71", "#a3f0b8", "#1e8449"),
    'blue':       ("#3498db", "#a3d4f5", "#1a5276"),
    'yellow':     ("#f1c40f", "#fdf6b2", "#9a7d0a"),
    'purple':     ("#9b59b6", "#d7b8f3", "#6c3483"),
    'gray0':      ("#95a5a6", "#dfe6e9", "#616a6b"),
}

_LABEL = {
    'red': 'RED', 'lawn green': 'GREEN', 'blue': 'BLUE',
    'yellow': 'YELLOW', 'purple': 'PURPLE', 'gray0': 'GRAY',
}

PINS_PER_PLAYER = 10

class BoardGUI:

    def __init__(self, board: HexBoard, pins, player_roles=None):
        self.board = board
        self.pins  = pins
        self._player_roles = {
            str(color): str(role)
            for color, role in (player_roles or {}).items()
        }

        self._selected_pin   = None
        self._valid_dests    = []
        self._pending_action = None
        self._click_enabled  = False
        self._current_player = None

        # The pin list is the most reliable place to discover which colours are
        # active in this game, so we derive the panel order from it.
        seen = []
        for p in pins:
            if p.color not in seen:
                seen.append(p.color)
        self._player_colors = seen

        # Progress bars repeatedly ask "how many pins are home?". Caching each
        # colour's target cells once keeps those updates cheap.
        self._target_cache = {
            c: self._compute_targets(c) for c in self._player_colors
        }

        self.window = tk.Tk()
        self._status_text = tk.StringVar()
        self.window.title("Chinese Checkers — IKT460")
        self.window.configure(bg=_BG)
        self.window.resizable(False, False)

        top = tk.Frame(self.window, bg=_PANEL_BG, pady=10)
        top.pack(fill="x")

        tk.Label(
            top, text="◈  CHINESE CHECKERS", bg=_PANEL_BG, fg=_TEXT_ACCENT,
            font=("Helvetica", 15, "bold"), padx=16
        ).pack(side="left")

        self._turn_var = tk.StringVar(value="")
        self._turn_label = tk.Label(
            top, textvariable=self._turn_var,
            bg=_PANEL_BG, fg=_TEXT_DIM, font=("Helvetica", 10), padx=6
        )
        self._turn_label.pack(side="left")

        self._status_label = tk.Label(
            top, textvariable=self._status_text,
            bg=_PANEL_BG, fg=_TEXT_MAIN, font=("Helvetica", 11), padx=16
        )
        self._status_label.pack(side="right")
        self._status_text.set("Setting up game…")

        # The canvas and side panel live together so the board stays centered
        # while the player progress panel remains fixed on the right.
        main_frame = tk.Frame(self.window, bg=_BG)
        main_frame.pack(fill="both", expand=True)

        xs = [x for x, y in board.cartesian]
        ys = [y for x, y in board.cartesian]
        raw_w = max(xs) - min(xs)
        raw_h = max(ys) - min(ys)
        padding = 50

        screen_w = self.window.winfo_screenwidth()
        screen_h = self.window.winfo_screenheight()
        scale_w  = (screen_w - 230) / (raw_w + 2 * padding)
        scale_h  = (screen_h - 180) / (raw_h + 2 * padding)
        self.scale = min(scale_w, scale_h, 1.2)

        self.offset_x = (-min(xs) + padding) * self.scale
        self.offset_y = (-min(ys) + padding) * self.scale

        canvas_w = int((raw_w + 2 * padding) * self.scale)
        canvas_h = int((raw_h + 2 * padding) * self.scale)

        self.canvas = tk.Canvas(
            main_frame, width=canvas_w, height=canvas_h,
            bg=_BG, highlightthickness=0
        )
        self.canvas.pack(side="left", padx=(10, 0), pady=10)
        self.canvas.bind("<Button-1>", self._on_click)

        panel = tk.Frame(main_frame, bg=_PANEL_BG, width=185)
        panel.pack(side="right", fill="y", padx=10, pady=10)
        panel.pack_propagate(False)

        tk.Label(
            panel, text="PLAYERS", bg=_PANEL_BG, fg=_TEXT_ACCENT,
            font=("Helvetica", 10, "bold"), pady=10
        ).pack()

        self._bar_canvases = {}
        self._active_dots = {}

        for color in self._player_colors:
            self._build_player_row(panel, color)

        tk.Frame(panel, bg="#2a2a4a", height=1).pack(fill="x", padx=10, pady=10)

        # The legend is small, but it saves a surprising amount of guesswork
        # when you return to the GUI after training code for a while.
        tk.Label(
            panel, text="LEGEND", bg=_PANEL_BG, fg=_TEXT_ACCENT,
            font=("Helvetica", 10, "bold")
        ).pack()

        for dot_color, label in [(_CELL_VALID, "Valid move"), ("#ffffff", "Selected pin")]:
            row = tk.Frame(panel, bg=_PANEL_BG)
            row.pack(fill="x", padx=12, pady=3)
            c = tk.Canvas(row, width=12, height=12, bg=_PANEL_BG, highlightthickness=0)
            c.pack(side="left")
            c.create_oval(1, 1, 11, 11, fill=dot_color, outline="")
            tk.Label(row, text=label, bg=_PANEL_BG, fg=_TEXT_DIM,
                     font=("Helvetica", 8)).pack(side="left", padx=6)

        self.draw_board()
        self.draw_pins()
        if self._player_roles:
            self._status_text.set(self._matchup_status())

    def _build_player_row(self, parent, color):
        main_color = _PIN.get(color, ("#888", "#ccc", "#444"))[0]
        label_text = _LABEL.get(color, color.upper())
        role_text = self._player_roles.get(str(color), "")

        header = tk.Frame(parent, bg=_PANEL_BG)
        header.pack(fill="x", padx=12, pady=(6, 0))

        # The turn dot gives a quick "who moves now?" signal without forcing the
        # user to read the status text every turn.
        dot_canvas = tk.Canvas(header, width=10, height=10, bg=_PANEL_BG, highlightthickness=0)
        dot_canvas.pack(side="left")
        dot_canvas.create_oval(2, 2, 9, 9, fill=_PANEL_BG, outline=_TEXT_DIM, tags="dot")
        self._active_dots[color] = dot_canvas

        # The swatch repeats the pin colour next to the label so the side panel
        # stays readable even when several home triangles are on screen.
        swatch = tk.Canvas(header, width=13, height=13, bg=_PANEL_BG, highlightthickness=0)
        swatch.pack(side="left", padx=(4, 0))
        swatch.create_oval(1, 1, 12, 12, fill=main_color, outline="")

        tk.Label(
            header, text=label_text, bg=_PANEL_BG, fg=_TEXT_MAIN,
            font=("Helvetica", 9, "bold"), padx=6
        ).pack(side="left")
        if role_text:
            tk.Label(
                header, text=f"({role_text})", bg=_PANEL_BG, fg=_TEXT_DIM,
                font=("Helvetica", 8)
            ).pack(side="left")

        # Each row gets a 10-dot strip because every player is always trying to
        # get exactly 10 pins home.
        dot_row = tk.Canvas(parent, height=12, bg=_PANEL_BG, highlightthickness=0)
        dot_row.pack(fill="x", padx=14, pady=(3, 8))
        self._bar_canvases[color] = (dot_row, main_color)
        dot_row.bind("<Configure>", lambda e, c=color: self._draw_bar(c))

    def _compute_targets(self, color):
        try:
            opp = self.board.colour_opposites.get(str(color), "")
            return set(self.board.axial_of_colour(opp)) if opp else set()
        except Exception:
            return set()

    def _pins_home(self, color):
        targets = self._target_cache.get(color, set())
        return sum(1 for p in self.pins if p.color == color and p.axialindex in targets)

    def _draw_bar(self, color):
        canvas, main_color = self._bar_canvases[color]
        n = self._pins_home(color)
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        canvas.delete("all")
        if w < 10:
            return
        dot_d  = max(6, min(h - 2, (w - 2) // PINS_PER_PLAYER - 2))
        spacing = (w - 2) / PINS_PER_PLAYER
        cy = h // 2
        for i in range(PINS_PER_PLAYER):
            cx = int(spacing * i + spacing / 2)
            if i < n:
                canvas.create_oval(cx - dot_d//2, cy - dot_d//2,
                                   cx + dot_d//2, cy + dot_d//2,
                                   fill=main_color, outline="")
            else:
                canvas.create_oval(cx - dot_d//2, cy - dot_d//2,
                                   cx + dot_d//2, cy + dot_d//2,
                                   fill="", outline=_TEXT_DIM, width=1)

    def _update_progress(self):
        for color in self._player_colors:
            self._draw_bar(color)

    def _update_active_dot(self, active_color):
        for color, dot in self._active_dots.items():
            main_color = _PIN.get(color, ("#888",))[0]
            dot.delete("dot")
            if color == active_color:
                dot.create_oval(2, 2, 9, 9, fill=main_color, outline="", tags="dot")
            else:
                dot.create_oval(2, 2, 9, 9, fill=_PANEL_BG, outline=_TEXT_DIM, tags="dot")

    def _matchup_status(self):
        parts = []
        for color in self._player_colors:
            role = self._player_roles.get(str(color), "")
            if role:
                parts.append(f"{_LABEL.get(color, color.upper())}={role}")
        return " | ".join(parts) if parts else "Setting up game…"

    def _to_canvas(self, x, y):
        return x * self.scale + self.offset_x, y * self.scale + self.offset_y

    def _hole_r(self):
        return max(7, int(self.board.hole_radius * self.scale))

    def draw_board(self):
        r = self._hole_r()
        active_zones = set(self._player_colors)
        # We also keep the corresponding target zones visible, because they help
        # explain where each colour is trying to finish.
        for c in self._player_colors:
            opp = self.board.colour_opposites.get(c, "")
            if opp:
                active_zones.add(opp)

        for idx, cell in enumerate(self.board.cells):
            cx, cy = self._to_canvas(cell.x, cell.y)

            if idx in self._valid_dests:
                self.canvas.create_oval(
                    cx - r - 4, cy - r - 4, cx + r + 4, cy + r + 4,
                    fill="", outline=_CELL_VALID_OUT, width=2
                )
                fill, outline, lw = _CELL_VALID, _CELL_VALID_OUT, 2
            elif cell.postype == 'board':
                fill, outline, lw = _CELL_BOARD, _CELL_OUTLINE, 1
            else:
                ghost = cell.postype not in active_zones
                fill    = (_HOME_FILL_GHOST if ghost else _HOME_FILL).get(cell.postype, _CELL_BOARD)
                outline = (_HOME_OUT_GHOST  if ghost else _HOME_OUT ).get(cell.postype, _CELL_OUTLINE)
                lw = 1 if ghost else 2

            self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                fill=fill, outline=outline, width=lw
            )

    def _draw_marble(self, cx, cy, r, color, selected=False):
        main, light, dark = _PIN.get(color, ("#888", "#ccc", "#444"))

        if selected:
            for offset in (9, 6):
                self.canvas.create_oval(
                    cx - r - offset, cy - r - offset,
                    cx + r + offset, cy + r + offset,
                    fill="", outline=_SELECTED_GLOW,
                    width=1 if offset == 9 else 2
                )

        # The layered circles are simple, but they make the pins read more like
        # marbles than flat discs, which improves move visibility.
        self.canvas.create_oval(
            cx - r + 3, cy - r + 3, cx + r + 3, cy + r + 3,
            fill=dark, outline=""
        )
        self.canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            fill=main, outline=dark, width=1
        )
        gr = max(3, r // 3)
        gx, gy = cx - r * 0.28, cy - r * 0.28
        self.canvas.create_oval(
            gx - gr, gy - gr, gx + gr, gy + gr,
            fill=light, outline=""
        )
        sr = max(1, r // 6)
        self.canvas.create_oval(
            gx - sr + 1, gy - sr + 1, gx + sr + 1, gy + sr + 1,
            fill="#ffffff", outline=""
        )

    def draw_pins(self, override_positions=None):
        pr = max(5, int(self.board.hole_radius * 0.76 * self.scale))
        for pin in self.pins:
            key = (pin.color, pin.id)
            if override_positions and key in override_positions:
                x, y = override_positions[key]
            else:
                x, y = self.board.cartesian[int(pin.axialindex)]
            cx, cy = self._to_canvas(x, y)
            is_sel = (
                self._selected_pin is not None
                and pin.id    == self._selected_pin.id
                and pin.color == self._selected_pin.color
            )
            self._draw_marble(cx, cy, pr, pin.color, selected=is_sel)

    def refresh(self, newpins, status_msg=None, override_positions=None):
        self.canvas.delete("all")
        self.pins = newpins
        self.draw_board()
        self.draw_pins(override_positions=override_positions)
        self._update_progress()
        if status_msg:
            self._status_text.set(status_msg)

    def animate_move(self, newpins, pin_id, pin_color, move_path, status_msg=None):
        if not move_path or len(move_path) < 2:
            self.refresh(newpins, status_msg=status_msg)
            self.window.update()
            return

        self.pins = newpins
        points = [self.board.cartesian[int(idx)] for idx in move_path]
        frames_per_segment = 10
        frame_delay = 0.016

        for start, end in zip(points, points[1:]):
            sx, sy = start
            ex, ey = end
            for step in range(1, frames_per_segment + 1):
                t = step / frames_per_segment
                # Smooth-step keeps the animation from looking robotic at the
                # start and end of each segment.
                t = t * t * (3 - 2 * t)
                self.refresh(
                    newpins,
                    status_msg=status_msg,
                    override_positions={(pin_color, pin_id): (sx + (ex - sx) * t, sy + (ey - sy) * t)},
                )
                self.window.update()
                time.sleep(frame_delay)

        self.refresh(newpins, status_msg=status_msg)
        self.window.update()

    def show_winner(self, player_color):
        main_color = _PIN.get(player_color, ("#888", "#ccc", "#444"))[0]
        label = _LABEL.get(player_color, player_color.upper())
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        bx, by = cw // 2, ch // 2

        # The winner banner deliberately dims the board so the result is obvious
        # even if the final move ended in a crowded area.
        self.canvas.create_rectangle(
            0, 0, cw, ch, fill="#0a0a18", stipple="gray50", outline=""
        )
        self.canvas.create_rectangle(
            bx - 198, by - 78, bx + 202, by + 72,
            fill="#000000", outline=""
        )
        self.canvas.create_rectangle(
            bx - 200, by - 80, bx + 200, by + 70,
            fill="#12122a", outline=main_color, width=3
        )
        self.canvas.create_text(
            bx, by - 35, text="WINNER", fill=main_color,
            font=("Helvetica", 28, "bold")
        )
        self.canvas.create_text(
            bx, by + 15, text=label, fill="#ffffff",
            font=("Helvetica", 18, "bold")
        )
        self._status_text.set(f"🏆  {label} wins!")
        self._update_active_dot(player_color)
        self.window.update()

    def set_status(self, text):
        self._status_text.set(text)

    def set_turn(self, turn):
        self._turn_var.set(f"Turn {turn}")

    def _redraw(self):
        self.canvas.delete("all")
        self.draw_board()
        self.draw_pins()

    def enable_click(self, current_player):
        self._click_enabled  = True
        self._current_player = current_player
        self._selected_pin   = None
        self._valid_dests    = []
        self._pending_action = None
        label = _LABEL.get(current_player, current_player.upper())
        self._status_text.set(f"◉  Your turn — {label}")
        self._update_active_dot(current_player)

    def disable_click(self):
        self._click_enabled = False
        self._selected_pin  = None
        self._valid_dests   = []
        self._redraw()

    def wait_for_click_action(self):
        self._pending_action = None
        while self._pending_action is None:
            self.window.update()
        return self._pending_action

    def _on_click(self, event):
        if not self._click_enabled:
            return

        r = self._hole_r()
        clicked_cell = None
        for idx, cell in enumerate(self.board.cells):
            ccx, ccy = self._to_canvas(cell.x, cell.y)
            if (event.x - ccx) ** 2 + (event.y - ccy) ** 2 <= r ** 2:
                clicked_cell = idx
                break

        if clicked_cell is None:
            return

        # First click selects one of the current player's pins and highlights
        # every legal destination the board rules allow from there.
        if self._selected_pin is None:
            for pin in self.pins:
                if pin.color == self._current_player and pin.axialindex == clicked_cell:
                    self._selected_pin = pin
                    self._valid_dests = list(pin.get_legal_moves())
                    self._redraw()
                    n = len(self._valid_dests)
                    self._status_text.set(
                        f"Pin {pin.id} selected — {n} move{'s' if n != 1 else ''} available"
                    )
                    return
            return

        # Clicking a highlighted destination turns the GUI selection into an
        # actual action tuple for the game loop.
        if clicked_cell in self._valid_dests:
            pid = self._selected_pin.id
            self._selected_pin   = None
            self._valid_dests    = []
            self._click_enabled  = False
            self._pending_action = (pid, clicked_cell)
            self._redraw()
            return

        # Clicking another friendly pin switches focus instead of forcing the
        # user to deselect first.
        for pin in self.pins:
            if pin.color == self._current_player and pin.axialindex == clicked_cell:
                self._selected_pin = pin
                self._valid_dests = list(pin.get_legal_moves())
                self._redraw()
                n = len(self._valid_dests)
                self._status_text.set(
                    f"Pin {pin.id} selected — {n} move{'s' if n != 1 else ''} available"
                )
                return

        # Any other click simply clears the current selection.
        self._selected_pin = None
        self._valid_dests  = []
        self._redraw()
        label = _LABEL.get(self._current_player, self._current_player.upper())
        self._status_text.set(f"◉  Your turn — {label}")

    def run(self):
        self.window.mainloop()
