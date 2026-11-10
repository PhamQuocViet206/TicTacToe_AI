"""Tkinter GUI for the TicTacToe / Caro game with modern, minimalist design."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

try:
    from .ai_controller import AIController
    from .constants import BOARD_SIZE, PLAYER_O, PLAYER_X
    from .game import TicTacToeGame
except ImportError:  # pragma: no cover
    from ai_controller import AIController
    from constants import BOARD_SIZE, PLAYER_O, PLAYER_X
    from game import TicTacToeGame


# Design Palette & Fonts
COLOR_BG = "#f8fafc"            # App background (Slate 50)
COLOR_PANEL_BG = "#ffffff"      # Card background
COLOR_BORDER = "#e2e8f0"        # Card border (Slate 200)
COLOR_TEXT_MAIN = "#0f172a"     # Main text (Slate 900)
COLOR_TEXT_MUTED = "#64748b"    # Secondary text (Slate 500)

COLOR_BOARD_BG = "#faf8f5"      # Board canvas warm ivory
COLOR_GRID_LINE = "#cbd5e1"     # Board inner lines (Slate 300)
COLOR_GRID_BORDER = "#94a3b8"   # Board outer boundary (Slate 400)
COLOR_STAR_POINT = "#94a3b8"    # Hoshi star points

COLOR_X = "#e11d48"             # Player X vivid Crimson (Rose 600)
COLOR_X_SHADOW = "#fecdd3"      # Player X soft shadow (Rose 200)
COLOR_X_GHOST = "#fda4af"       # Player X hover preview

COLOR_O = "#0284c7"             # Player O vivid Cobalt (Sky 600)
COLOR_O_SHADOW = "#bae6fd"      # Player O soft shadow (Sky 200)
COLOR_O_GHOST = "#7dd3fc"       # Player O hover preview

COLOR_LAST_MOVE = "#f59e0b"     # Amber 500 for last move ring
COLOR_WIN_GLOW = "#dcfce7"      # Victory stone aura (Green 100)
COLOR_WIN_BORDER = "#22c55e"    # Victory stone ring (Green 500)
COLOR_WIN_LINE = "#16a34a"      # Victory strike line (Green 600)

FONT_FAMILY = "Segoe UI"


class TicTacToeGUI:
    """Graphical interface for playing TicTacToe locally or against an AI."""

    def __init__(self, root: tk.Tk | None = None):
        self.root = root if root is not None else tk.Tk()
        self.root.title("TicTacToe AI - Cờ Caro 15x15")
        self.root.geometry("1060x690")
        self.root.resizable(False, False)
        self.root.configure(bg=COLOR_BG)

        self.game = TicTacToeGame(board_size=BOARD_SIZE)
        self.history = [self.game.copy()]
        self.cell_size = 36
        self.margin = 32
        self.ai_busy = False
        self.ai_player = None
        self.batch_active = False
        self.batch_target = 0
        self.batch_completed = 0
        self.batch_x_wins = 0
        self.batch_o_wins = 0
        self.batch_draws = 0
        self._scheduled_callback = None
        self.hover_cell: tuple[int, int] | None = None
        self.advanced_expanded = False

        # State Variables
        self.mode_var = tk.StringVar(value="Human vs Human")
        self.ai_var = tk.StringVar(value="random")
        self.ai_o_var = tk.StringVar(value="random")
        self.rollout_var = tk.StringVar(value="random")
        self.first_player_var = tk.StringVar(value="X")
        self.ai_iterations_var = tk.StringVar(value="1000")
        self.uct_var = tk.StringVar(value="1.40")
        self.batch_games_var = tk.StringVar(value="10")
        self.status_var = tk.StringVar(value="Game ready.")
        self.turn_var = tk.StringVar(value="Current turn: X")
        self.batch_stats_var = tk.StringVar(value="AI match results: no games played")
        self.move_count_var = tk.StringVar(value="Nước đi: 0")

        self._configure_styles()
        self._build_ui()
        self._reset_game()

    def _configure_styles(self) -> None:
        """Configure ttk widget styles for modern minimalism."""
        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        self.style.configure(
            "Clean.TCombobox",
            fieldbackground="#ffffff",
            background="#f8fafc",
            foreground=COLOR_TEXT_MAIN,
            arrowcolor=COLOR_TEXT_MUTED,
            bordercolor=COLOR_BORDER,
            lightcolor=COLOR_BORDER,
            darkcolor=COLOR_BORDER,
            padding=4,
            font=(FONT_FAMILY, 10),
        )
        self.style.map(
            "Clean.TCombobox",
            fieldbackground=[("readonly", "#ffffff")],
            selectbackground=[("readonly", "#ffffff")],
            selectforeground=[("readonly", COLOR_TEXT_MAIN)],
        )

        self.style.configure(
            "Clean.TEntry",
            fieldbackground="#ffffff",
            foreground=COLOR_TEXT_MAIN,
            bordercolor=COLOR_BORDER,
            lightcolor=COLOR_BORDER,
            darkcolor=COLOR_BORDER,
            padding=4,
            font=(FONT_FAMILY, 10),
        )

    def _build_ui(self) -> None:
        """Construct the visual, minimalist two-column layout."""
        main_container = tk.Frame(self.root, bg=COLOR_BG, padx=20, pady=16)
        main_container.pack(fill=tk.BOTH, expand=True)

        # ----------------- Left: Game Board Panel -----------------
        board_panel = tk.Frame(
            main_container,
            bg=COLOR_PANEL_BG,
            highlightbackground=COLOR_BORDER,
            highlightthickness=1,
            padx=12,
            pady=12,
        )
        board_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        canvas_dim = self.margin * 2 + self.cell_size * self.game.board_size
        self.canvas = tk.Canvas(
            board_panel,
            width=canvas_dim,
            height=canvas_dim,
            bg=COLOR_BOARD_BG,
            highlightthickness=1,
            highlightbackground=COLOR_BORDER,
        )
        self.canvas.pack()

        # Canvas Interactions
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", self._on_canvas_leave)

        # ----------------- Right: Control & Info Sidebar -----------------
        sidebar = tk.Frame(main_container, bg=COLOR_BG, padx=14, pady=0)
        sidebar.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 1. Header & Title Card
        title_frame = tk.Frame(sidebar, bg=COLOR_BG)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        title_lbl = tk.Label(
            title_frame,
            text="CARO AI",
            font=(FONT_FAMILY, 16, "bold"),
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_BG,
        )
        title_lbl.pack(side=tk.LEFT)

        sub_lbl = tk.Label(
            title_frame,
            text="15 × 15 • Luật 5 quân",
            font=(FONT_FAMILY, 9),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_BG,
        )
        sub_lbl.pack(side=tk.LEFT, padx=(10, 0), pady=(4, 0))

        # 2. Status & Turn Card
        self.status_card = tk.Frame(
            sidebar,
            bg=COLOR_PANEL_BG,
            highlightbackground=COLOR_BORDER,
            highlightthickness=1,
            padx=14,
            pady=10,
        )
        self.status_card.pack(fill=tk.X, pady=(0, 10))

        # Status row: Player Badge + Turn Text
        status_row = tk.Frame(self.status_card, bg=COLOR_PANEL_BG)
        status_row.pack(fill=tk.X)

        self.turn_badge = tk.Label(
            status_row,
            text="X",
            font=(FONT_FAMILY, 11, "bold"),
            fg="#ffffff",
            bg=COLOR_X,
            width=3,
            pady=2,
        )
        self.turn_badge.pack(side=tk.LEFT, padx=(0, 10))

        turn_info_frame = tk.Frame(status_row, bg=COLOR_PANEL_BG)
        turn_info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.turn_display_lbl = tk.Label(
            turn_info_frame,
            textvariable=self.turn_var,
            font=(FONT_FAMILY, 11, "bold"),
            fg=COLOR_TEXT_MAIN,
            anchor="w",
        )
        self.turn_display_lbl.pack(fill=tk.X)

        self.status_display_lbl = tk.Label(
            turn_info_frame,
            textvariable=self.status_var,
            font=(FONT_FAMILY, 9),
            fg=COLOR_TEXT_MUTED,
            anchor="w",
            wraplength=220,
            justify=tk.LEFT,
        )
        self.status_display_lbl.pack(fill=tk.X, pady=(2, 0))

        # Pill for move count
        self.move_count_lbl = tk.Label(
            status_row,
            textvariable=self.move_count_var,
            font=(FONT_FAMILY, 9, "bold"),
            fg=COLOR_TEXT_MUTED,
            bg="#f1f5f9",
            padx=8,
            pady=3,
        )
        self.move_count_lbl.pack(side=tk.RIGHT)

        # 3. Settings Card (Context-aware & Minimalist)
        self.settings_card = tk.Frame(
            sidebar,
            bg=COLOR_PANEL_BG,
            highlightbackground=COLOR_BORDER,
            highlightthickness=1,
            padx=14,
            pady=12,
        )
        self.settings_card.pack(fill=tk.X, pady=(0, 10))

        # Mode Selection
        tk.Label(
            self.settings_card,
            text="Chế độ chơi (Game Mode)",
            font=(FONT_FAMILY, 9, "bold"),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_PANEL_BG,
        ).pack(anchor="w", pady=(0, 4))

        self.mode_combo = ttk.Combobox(
            self.settings_card,
            textvariable=self.mode_var,
            values=["Human vs Human", "Human vs AI", "AI vs AI"],
            state="readonly",
            style="Clean.TCombobox",
        )
        self.mode_combo.pack(fill=tk.X, pady=(0, 8))

        # Dynamic Section Container
        self.dynamic_frame = tk.Frame(self.settings_card, bg=COLOR_PANEL_BG)
        self.dynamic_frame.pack(fill=tk.X)

        # Advanced MCTS Toggle Button
        self.adv_toggle_btn = tk.Button(
            self.settings_card,
            text="⚙ Cài đặt AI nâng cao  ▾",
            font=(FONT_FAMILY, 9),
            fg="#2563eb",
            bg=COLOR_PANEL_BG,
            activebackground=COLOR_PANEL_BG,
            activeforeground="#1d4ed8",
            bd=0,
            cursor="hand2",
            anchor="w",
            command=self._toggle_advanced_settings,
        )

        # Advanced Settings Frame (Collapsed by default)
        self.advanced_frame = tk.Frame(self.settings_card, bg="#f8fafc", padx=8, pady=8)

        # 4. Action Buttons Card
        action_card = tk.Frame(
            sidebar,
            bg=COLOR_PANEL_BG,
            highlightbackground=COLOR_BORDER,
            highlightthickness=1,
            padx=14,
            pady=12,
        )
        action_card.pack(fill=tk.X, pady=(0, 10))

        # Primary Button: New Game
        new_game_btn = tk.Button(
            action_card,
            text="🎮  Ván mới (New Game)",
            command=self._reset_game,
            font=(FONT_FAMILY, 10, "bold"),
            fg="#ffffff",
            bg="#2563eb",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            cursor="hand2",
            pady=7,
        )
        new_game_btn.pack(fill=tk.X, pady=(0, 6))

        # Secondary Actions Row: Undo & Restart
        btn_row = tk.Frame(action_card, bg=COLOR_PANEL_BG)
        btn_row.pack(fill=tk.X)

        undo_btn = tk.Button(
            btn_row,
            text="↩ Đi lại (Undo)",
            command=self.undo_move,
            font=(FONT_FAMILY, 9),
            fg=COLOR_TEXT_MAIN,
            bg="#f1f5f9",
            activebackground="#e2e8f0",
            relief="flat",
            bd=0,
            cursor="hand2",
            pady=5,
        )
        undo_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        restart_btn = tk.Button(
            btn_row,
            text="🔄 Chơi lại",
            command=self.restart_game,
            font=(FONT_FAMILY, 9),
            fg=COLOR_TEXT_MAIN,
            bg="#f1f5f9",
            activebackground="#e2e8f0",
            relief="flat",
            bd=0,
            cursor="hand2",
            pady=5,
        )
        restart_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

        # 5. AI Match Stats Card
        self.stats_card = tk.Frame(
            sidebar,
            bg=COLOR_PANEL_BG,
            highlightbackground=COLOR_BORDER,
            highlightthickness=1,
            padx=14,
            pady=10,
        )
        # Packed conditionally or kept visible cleanly

        stats_title_row = tk.Frame(self.stats_card, bg=COLOR_PANEL_BG)
        stats_title_row.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            stats_title_row,
            text="Thống kê đấu AI",
            font=(FONT_FAMILY, 9, "bold"),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_PANEL_BG,
        ).pack(side=tk.LEFT)

        # 3 Score Badges: X Wins | O Wins | Draws
        badges_row = tk.Frame(self.stats_card, bg=COLOR_PANEL_BG)
        badges_row.pack(fill=tk.X, pady=(0, 6))

        self.badge_x = tk.Label(
            badges_row,
            text="X: 0",
            font=(FONT_FAMILY, 9, "bold"),
            fg=COLOR_X,
            bg="#fee2e2",
            padx=8,
            pady=3,
        )
        self.badge_x.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self.badge_draw = tk.Label(
            badges_row,
            text="Hòa: 0",
            font=(FONT_FAMILY, 9, "bold"),
            fg=COLOR_TEXT_MUTED,
            bg="#f1f5f9",
            padx=8,
            pady=3,
        )
        self.badge_draw.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.badge_o = tk.Label(
            badges_row,
            text="O: 0",
            font=(FONT_FAMILY, 9, "bold"),
            fg=COLOR_O,
            bg="#e0f2fe",
            padx=8,
            pady=3,
        )
        self.badge_o.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        self.batch_stats_lbl = tk.Label(
            self.stats_card,
            textvariable=self.batch_stats_var,
            font=(FONT_FAMILY, 8),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_PANEL_BG,
            wraplength=280,
            justify=tk.LEFT,
        )
        self.batch_stats_lbl.pack(anchor="w")

        # Watchers
        self.mode_var.trace_add("write", lambda *_: self._update_ai_settings())
        self.ai_var.trace_add("write", lambda *_: self._update_ai_settings())
        self.ai_o_var.trace_add("write", lambda *_: self._update_ai_settings())
        self.first_player_var.trace_add("write", lambda *_: self._update_ai_settings())

        self._refresh_dynamic_controls()

    def _refresh_dynamic_controls(self) -> None:
        """Render context-specific controls based on current mode."""
        for widget in self.dynamic_frame.winfo_children():
            widget.destroy()

        mode = self.mode_var.get()

        if mode == "Human vs Human":
            # Only First Player matters
            tk.Label(
                self.dynamic_frame,
                text="Quân đi trước",
                font=(FONT_FAMILY, 9, "bold"),
                fg=COLOR_TEXT_MUTED,
                bg=COLOR_PANEL_BG,
            ).pack(anchor="w", pady=(0, 2))
            first_combo = ttk.Combobox(
                self.dynamic_frame,
                textvariable=self.first_player_var,
                values=["X", "O"],
                state="readonly",
                style="Clean.TCombobox",
            )
            first_combo.pack(fill=tk.X, pady=(0, 6))

            self.adv_toggle_btn.pack_forget()
            self.advanced_frame.pack_forget()
            self.stats_card.pack_forget()

        elif mode == "Human vs AI":
            # AI Algorithm + First Player (Human X or O)
            row = tk.Frame(self.dynamic_frame, bg=COLOR_PANEL_BG)
            row.pack(fill=tk.X, pady=(0, 6))

            col1 = tk.Frame(row, bg=COLOR_PANEL_BG)
            col1.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
            tk.Label(
                col1,
                text="Thuật toán AI",
                font=(FONT_FAMILY, 9, "bold"),
                fg=COLOR_TEXT_MUTED,
                bg=COLOR_PANEL_BG,
            ).pack(anchor="w", pady=(0, 2))
            ai_combo = ttk.Combobox(
                col1,
                textvariable=self.ai_var,
                values=["random", "minimax", "mcts"],
                state="readonly",
                style="Clean.TCombobox",
            )
            ai_combo.pack(fill=tk.X)

            col2 = tk.Frame(row, bg=COLOR_PANEL_BG)
            col2.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))
            tk.Label(
                col2,
                text="Đi trước",
                font=(FONT_FAMILY, 9, "bold"),
                fg=COLOR_TEXT_MUTED,
                bg=COLOR_PANEL_BG,
            ).pack(anchor="w", pady=(0, 2))
            first_combo = ttk.Combobox(
                col2,
                textvariable=self.first_player_var,
                values=["X", "O"],
                state="readonly",
                style="Clean.TCombobox",
            )
            first_combo.pack(fill=tk.X)

            # Advanced MCTS Toggle
            self.adv_toggle_btn.pack(fill=tk.X, pady=(4, 2))
            self._build_advanced_inputs()
            if self.advanced_expanded:
                self.advanced_frame.pack(fill=tk.X, pady=(4, 6))
            else:
                self.advanced_frame.pack_forget()

            self.stats_card.pack_forget()

        elif mode == "AI vs AI":
            # AI X vs AI O
            row = tk.Frame(self.dynamic_frame, bg=COLOR_PANEL_BG)
            row.pack(fill=tk.X, pady=(0, 6))

            col1 = tk.Frame(row, bg=COLOR_PANEL_BG)
            col1.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
            tk.Label(
                col1,
                text="Thuật toán X",
                font=(FONT_FAMILY, 9, "bold"),
                fg=COLOR_TEXT_MUTED,
                bg=COLOR_PANEL_BG,
            ).pack(anchor="w", pady=(0, 2))
            ai_x_combo = ttk.Combobox(
                col1,
                textvariable=self.ai_var,
                values=["random", "minimax", "mcts"],
                state="readonly",
                style="Clean.TCombobox",
            )
            ai_x_combo.pack(fill=tk.X)

            col2 = tk.Frame(row, bg=COLOR_PANEL_BG)
            col2.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))
            tk.Label(
                col2,
                text="Thuật toán O",
                font=(FONT_FAMILY, 9, "bold"),
                fg=COLOR_TEXT_MUTED,
                bg=COLOR_PANEL_BG,
            ).pack(anchor="w", pady=(0, 2))
            ai_o_combo = ttk.Combobox(
                col2,
                textvariable=self.ai_o_var,
                values=["random", "minimax", "mcts"],
                state="readonly",
                style="Clean.TCombobox",
            )
            ai_o_combo.pack(fill=tk.X)

            # Match Count & Action Buttons
            batch_row = tk.Frame(self.dynamic_frame, bg=COLOR_PANEL_BG)
            batch_row.pack(fill=tk.X, pady=(0, 6))

            batch_col = tk.Frame(batch_row, bg=COLOR_PANEL_BG)
            batch_col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
            tk.Label(
                batch_col,
                text="Số ván đấu",
                font=(FONT_FAMILY, 9, "bold"),
                fg=COLOR_TEXT_MUTED,
                bg=COLOR_PANEL_BG,
            ).pack(anchor="w", pady=(0, 2))
            ttk.Entry(
                batch_col,
                textvariable=self.batch_games_var,
                style="Clean.TEntry",
            ).pack(fill=tk.X)

            # Batch Buttons
            ai_btn_col = tk.Frame(batch_row, bg=COLOR_PANEL_BG)
            ai_btn_col.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))
            tk.Label(ai_btn_col, text=" ", font=(FONT_FAMILY, 9), bg=COLOR_PANEL_BG).pack(anchor="w", pady=(0, 2))

            sub_btn_row = tk.Frame(ai_btn_col, bg=COLOR_PANEL_BG)
            sub_btn_row.pack(fill=tk.X)

            start_ai_btn = tk.Button(
                sub_btn_row,
                text="▶ Bắt đầu",
                command=self.start_ai_matches,
                font=(FONT_FAMILY, 9, "bold"),
                fg="#ffffff",
                bg="#10b981",
                activebackground="#059669",
                relief="flat",
                bd=0,
                cursor="hand2",
                pady=4,
            )
            start_ai_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

            stop_ai_btn = tk.Button(
                sub_btn_row,
                text="⏹ Dừng",
                command=self.stop_ai_matches,
                font=(FONT_FAMILY, 9, "bold"),
                fg="#ffffff",
                bg="#ef4444",
                activebackground="#dc2626",
                relief="flat",
                bd=0,
                cursor="hand2",
                pady=4,
            )
            stop_ai_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

            # Advanced MCTS Toggle
            self.adv_toggle_btn.pack(fill=tk.X, pady=(4, 2))
            self._build_advanced_inputs()
            if self.advanced_expanded:
                self.advanced_frame.pack(fill=tk.X, pady=(4, 6))
            else:
                self.advanced_frame.pack_forget()

            # Show AI Stats Card
            self.stats_card.pack(fill=tk.X, pady=(0, 10))

    def _build_advanced_inputs(self) -> None:
        """Construct entries for advanced MCTS parameters inside advanced_frame."""
        for w in self.advanced_frame.winfo_children():
            w.destroy()

        r1 = tk.Frame(self.advanced_frame, bg="#f8fafc")
        r1.pack(fill=tk.X, pady=2)
        tk.Label(r1, text="Số lần lặp MCTS:", font=(FONT_FAMILY, 8), bg="#f8fafc", fg=COLOR_TEXT_MUTED).pack(side=tk.LEFT)
        ttk.Entry(r1, textvariable=self.ai_iterations_var, width=8, style="Clean.TEntry").pack(side=tk.RIGHT)

        r2 = tk.Frame(self.advanced_frame, bg="#f8fafc")
        r2.pack(fill=tk.X, pady=2)
        tk.Label(r2, text="Hệ số UCT (c):", font=(FONT_FAMILY, 8), bg="#f8fafc", fg=COLOR_TEXT_MUTED).pack(side=tk.LEFT)
        ttk.Entry(r2, textvariable=self.uct_var, width=8, style="Clean.TEntry").pack(side=tk.RIGHT)

        r3 = tk.Frame(self.advanced_frame, bg="#f8fafc")
        r3.pack(fill=tk.X, pady=2)
        tk.Label(r3, text="Rollout Policy:", font=(FONT_FAMILY, 8), bg="#f8fafc", fg=COLOR_TEXT_MUTED).pack(side=tk.LEFT)
        ttk.Combobox(
            r3,
            textvariable=self.rollout_var,
            values=["random", "heuristic"],
            state="readonly",
            width=10,
            style="Clean.TCombobox",
        ).pack(side=tk.RIGHT)

    def _toggle_advanced_settings(self) -> None:
        """Expand or collapse the advanced MCTS configuration frame."""
        self.advanced_expanded = not self.advanced_expanded
        if self.advanced_expanded:
            self.adv_toggle_btn.configure(text="⚙ Cài đặt AI nâng cao  ▴")
            self.advanced_frame.pack(fill=tk.X, pady=(4, 6))
        else:
            self.adv_toggle_btn.configure(text="⚙ Cài đặt AI nâng cao  ▾")
            self.advanced_frame.pack_forget()

    def _update_ai_settings(self) -> None:
        """Refresh AI assignments and dynamic UI controls after any mode change."""
        self._refresh_dynamic_controls()
        if self.batch_active:
            self._stop_ai_matches()
            self._cancel_scheduled_callback()
        self.ai_player = self._ai_player_for_current_mode()
        if self.mode_var.get() == "Human vs AI" and self._is_ai_turn():
            self.status_var.set("AI is thinking...")
            self._schedule_ai_turn()
        else:
            self._update_status_message()

    def _ai_player_for_current_mode(self):
        if self.mode_var.get() != "Human vs AI":
            return None
        if self.first_player_var.get() == "X":
            return PLAYER_O
        return PLAYER_X

    def _reset_game(self) -> None:
        self._stop_ai_matches()
        self._cancel_scheduled_callback()
        self._new_game_state()
        self._update_status_message()

    def _new_game_state(self) -> None:
        self.game = TicTacToeGame(board_size=BOARD_SIZE)
        self.game.current_player = PLAYER_X if self.first_player_var.get() == "X" else PLAYER_O
        self.history = [self.game.copy()]
        self.ai_busy = False
        self.ai_player = self._ai_player_for_current_mode()
        self.hover_cell = None
        self._draw_board()

    def restart_game(self) -> None:
        self._reset_game()

    def undo_move(self) -> None:
        if self.batch_active:
            self._stop_ai_matches()
            self._cancel_scheduled_callback()
        if len(self.history) <= 1:
            self.status_var.set("No previous move to undo.")
            return

        # In Human vs AI mode, undo both AI and Human move if applicable
        if self.mode_var.get() == "Human vs AI" and len(self.history) >= 3 and not self.game.is_terminal():
            self.history.pop()
            self.history.pop()
        else:
            self.history.pop()

        self.game = self.history[-1].copy()
        self.ai_busy = False
        self.hover_cell = None
        self._draw_board()
        self._update_status_message()

    def _handle_cell_click(self, row: int, col: int) -> None:
        if self.ai_busy or self.game.is_terminal():
            return

        if self.mode_var.get() == "AI vs AI":
            self.status_var.set("Use Start AI Matches to run the selected AIs.")
            return

        if self.mode_var.get() == "Human vs AI" and self.game.current_player == self.ai_player:
            self.status_var.set("AI is thinking...")
            return

        move = (row, col)
        if not self.game.is_valid_move(move):
            self.status_var.set("Invalid move. Choose an empty square on the board.")
            return

        self._apply_move(move)

    def _on_canvas_click(self, event) -> None:
        row, col = self._cell_from_event(event.x, event.y)
        if row is None or col is None:
            return
        self._handle_cell_click(row, col)

    def _cell_from_event(self, x: int, y: int):
        board_left = self.margin
        board_top = self.margin
        board_right = board_left + self.cell_size * self.game.board_size
        board_bottom = board_top + self.cell_size * self.game.board_size
        if x < board_left or y < board_top or x > board_right or y > board_bottom:
            return None, None

        col = int((x - board_left) // self.cell_size)
        row = int((y - board_top) // self.cell_size)
        if row >= self.game.board_size or col >= self.game.board_size:
            return None, None
        return row, col

    def _on_canvas_motion(self, event) -> None:
        """Provide a subtle ghost preview and hand cursor when hovering over empty cells."""
        if self.game.is_terminal() or self.ai_busy or self.batch_active:
            self._clear_hover()
            return

        if self.mode_var.get() == "AI vs AI":
            self._clear_hover()
            return

        if self.mode_var.get() == "Human vs AI" and self.game.current_player == self.ai_player:
            self._clear_hover()
            return

        row, col = self._cell_from_event(event.x, event.y)
        if row is None or col is None or self.game.board[row][col] != 0:
            self._clear_hover()
            return

        if self.hover_cell == (row, col):
            return

        self.hover_cell = (row, col)
        self.canvas.delete("hover_preview")
        self.canvas.config(cursor="hand2")

        cx = self.margin + col * self.cell_size + self.cell_size / 2
        cy = self.margin + row * self.cell_size + self.cell_size / 2
        half = self.cell_size / 2 - 2

        # Soft cell highlight
        self.canvas.create_rectangle(
            cx - half,
            cy - half,
            cx + half,
            cy + half,
            fill="#f1f5f9",
            outline=COLOR_GRID_BORDER,
            width=1,
            dash=(2, 2),
            tags="hover_preview",
        )

        # Faint ghost piece
        span = self.cell_size * 0.26
        if self.game.current_player == PLAYER_X:
            self.canvas.create_line(
                cx - span, cy - span, cx + span, cy + span,
                fill=COLOR_X_GHOST, width=2.5, capstyle=tk.ROUND, tags="hover_preview"
            )
            self.canvas.create_line(
                cx - span, cy + span, cx + span, cy - span,
                fill=COLOR_X_GHOST, width=2.5, capstyle=tk.ROUND, tags="hover_preview"
            )
        else:
            self.canvas.create_oval(
                cx - span, cy - span, cx + span, cy + span,
                outline=COLOR_O_GHOST, width=2.5, tags="hover_preview"
            )

    def _on_canvas_leave(self, _event=None) -> None:
        """Clear hover elements when cursor exits the canvas."""
        self._clear_hover()

    def _clear_hover(self) -> None:
        if self.hover_cell is not None:
            self.hover_cell = None
            self.canvas.delete("hover_preview")
            self.canvas.config(cursor="")

    def _apply_move(self, move) -> None:
        if not self.game.make_move(move):
            self.status_var.set("Move rejected by the game rules.")
            return

        self.history.append(self.game.copy())
        self._clear_hover()
        self._draw_board()
        self._update_status_message()

        if self.game.is_terminal():
            result = self.game.get_result()
            if result == 1:
                self.status_var.set("X wins!")
            elif result == -1:
                self.status_var.set("O wins!")
            elif result == 0:
                self.status_var.set("Draw game.")
            if self.batch_active:
                self._record_ai_match()
            return

        if self._is_ai_turn() or (self.batch_active and self.mode_var.get() == "AI vs AI"):
            self._schedule_ai_turn()

    def _schedule_ai_turn(self) -> None:
        if not self._is_ai_turn() and not (self.batch_active and self.mode_var.get() == "AI vs AI"):
            return
        self._cancel_scheduled_callback()
        self.ai_busy = True
        self.status_var.set("AI is thinking...")
        self._scheduled_callback = self.root.after(160, self._run_ai_turn)

    def _run_ai_turn(self) -> None:
        self._scheduled_callback = None
        if self.game.is_terminal() or not (self._is_ai_turn() or (self.batch_active and self.mode_var.get() == "AI vs AI")):
            self.ai_busy = False
            self._update_status_message()
            return

        controller = self._controller_for_current_player()
        move = controller.get_move(self.game)
        if move is None:
            self.ai_busy = False
            self.status_var.set(controller.last_status)
            if self.batch_active:
                self._stop_ai_matches()
            return

        if not self.game.is_valid_move(move):
            self.ai_busy = False
            self.status_var.set("AI returned an invalid move.")
            if self.batch_active:
                self._stop_ai_matches()
            return

        self.ai_busy = False
        self._apply_move(move)

    def _controller_for_current_player(self) -> AIController:
        algorithm = self.ai_var.get()
        if self.mode_var.get() == "AI vs AI" and self.game.current_player == PLAYER_O:
            algorithm = self.ai_o_var.get()
        return AIController(
            algorithm=algorithm,
            iterations=int(self.ai_iterations_var.get() or 1000),
            uct_c=float(self.uct_var.get() or 1.4),
            rollout_policy=self.rollout_var.get(),
        )

    def start_ai_matches(self) -> None:
        if self.mode_var.get() != "AI vs AI":
            self.status_var.set("Select AI vs AI mode to start a match series.")
            return
        try:
            game_count = int(self.batch_games_var.get())
        except ValueError:
            self.status_var.set("Enter a positive whole number of AI matches.")
            return
        if game_count < 1:
            self.status_var.set("Enter a positive whole number of AI matches.")
            return

        self._cancel_scheduled_callback()
        self.batch_target = game_count
        self.batch_completed = 0
        self.batch_x_wins = 0
        self.batch_o_wins = 0
        self.batch_draws = 0
        self.batch_active = True
        self._update_batch_stats()
        self._start_next_ai_match()

    def _start_next_ai_match(self) -> None:
        self._scheduled_callback = None
        if not self.batch_active:
            return
        self._new_game_state()
        player_name = "X" if self.game.current_player == PLAYER_X else "O"
        self.turn_var.set(f"Match {self.batch_completed + 1}/{self.batch_target}: {player_name} to move")
        self.status_var.set(f"AI match {self.batch_completed + 1}/{self.batch_target}")
        self._schedule_ai_turn()

    def _record_ai_match(self) -> None:
        result = self.game.get_result()
        if result == PLAYER_X:
            self.batch_x_wins += 1
        elif result == PLAYER_O:
            self.batch_o_wins += 1
        elif result == 0:
            self.batch_draws += 1
        else:
            return

        self.batch_completed += 1
        self._update_batch_stats()
        if self.batch_completed >= self.batch_target:
            self.batch_active = False
            self.ai_busy = False
            self.status_var.set(f"AI matches complete: {self.batch_completed}/{self.batch_target}.")
            return

        self.status_var.set(f"Match complete. Starting match {self.batch_completed + 1}/{self.batch_target}...")
        self._scheduled_callback = self.root.after(250, self._start_scheduled_ai_match)

    def _start_scheduled_ai_match(self) -> None:
        self._scheduled_callback = None
        self._start_next_ai_match()

    def _update_batch_stats(self) -> None:
        # Keep variable string format intact for unit tests
        if self.batch_completed == 0:
            self.batch_stats_var.set(
                f"Results: X ({self.ai_var.get()}) 0 | "
                f"O ({self.ai_o_var.get()}) 0 | Draws 0 | Completed 0/{self.batch_target}"
            )
        else:
            x_rate = self.batch_x_wins / self.batch_completed * 100
            o_rate = self.batch_o_wins / self.batch_completed * 100
            draw_rate = self.batch_draws / self.batch_completed * 100
            self.batch_stats_var.set(
                f"Results: X ({self.ai_var.get()}) {self.batch_x_wins} ({x_rate:.1f}%) | "
                f"O ({self.ai_o_var.get()}) {self.batch_o_wins} ({o_rate:.1f}%) | "
                f"Draws {self.batch_draws} ({draw_rate:.1f}%) | "
                f"Completed {self.batch_completed}/{self.batch_target}"
            )

        # Update visual badges
        self.badge_x.configure(text=f"X: {self.batch_x_wins}")
        self.badge_o.configure(text=f"O: {self.batch_o_wins}")
        self.badge_draw.configure(text=f"Hòa: {self.batch_draws}")

    def _stop_ai_matches(self) -> None:
        self.batch_active = False
        self.ai_busy = False

    def stop_ai_matches(self) -> None:
        if not self.batch_active:
            self.status_var.set("No AI match series is running.")
            return
        self._stop_ai_matches()
        self._cancel_scheduled_callback()
        self.status_var.set(f"AI match series stopped after {self.batch_completed}/{self.batch_target} games.")

    def _cancel_scheduled_callback(self) -> None:
        if self._scheduled_callback is not None:
            self.root.after_cancel(self._scheduled_callback)
            self._scheduled_callback = None

    def _is_ai_turn(self) -> bool:
        return (
            self.mode_var.get() == "Human vs AI"
            and self.ai_player is not None
            and not self.game.is_terminal()
            and self.game.current_player == self.ai_player
        )

    def _update_status_message(self) -> None:
        player_name = "X" if self.game.current_player == PLAYER_X else "O"
        self.turn_var.set(f"Current turn: {player_name}")
        self.move_count_var.set(f"Nước: {self.game.moves_count}")

        # Update visual turn badge in status card
        if not self.game.is_terminal():
            if self.game.current_player == PLAYER_X:
                self.turn_badge.configure(text="X", bg=COLOR_X, fg="#ffffff")
            else:
                self.turn_badge.configure(text="O", bg=COLOR_O, fg="#ffffff")

        if self.game.is_terminal():
            result = self.game.get_result()
            if result == 1:
                self.status_var.set("X wins!")
                self.turn_badge.configure(text="🏆", bg=COLOR_X, fg="#ffffff")
                self.turn_var.set("Quân X Chiến Thắng!")
            elif result == -1:
                self.status_var.set("O wins!")
                self.turn_badge.configure(text="🏆", bg=COLOR_O, fg="#ffffff")
                self.turn_var.set("Quân O Chiến Thắng!")
            elif result == 0:
                self.status_var.set("Draw game.")
                self.turn_badge.configure(text="🤝", bg=COLOR_TEXT_MUTED, fg="#ffffff")
                self.turn_var.set("Ván Đấu Hòa!")
            else:
                self.status_var.set("Game over.")
            return

        if self.mode_var.get() == "Human vs AI":
            if self._is_ai_turn():
                self.status_var.set("AI is thinking...")
            else:
                self.status_var.set("Your turn.")
        elif self.mode_var.get() == "AI vs AI":
            if not self.batch_active:
                self.status_var.set("Select the number of matches and start the AI series.")
        else:
            self.status_var.set(f"{player_name} to move.")

    def _get_winning_line(self) -> list[tuple[int, int]] | None:
        """Locate the consecutive winning coordinates to highlight."""
        if not self.game.is_terminal() or self.game.winner is None:
            return None

        winner = self.game.winner
        size = self.game.board_size
        win_len = self.game.win_length
        directions = ((0, 1), (1, 0), (1, 1), (1, -1))

        # Check candidate starting from last_move first
        candidates = []
        if self.game.last_move:
            candidates.append(self.game.last_move)
        for r in range(size):
            for c in range(size):
                if (r, c) not in candidates and self.game.board[r][c] == winner:
                    candidates.append((r, c))

        for r0, c0 in candidates:
            if self.game.board[r0][c0] != winner:
                continue
            for dr, dc in directions:
                line = [(r0, c0)]
                # Forward
                step = 1
                while True:
                    nr, nc = r0 + dr * step, c0 + dc * step
                    if self.game.is_inside_board(nr, nc) and self.game.board[nr][nc] == winner:
                        line.append((nr, nc))
                        step += 1
                    else:
                        break
                # Backward
                step = 1
                while True:
                    nr, nc = r0 - dr * step, c0 - dc * step
                    if self.game.is_inside_board(nr, nc) and self.game.board[nr][nc] == winner:
                        line.insert(0, (nr, nc))
                        step += 1
                    else:
                        break
                if len(line) >= win_len:
                    return line
        return None

    def _draw_board(self) -> None:
        """Render the complete game board, grid lines, stones, and markers."""
        self.canvas.delete("all")
        size = self.game.board_size
        board_width = self.cell_size * size
        board_height = self.cell_size * size

        # 1. Coordinate Labels (Subtle numbers around board)
        for index in range(size):
            cx = self.margin + index * self.cell_size + self.cell_size / 2
            cy = self.margin + index * self.cell_size + self.cell_size / 2
            # Top axis
            self.canvas.create_text(
                cx, 16, text=str(index + 1), fill=COLOR_TEXT_MUTED, font=(FONT_FAMILY, 8, "bold")
            )
            # Left axis
            self.canvas.create_text(
                14, cy, text=str(index + 1), fill=COLOR_TEXT_MUTED, font=(FONT_FAMILY, 8, "bold")
            )

        # 2. Grid Lines
        for index in range(size + 1):
            x = self.margin + index * self.cell_size
            y = self.margin + index * self.cell_size
            is_boundary = index in (0, size)
            line_color = COLOR_GRID_BORDER if is_boundary else COLOR_GRID_LINE
            line_width = 1.5 if is_boundary else 1.0

            # Vertical lines
            self.canvas.create_line(x, self.margin, x, self.margin + board_height, fill=line_color, width=line_width)
            # Horizontal lines
            self.canvas.create_line(self.margin, y, self.margin + board_width, y, fill=line_color, width=line_width)

        # 3. Star points (Hoshi landmarks on standard 15x15 Caro board)
        star_points = [(3, 3), (3, 11), (7, 7), (11, 3), (11, 11)]
        for sr, sc in star_points:
            sx = self.margin + sc * self.cell_size + self.cell_size / 2
            sy = self.margin + sr * self.cell_size + self.cell_size / 2
            self.canvas.create_oval(sx - 3, sy - 3, sx + 3, sy + 3, fill=COLOR_STAR_POINT, outline="")

        # 4. Winning Line Highlight (if terminal with winner)
        winning_line = self._get_winning_line()
        if winning_line:
            # Aura glow behind each winning stone
            for wr, wc in winning_line:
                wcx = self.margin + wc * self.cell_size + self.cell_size / 2
                wcy = self.margin + wr * self.cell_size + self.cell_size / 2
                half = self.cell_size / 2 - 2
                self.canvas.create_rectangle(
                    wcx - half, wcy - half, wcx + half, wcy + half,
                    fill=COLOR_WIN_GLOW, outline=COLOR_WIN_BORDER, width=1.5
                )

        # 5. Render Stones
        span = self.cell_size * 0.27
        for row in range(size):
            for col in range(size):
                cell_value = self.game.board[row][col]
                if cell_value == 0:
                    continue

                center_x = self.margin + col * self.cell_size + self.cell_size / 2
                center_y = self.margin + row * self.cell_size + self.cell_size / 2

                if cell_value == PLAYER_X:
                    # Drop shadow
                    self.canvas.create_line(
                        center_x - span + 1, center_y - span + 1,
                        center_x + span + 1, center_y + span + 1,
                        fill=COLOR_X_SHADOW, width=3.5, capstyle=tk.ROUND
                    )
                    self.canvas.create_line(
                        center_x - span + 1, center_y + span + 1,
                        center_x + span + 1, center_y - span + 1,
                        fill=COLOR_X_SHADOW, width=3.5, capstyle=tk.ROUND
                    )
                    # Main X stroke
                    self.canvas.create_line(
                        center_x - span, center_y - span,
                        center_x + span, center_y + span,
                        fill=COLOR_X, width=3.2, capstyle=tk.ROUND
                    )
                    self.canvas.create_line(
                        center_x - span, center_y + span,
                        center_x + span, center_y - span,
                        fill=COLOR_X, width=3.2, capstyle=tk.ROUND
                    )
                else:
                    # Drop shadow
                    self.canvas.create_oval(
                        center_x - span + 1, center_y - span + 1,
                        center_x + span + 1, center_y + span + 1,
                        outline=COLOR_O_SHADOW, width=3.5
                    )
                    # Main O stroke
                    self.canvas.create_oval(
                        center_x - span, center_y - span,
                        center_x + span, center_y + span,
                        outline=COLOR_O, width=3.2
                    )

                # Last Move Highlight Ring
                if self.game.last_move == (row, col):
                    half = self.cell_size / 2 - 3
                    self.canvas.create_rectangle(
                        center_x - half, center_y - half,
                        center_x + half, center_y + half,
                        outline=COLOR_LAST_MOVE, width=2.0
                    )
                    self.canvas.create_oval(
                        center_x - 2, center_y - 2,
                        center_x + 2, center_y + 2,
                        fill=COLOR_LAST_MOVE, outline=""
                    )

        # 6. Strike-through line connecting the winning stones
        if winning_line and len(winning_line) >= 2:
            first_r, first_c = winning_line[0]
            last_r, last_c = winning_line[-1]
            x1 = self.margin + first_c * self.cell_size + self.cell_size / 2
            y1 = self.margin + first_r * self.cell_size + self.cell_size / 2
            x2 = self.margin + last_c * self.cell_size + self.cell_size / 2
            y2 = self.margin + last_r * self.cell_size + self.cell_size / 2
            self.canvas.create_line(
                x1, y1, x2, y2, fill=COLOR_WIN_LINE, width=4.0, capstyle=tk.ROUND
            )

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    app = TicTacToeGUI()
    app.run()
