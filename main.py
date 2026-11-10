"""Entry point for the TicTacToe GUI application."""

import tkinter as tk

from tictactoe.gui import TicTacToeGUI


def main() -> None:
    root = tk.Tk()
    app = TicTacToeGUI(root)
    app.run()


if __name__ == "__main__":
    main()
