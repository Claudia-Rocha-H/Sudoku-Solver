import tkinter as tk
from tkinter import filedialog, messagebox

from sudoku.core import is_valid_initial_board, solve_board
from sudoku.vision import load_sudoku_from_image


class SudokuGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sudoku Solver")
        self.root.configure(padx=20, pady=20, bg="#f3f6fa")

        self.cells = [[None] * 9 for _ in range(9)]
        self.original_cells = set()

        self.default_bg_1 = "#ffffff"
        self.default_bg_2 = "#e9f5e9"
        self.fixed_bg = "#dfe8f5"
        self.solved_bg = "#fff8dc"

        self._build_header()
        self._build_board()
        self._build_controls()
        self._build_status_bar()
        self.cells[0][0].focus_set()

    def _build_header(self):
        title = tk.Label(
            self.root,
            text="Sudoku Solver",
            font=("Arial", 20, "bold"),
            bg="#f3f6fa",
            fg="#1f2d3d",
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            self.root,
            text="Enter numbers from 1 to 9 and leave unknown cells empty.",
            font=("Arial", 10),
            bg="#f3f6fa",
            fg="#465a6c",
        )
        subtitle.pack(anchor="w", pady=(0, 12))

    def _build_board(self):
        board = tk.Frame(self.root, bg="#1f2d3d", padx=2, pady=2)
        board.pack()

        validator = (self.root.register(self.validate_cell_input), "%P")

        for row in range(9):
            for col in range(9):
                cell_bg = self._cell_bg(row, col)
                frame = tk.Frame(board, width=48, height=48, bg=cell_bg)
                frame.grid(
                    row=row,
                    column=col,
                    padx=(2 if col % 3 == 0 else 1, 2 if col % 3 == 2 else 1),
                    pady=(2 if row % 3 == 0 else 1, 2 if row % 3 == 2 else 1),
                )
                frame.pack_propagate(False)

                entry = tk.Entry(
                    frame,
                    width=2,
                    justify="center",
                    font=("Arial", 18, "bold"),
                    borderwidth=0,
                    relief="flat",
                    bg=cell_bg,
                    validate="key",
                    validatecommand=validator,
                )
                entry.pack(fill="both", expand=True)
                entry.bind("<Up>", lambda event, r=row, c=col: self.move_focus(r, c, "up"))
                entry.bind("<Down>", lambda event, r=row, c=col: self.move_focus(r, c, "down"))
                entry.bind("<Left>", lambda event, r=row, c=col: self.move_focus(r, c, "left"))
                entry.bind("<Right>", lambda event, r=row, c=col: self.move_focus(r, c, "right"))
                self.cells[row][col] = entry

    def _build_controls(self):
        controls = tk.Frame(self.root, bg="#f3f6fa")
        controls.pack(fill="x", pady=(14, 8))

        buttons = [
            {
                "text": "Solve",
                "command": self.solve_from_ui,
                "font": ("Arial", 11, "bold"),
                "bg": "#2d7dd2",
                "fg": "white",
                "activebackground": "#1f5fa8",
                "activeforeground": "white",
                "padx": 14,
                "pady": 6,
                "borderwidth": 0,
                "pack": {"side": "left"},
            },
            {
                "text": "Clear",
                "command": self.clear_board,
                "font": ("Arial", 11),
                "padx": 14,
                "pady": 6,
                "borderwidth": 0,
                "bg": "#d9e2ec",
                "activebackground": "#c6d3e1",
                "pack": {"side": "left", "padx": 8},
            },
            {
                "text": "New Sudoku",
                "command": self.new_sudoku,
                "font": ("Arial", 11),
                "padx": 14,
                "pady": 6,
                "borderwidth": 0,
                "bg": "#d9e2ec",
                "activebackground": "#c6d3e1",
                "pack": {"side": "left"},
            },
            {
                "text": "Load Image",
                "command": self.load_from_image,
                "font": ("Arial", 11),
                "padx": 14,
                "pady": 6,
                "borderwidth": 0,
                "bg": "#d9e2ec",
                "activebackground": "#c6d3e1",
                "pack": {"side": "left", "padx": (8, 0)},
            },
        ]

        for cfg in buttons:
            pack_cfg = cfg.pop("pack")
            button = tk.Button(controls, **cfg)
            button.pack(**pack_cfg)

    def _build_status_bar(self):
        self.status_label = tk.Label(
            self.root,
            text="Ready to solve.",
            font=("Arial", 10),
            bg="#f3f6fa",
            fg="#2f4858",
        )
        self.status_label.pack(anchor="w", pady=(4, 0))

    def _cell_bg(self, row, col):
        return self.default_bg_1 if (row // 3 + col // 3) % 2 == 0 else self.default_bg_2

    def _set_cell_default_style(self, row, col):
        self.cells[row][col].config(bg=self._cell_bg(row, col), fg="#1f2d3d")

    def validate_cell_input(self, value):
        if value == "":
            return True
        return len(value) == 1 and value in "123456789"

    def move_focus(self, row, col, direction):
        if direction == "right":
            next_row, next_col = row, (col + 1) % 9
        elif direction == "left":
            next_row, next_col = row, (col - 1) % 9
        elif direction == "down":
            next_row, next_col = (row + 1) % 9, col
        else:
            next_row, next_col = (row - 1) % 9, col

        self.cells[next_row][next_col].focus_set()
        return "break"

    def get_data(self):
        sudoku_data = [[0] * 9 for _ in range(9)]
        original_cells = set()

        for i in range(9):
            for j in range(9):
                value = self.cells[i][j].get().strip()
                if value:
                    sudoku_data[i][j] = int(value)
                    original_cells.add((i, j))

        return sudoku_data, original_cells

    def set_board_data(self, board_data, mark_as_original=True):
        self.original_cells.clear()

        for i in range(9):
            for j in range(9):
                entry = self.cells[i][j]
                entry.delete(0, "end")

                value = board_data[i][j]
                if value != 0:
                    entry.insert(0, str(value))
                    if mark_as_original:
                        self.original_cells.add((i, j))

                if mark_as_original and value != 0:
                    entry.config(bg=self.fixed_bg, fg="#1f2d3d")
                else:
                    self._set_cell_default_style(i, j)

    def _status_update(self, text, color="#2f4858"):
        self.status_label.config(text=text, fg=color)
        self.root.update_idletasks()

    def load_from_image(self):
        file_path = filedialog.askopenfilename(
            title="Select a Sudoku image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )

        if not file_path:
            return

        self._status_update("Processing image...", "#2f4858")

        try:
            board_data, detected_count = load_sudoku_from_image(
                file_path,
                status_callback=self._status_update,
            )
        except RuntimeError as error:
            msg = str(error)
            if "Train it" in msg or "digit_cnn" in msg:
                messagebox.showerror(
                    "Model not found",
                    "The CNN model does not exist yet.\n\nRun in the project folder:\n"
                    "  python train_cnn.py --epochs 30 --batch-size 128",
                )
            else:
                messagebox.showerror("Missing dependencies", msg)
            self._status_update("CNN model not found. Run train_cnn.py first.", "#a4161a")
            return
        except Exception as error:
            messagebox.showerror("Error processing image", str(error))
            self._status_update("Could not read the selected image.", "#a4161a")
            return

        self.set_board_data(board_data, mark_as_original=True)

        if detected_count == 0:
            self.status_label.config(text="No numbers were detected in the image.", fg="#8a4b08")
            messagebox.showwarning("No detection", "No digits were detected. Try a clearer image.")
            return

        if is_valid_initial_board(board_data):
            self.status_label.config(
                text=f"Image loaded: {detected_count} cells detected. You can solve now.",
                fg="#2f4858",
            )
        else:
            self.status_label.config(
                text="Image loaded, but conflicts were detected. Fix them before solving.",
                fg="#a4161a",
            )
            messagebox.showwarning(
                "Conflicts detected",
                "OCR loaded the board, but there are conflicts in rows/columns/boxes. Review the values.",
            )

    def solve_from_ui(self):
        sudoku_data, original_cells = self.get_data()
        self.original_cells = original_cells

        if len(original_cells) == 0:
            messagebox.showwarning("Empty Sudoku", "Enter at least one number to solve.")
            self.status_label.config(text="There is no data to solve.", fg="#8a4b08")
            return

        if not is_valid_initial_board(sudoku_data):
            messagebox.showerror("Invalid input", "The initial board has conflicts (row, column, or box).")
            self.status_label.config(text="Invalid board. Fix the conflicts.", fg="#a4161a")
            return

        if solve_board(sudoku_data):
            self.print_sudoku(sudoku_data)
            self.status_label.config(text="Sudoku solved. You can clear it or start a new one.", fg="#1b7f3b")
        else:
            messagebox.showinfo("No solution", "No solution was found for this Sudoku.")
            self.status_label.config(text="Sudoku has no solution.", fg="#a4161a")

    def print_sudoku(self, final_sudoku):
        for i in range(9):
            for j in range(9):
                entry = self.cells[i][j]
                entry.delete(0, "end")
                entry.insert(0, str(final_sudoku[i][j]))
                if (i, j) in self.original_cells:
                    entry.config(bg=self.fixed_bg, fg="#1f2d3d")
                else:
                    entry.config(bg=self.solved_bg, fg="#184e77")

    def clear_board(self):
        self.original_cells.clear()
        for i in range(9):
            for j in range(9):
                entry = self.cells[i][j]
                entry.delete(0, "end")
                self._set_cell_default_style(i, j)
        self.status_label.config(text="Board cleared.", fg="#2f4858")

    def new_sudoku(self):
        self.clear_board()
        self.status_label.config(text="New Sudoku ready for input.", fg="#2f4858")
