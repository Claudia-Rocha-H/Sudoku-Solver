def solve_board(board):
    row, col = find_empty(board)
    if row == -1:
        return True

    for value in range(1, 10):
        if is_valid(board, row, col, value):
            board[row][col] = value
            if solve_board(board):
                return True
        board[row][col] = 0
    return False


def find_empty(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                return row, col
    return -1, -1


def is_valid(board, row, col, value):
    for j in range(9):
        if value == board[row][j]:
            return False

    for i in range(9):
        if value == board[i][col]:
            return False

    box_row = (row // 3) * 3
    box_col = (col // 3) * 3
    for i in range(box_row, box_row + 3):
        for j in range(box_col, box_col + 3):
            if board[i][j] == value:
                return False
    return True


def is_valid_initial_board(board):
    for i in range(9):
        row_values = [n for n in board[i] if n != 0]
        if len(row_values) != len(set(row_values)):
            return False

    for j in range(9):
        col_values = [board[i][j] for i in range(9) if board[i][j] != 0]
        if len(col_values) != len(set(col_values)):
            return False

    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            values = []
            for i in range(box_row, box_row + 3):
                for j in range(box_col, box_col + 3):
                    if board[i][j] != 0:
                        values.append(board[i][j])
            if len(values) != len(set(values)):
                return False
    return True
