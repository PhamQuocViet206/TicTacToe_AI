# TicTacToe AI Project

## 1. Giới thiệu

Dự án này triển khai trò chơi TicTacToe 15x15 theo luật 5 quân liên tiếp bằng Python và Tkinter. Mục tiêu là cung cấp một base game đúng luật, đầy đủ API để các thành viên trong nhóm tích hợp thuật toán AI, và một giao diện đồ họa có thể chạy trên Windows.

## 2. Kiến trúc phần mềm

### Bước 1: Kiến trúc đề xuất

```text
TicTacToe/
├── main.py                    # Chạy ứng dụng GUI
├── README.md                  # Tài liệu chính
├── README_API.md              # Tài liệu API cho các thành viên AI
├── tictactoe/
│   ├── __init__.py            # Export package
│   ├── constants.py           # Hằng số chung
│   ├── game.py               # Logic luật chơi TicTacToe
│   ├── gui.py                # Giao diện Tkinter
│   ├── ai_controller.py      # Adapter giữa GUI và AI
│   ├── minimax.py            # Placeholder cho Minimax
│   ├── mcts.py               # Placeholder cho MCTS
│   ├── rollout.py            # Các policy chọn nước đi trong rollout
│   ├── test_game.py          # Unit test base game
│   └── test_gui.py           # Unit test giao diện
└── requirements.txt          # Không bắt buộc nếu dùng Python chuẩn
```

### Trách nhiệm từng module

- `constants.py`: chứa các hằng số như kích thước bàn cờ, giá trị quân, độ dài thắng.
- `game.py`: quản lý trạng thái, nước đi hợp lệ, kiểm tra thắng/hòa, copy trạng thái, API cho AI.
- `gui.py`: vẽ bàn cờ, xử lý chuột, hiển thị lượt đi, trạng thái thắng/hòa, hỗ trợ chế độ người vs người và người vs AI.
- `ai_controller.py`: lớp trung gian để GUI không phụ thuộc vào thuật toán cụ thể.
- `minimax.py`, `mcts.py`: placeholder để tích hợp thuật toán.
- `rollout.py`: các chính sách rollout `random` và `heuristic`; đây là chiến lược chọn nước đi trong mô phỏng MCTS, không phải thuật toán AI riêng.
- `test_game.py` và `test_gui.py`: kiểm thử chức năng cơ bản và GUI.

## 3. Yêu cầu công nghệ

- Python 3.11+
- Tkinter (được tích hợp sẵn với Python Windows)
- Không cần NumPy cho phiên bản hiện tại

## 4. Cài đặt

### Windows (VS Code)

1. Mở VS Code.
2. Chọn thư mục dự án.
3. Mở terminal trong VS Code.
4. Kiểm tra Python:

```powershell
python --version
```

5. Nếu Python chưa được cài đặt, tải Python 3.11+ từ https://www.python.org/downloads/.
6. Chọn Python trong VS Code bằng `Ctrl+Shift+P` > `Python: Select Interpreter`.

## 5. Chạy chương trình

Từ thư mục gốc của dự án:

```powershell
python main.py
```

Sau khi ứng dụng mở, bạn có thể:

- Chơi người với người.
- Chọn chế độ người vs AI.
- Chọn người đi trước cho ván thủ công/người vs AI.
- Chọn thuật toán AI (random là AI thử nghiệm demo).
- Chọn chế độ AI vs AI, cấu hình thuật toán riêng cho X và O, đặt số ván rồi nhấn `Start AI Matches`. Kết quả hiển thị số trận thắng và tỷ lệ thắng của X/O cùng tỷ lệ hòa; X/O và người đi trước được giữ cố định trong cả loạt.
- Dùng `Stop AI Matches` hoặc `New Game` để dừng loạt.
- Bắt đầu ván mới / undo / restart.

## 6. Cách chơi

- Bàn cờ 15x15.
- X đi trước, O đi sau.
- Ô trống = 0.
- Quân X = 1.
- Quân O = -1.
- Người thắng là người tạo chuỗi ít nhất 5 quân liên tiếp theo hàng, cột hoặc đường chéo.
- Nếu bàn cờ đầy mà không ai thắng thì hòa.

## 7. Quy tắc trò chơi

- Không cho phép đánh ngoài bàn cờ.
- Không cho phép đánh vào ô đã có quân.
- Không cho phép đánh tiếp sau khi ván đấu kết thúc.
- Luật thắng: `count >= 5`.

## 8. API của TicTacToeGame

### Lớp `TicTacToeGame`

```python
from tictactoe.game import TicTacToeGame

game = TicTacToeGame()

moves = game.get_valid_moves()
next_state = game.copy()
next_state.make_move((5, 5))

if next_state.is_terminal():
    result = next_state.get_result()
```

### Các phương thức chính

- `reset()`
- `is_inside_board(row, col)`
- `is_valid_move(move)`
- `get_valid_moves()`
- `make_move(move, player=None)`
- `check_winner(move=None, player=None)`
- `is_draw()`
- `is_terminal()`
- `get_result()`
- `copy()`
- `display()`

## 9. Hướng dẫn tích hợp AI

### 9.1 Minimax

Các thành viên đang làm Minimax có thể tạo logic trong file `tictactoe/minimax.py` hoặc gọi `TicTacToeGame.copy()` để mô phỏng trạng thái. Đảm bảo không thay đổi trạng thái gốc trong quá trình mô phỏng.

### 9.2 MCTS

File `tictactoe/mcts.py` là nơi để triển khai MCTS + UCT. Khi chọn MCTS trong GUI, chọn thêm rollout policy `random` hoặc `heuristic`. Hai lựa chọn này chỉ điều khiển cách MCTS chọn nước trong rollout, không phải các thuật toán chơi độc lập. API policy nằm trong `tictactoe/rollout.py` và cung cấp các hàm `random_rollout(game)` và `heuristic_rollout(game)`.

### 9.3 Heuristic Rollout

`heuristic_rollout(game)` ưu tiên nước thắng ngay, sau đó chặn nước thắng ngay của đối thủ, nếu không có thì chọn ngẫu nhiên. MCTS có thể nhận hàm này làm rollout policy. Minimax và MCTS hiện chưa được tích hợp; lựa chọn rollout trong giao diện chưa tự tạo AI chơi độc lập.

### 9.4 Chạy nhiều ván AI vs AI

Chọn `AI vs AI`, thiết lập thuật toán riêng cho X và O, nhập số ván, sau đó nhấn `Start AI Matches`. GUI chạy từng nước qua lịch `Tkinter.after()` và cập nhật số ván thắng, tỷ lệ thắng cùng tỷ lệ hòa theo tổng số ván đã hoàn tất. X/O và người đi trước được giữ cố định trong loạt; người đi trước được chọn bằng điều khiển `First player`. `Stop AI Matches` hoặc `New Game` dừng loạt. Nếu chọn thuật toán chưa được tích hợp (Minimax/MCTS), loạt dừng và hiển thị lý do thay vì tính kết quả sai.

## 10. Chạy kiểm thử

```powershell
cd d:\Projects\TicTacToe
python -m unittest tictactoe.test_game tictactoe.test_gui tictactoe.test_rollout
```

## 11. Những phần chưa hoàn thiện

- Minimax chưa được tích hợp.
- MCTS chưa được tích hợp.
- Heuristic rollout policy đã có thể chọn và dùng độc lập trong mô phỏng, nhưng chưa được gọi bởi MCTS vì MCTS chưa được tích hợp.
- AI hiện tại chỉ có chế độ random demo để kiểm tra giao diện.

## 12. Kết luận

Phiên bản này đã hoàn thành base game đúng luật, GUI chạy được trên Tkinter, API chuẩn cho các thuật toán AI, và kiểm thử cơ bản đã được triển khai. Các thành viên còn lại có thể tích hợp thuật toán theo đúng hướng dẫn trên mà không cần viết lại toàn bộ hệ thống.
