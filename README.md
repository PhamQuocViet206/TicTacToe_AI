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
│   ├── minimax.py            # Minimax + alpha-beta
│   ├── mcts.py               # MCTS + UCT
│   ├── verify_mcts.py        # Script kiểm chứng MCTS (3x3 không thua, bảng W/D/L, W/N tại gốc)
│   ├── rollout.py            # Các policy chọn nước đi trong rollout
│   ├── test_game.py          # Unit test base game
│   ├── test_gui.py           # Unit test giao diện
│   ├── test_rollout.py       # Unit test rollout policy
│   └── test_mcts.py          # Unit test MCTS
└── requirements.txt          # Không bắt buộc nếu dùng Python chuẩn
```

### Trách nhiệm từng module

- `constants.py`: chứa các hằng số như kích thước bàn cờ, giá trị quân, độ dài thắng.
- `game.py`: quản lý trạng thái, nước đi hợp lệ, kiểm tra thắng/hòa, copy trạng thái, API cho AI.
- `gui.py`: vẽ bàn cờ, xử lý chuột, hiển thị lượt đi, trạng thái thắng/hòa, hỗ trợ chế độ người vs người và người vs AI.
- `ai_controller.py`: lớp trung gian để GUI không phụ thuộc vào thuật toán cụ thể.
- `minimax.py`: Minimax + alpha-beta.
- `mcts.py`: MCTS + UCT (`MCTSAgent`), đã tích hợp vào `ai_controller.py`; `verify_mcts.py`: script kiểm chứng và xuất bảng số liệu.
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
- Chọn chế độ AI vs AI, cấu hình thuật toán và cài đặt nâng cao riêng biệt cho cả X và O (số lần lặp MCTS, hệ số UCT c, rollout policy), đặt số ván rồi nhấn `Start AI Matches`. Kết quả hiển thị số trận thắng, tỷ lệ thắng của X/O, tỷ lệ hòa, đồng thời tính toán và in thời gian trung bình mỗi nước đi của từng AI; X/O và người đi trước được giữ cố định trong cả loạt.
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

`tictactoe/mcts.py` cài đặt MCTS + UCT với `UCT(i) = Wᵢ/Nᵢ + c·√(ln N_p / Nᵢ)`, gồm 4 pha Selection – Expansion – Simulation – Backpropagation. Khi chọn MCTS trong GUI, chọn thêm số iteration, hệ số c và rollout policy `random` hoặc `heuristic`. Rollout policy chỉ điều khiển cách MCTS chọn nước trong mô phỏng, không phải thuật toán chơi độc lập. API policy nằm trong `tictactoe/rollout.py`.

Một số điểm chính:

- Phần thưởng: thắng 1, hoà 0.5, thua 0, tính theo góc nhìn người vừa đi nước dẫn vào nút; mỗi tầng đảo thành `1 − r`.
- Nước cuối được chọn theo **N lớn nhất** (mặc định); `final_policy="max_wn"` chỉ dùng để so sánh trong phần phân tích.
- Bàn ≤ 5×5 xét mọi ô trống. Bàn lớn hơn chỉ mở rộng các ô trống cách quân đã đặt ≤ 2 ô; bàn rỗng đi ô giữa. Rollout vẫn chạy trên toàn bàn.
- `agent.root_stats()` trả bảng W, N, W/N, UCT tại gốc; `agent.last_stats` có số iteration, số nút cây, thời gian.

Thời gian một nước đo trên máy phát triển (rollout `random`):

| Bàn | Iteration | Thời gian |
|---|---|---|
| 3×3 | 5000 | ~1.2 s |
| 9×9, k = 5 | 1000 | ~4 s |
| 15×15, k = 5 | 100 | ~1.5 s |
| 15×15, k = 5 | 1000 | ~17 s |

GUI gọi AI đồng bộ, nên khi chơi 15×15 nên đặt **100–200 iteration**. Rollout `heuristic` hiện rất chậm trên 15×15 (khoảng 2.6 s mỗi rollout), chỉ nên dùng trên 3×3.

Kiểm chứng và xuất bảng số liệu bằng `tictactoe/verify_mcts.py` (xem docstring đầu file để biết đủ tuỳ chọn):

```powershell
python -m tictactoe.verify_mcts --iterations 5000 --games 50 --opponent perfect --seed 1   # 3x3 vs đối thủ tối ưu
python -m tictactoe.verify_mcts --iterations 5000 --games 50 --opponent random             # 3x3 vs ngẫu nhiên
python -m tictactoe.verify_mcts --threshold 100 200 500 1000 2000 5000 --games 50 --csv mcts_3x3.csv
python -m tictactoe.verify_mcts --board 15 --k 5 --iterations 100 --games 50 --opponent random
python -m tictactoe.verify_mcts --root-stats --iterations 1000 --seed 7 --moves "0,0 1,1 0,1"
python -m tictactoe.verify_mcts --exhaustive --iterations 5000 --seed 1
```

`--opponent perfect` là bộ giải negamax chính xác cho 3×3 (chọn ngẫu nhiên giữa các nước tối ưu). Khi `MinimaxAI` hoàn thành, dùng `--opponent minimax`.

Kết quả kiểm chứng (c = 1.4, rollout random, seed 1, mỗi dòng 50 ván, đổi lượt đi trước 25/25):

| Bàn | Đối thủ | Iteration | Thắng | Hoà | Thua |
|---|---|---|---|---|---|
| 3×3 | perfect | 100 / 200 / 500 / 1000 | 0 | 36 / 41 / 42 / 49 | 14 / 9 / 8 / 1 |
| 3×3 | perfect | 2000 / 5000 | 0 | 50 | **0** |
| 3×3 | random | 5000 | 48 | 2 | **0** |
| 3×3 | self | 5000 | 0 | 50 | 0 |
| 15×15, k = 5 | random | 100 | 41 | 0 | 9 |

Ngưỡng nhỏ nhất để MCTS không thua ván nào trên 3×3 là **2000 iteration**. Kiểm toàn diện ở 5000 iteration (`--exhaustive`, duyệt mọi phản ứng của đối thủ) cho 0 thua trên cả 100 nhánh khi MCTS đi trước và 461 nhánh khi MCTS đi sau.

### 9.3 Heuristic Rollout

`heuristic_rollout(game)` ưu tiên nước thắng ngay, sau đó chặn nước thắng ngay của đối thủ, nếu không có thì chọn ngẫu nhiên. MCTS nhận hàm này làm rollout policy khi chọn `heuristic` trong GUI. Minimax hiện chưa được tích hợp.

### 9.4 Chạy nhiều ván AI vs AI

Chọn `AI vs AI`, thiết lập thuật toán riêng cho X và O, nhập số ván, sau đó nhấn `Start AI Matches`. GUI chạy từng nước qua lịch `Tkinter.after()` và cập nhật số ván thắng, tỷ lệ thắng cùng tỷ lệ hòa theo tổng số ván đã hoàn tất. X/O và người đi trước được giữ cố định trong loạt; người đi trước được chọn bằng điều khiển `First player`. `Stop AI Matches` hoặc `New Game` dừng loạt. Nếu chọn thuật toán chưa được tích hợp (hiện là Minimax), hoặc nhập tham số MCTS không hợp lệ (ví dụ c < 0), loạt dừng và hiển thị lý do thay vì tính kết quả sai.

## 10. Chạy kiểm thử

Từ thư mục gốc của dự án:

```powershell
$env:PYTHONUTF8 = "1"
python -m unittest discover -s tictactoe -p "test_*.py" -t .
```

`PYTHONUTF8=1` tránh lỗi `UnicodeEncodeError` khi GUI in tiếng Việt ra console Windows (cp1252). Các test chạy lâu (kiểm chứng 50 ván 3×3, hiệu năng) chỉ chạy khi đặt thêm `$env:RUN_SLOW = "1"`.

## 11. Những phần chưa hoàn thiện

- Minimax chưa được tích hợp.
- MCTS trên 15×15 còn chậm, vì rollout gọi `get_valid_moves()` duyệt toàn bàn ở mỗi bước; `heuristic_rollout` còn chậm hơn nhiều.

## 12. Kết luận

Phiên bản này đã hoàn thành base game đúng luật, GUI chạy được trên Tkinter, API chuẩn cho các thuật toán AI, và kiểm thử cơ bản đã được triển khai. Các thành viên còn lại có thể tích hợp thuật toán theo đúng hướng dẫn trên mà không cần viết lại toàn bộ hệ thống.
