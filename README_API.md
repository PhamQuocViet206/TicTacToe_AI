# README_API

## Mục tiêu

Tài liệu API này mô tả cách các thành viên AI tương tác với base game và GUI.

## 1. TicTacToeGame

```python
from tictactoe.game import TicTacToeGame

game = TicTacToeGame(board_size=15, win_length=5)
```

### Thuộc tính

- `board`: ma trận 2D 15x15
- `board_size`: kích thước bàn cờ
- `win_length`: số quân liên tiếp để thắng
- `current_player`: người đang đánh (`1` cho X, `-1` cho O)
- `winner`: người thắng hoặc `None`
- `moves_count`: số lượt đã đánh
- `last_move`: nước đi gần nhất

### Phương thức quan trọng

```python
valid_moves = game.get_valid_moves()
next_state = game.copy()
next_state.make_move((7, 7))

if next_state.is_terminal():
    print(next_state.get_result())
```

### Một số nguyên tắc an toàn

- `copy()` phải tạo trạng thái độc lập.
- `make_move()` phải kiểm tra tính hợp lệ trước khi cập nhật.
- Nếu nước đi không hợp lệ, trạng thái gốc không bị thay đổi.
- `get_result()` trả về `1`, `-1`, `0` hoặc `None`.

## 2. AI Controller

```python
from tictactoe.ai_controller import AIController

controller = AIController(algorithm="random", iterations=1000, uct_c=1.4)
move = controller.get_move(game)
```

### Mục đích

- Tách GUI khỏi thuật toán cụ thể.
- Cho phép đổi thuật toán mà không sửa giao diện.

## 3. Rollout policy và MCTS

Rollout policy là chiến lược chọn nước đi bên trong mô phỏng MCTS, không phải một AI chơi độc lập. Chọn policy qua `rollout_policy`:

```python
from tictactoe.ai_controller import AIController
from tictactoe.rollout import heuristic_rollout, random_rollout

random_controller = AIController(algorithm="mcts", rollout_policy="random")
heuristic_controller = AIController(algorithm="mcts", rollout_policy="heuristic")
```

- `random_rollout(game)`: chọn ngẫu nhiên một nước đi hợp lệ.
- `heuristic_rollout(game)`: ưu tiên thắng ngay, tiếp đến chặn thắng ngay của đối thủ, sau đó chọn ngẫu nhiên.
- `tictactoe.mcts.MCTSAgent` nhận một callable rollout policy riêng (mặc định `random_rollout`).

### MCTSAgent

```python
import random
from tictactoe.game import TicTacToeGame
from tictactoe.mcts import MCTSAgent
from tictactoe.rollout import heuristic_rollout

game = TicTacToeGame(board_size=3, win_length=3)
random.seed(7)                       # rollout policy dùng random toàn cục
agent = MCTSAgent(
    iterations=1000,                 # số nguyên >= 1
    uct_c=1.4,                       # >= 0, xấp xỉ √2
    rollout_policy=heuristic_rollout,
    final_policy="max_n",            # "max_n" (mặc định) hoặc "max_wn" (chỉ để phân tích)
    seed=7,                          # RNG riêng cho mở rộng và phá hoà
    candidate_radius=None,           # None: tự chọn (bàn <= 5: mọi ô; bàn lớn: bán kính 2)
)
move = agent.get_move(game)          # (row, col); None nếu ván đã kết thúc; không đổi `game`
agent.root_stats()                   # [{"move", "W", "N", "WN", "uct"}, ...] sắp theo N giảm dần
agent.last_stats                     # iterations, tree_size, time_s, c, rollout, final_policy,
                                     # best_by_n, best_by_wn, max_n_equals_max_wn, warnings
```

Tham số không hợp lệ ném `ValueError`. `AIController` bắt lỗi này, trả `None` và ghi lý do vào `last_status`. Agent vừa dùng được lưu ở `controller.last_agent` để đọc `root_stats()` / `last_stats`.

Các thuật toán có thể tích hợp:

- `tictactoe.minimax.MinimaxAI` (chưa cài đặt)
- `tictactoe.mcts.MCTSAgent` (đã tích hợp)

### Ví dụ giao tiếp với GUI

```python
controller = AIController(algorithm="minimax")
move = controller.get_move(game)
```

Nếu thuật toán chưa được triển khai, `controller.get_move()` trả về `None` và ghi trạng thái rõ ràng bằng `last_status`.

## 4. API GUI

```python
from tictactoe.gui import TicTacToeGUI

gui = TicTacToeGUI()
```

### Bản chất

- GUI chịu trách nhiệm vẽ bàn cờ.
- GUI không trực tiếp thay đổi logic AI.
- GUI chỉ gọi `AIController` qua API chuẩn hóa.

### AI vs AI tournament

Trong GUI, chọn `AI vs AI`, chọn thuật toán riêng cho X và O, nhập số ván rồi nhấn `Start AI Matches`. Kết quả được cộng dồn theo quân X/O và hiển thị số trận thắng, tỷ lệ thắng và hòa trên số ván đã hoàn tất. Người đi trước được chọn qua `First player` và giữ cố định trong loạt. Có thể nhấn `Stop AI Matches` để dừng sớm. Chỉ thuật toán đã triển khai mới hoàn tất được (hiện là Random và MCTS); thuật toán placeholder dừng loạt với thông báo trạng thái.

## 5. Ví dụ phát triển AI tự do

```python
from tictactoe.game import TicTacToeGame

state = TicTacToeGame()
state.make_move((7, 7))
state.make_move((7, 8))

next_state = state.copy()
next_state.make_move((6, 6))

if next_state.is_terminal():
    print(next_state.get_result())
```

## 6. Lưu ý khi bàn giao cho nhóm

- Không sửa `game.py` để chứa logic Minimax hoặc MCTS.
- Không thay đổi API của `TicTacToeGame` khi chưa cần thiết.
- Mọi AI nên làm việc trên bản sao của trạng thái để tránh thay đổi game gốc.
- Mọi thuật toán mới chỉ nên gọi qua `AIController` hoặc thông qua `game.copy()`.
