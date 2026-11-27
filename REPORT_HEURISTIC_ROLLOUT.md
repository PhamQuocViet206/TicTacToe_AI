# Báo Cáo Mục 5: Mở Rộng MCTS Với Heuristic Rollout

- **Người thực hiện:** Hồ Đức Toàn (Mục 5 – Heuristic Rollout)
- **Nhánh phát triển:** `feature/heuristic-rollout`
- **Các file trong phạm vi Task 5:**
  - `tictactoe/rollout.py`: Triển khai `heuristic_rollout` theo quy trình 5 bước đơn giản hóa dựa trên 3 tiêu chí (**Position**, **Attack**, **Defense**) và bảng trọng số 6 mức; giữ nguyên `random_rollout` và `ROLLOUT_POLICIES`.
  - `tictactoe/test_rollout.py`: Bộ 23 unit test kiểm thử toàn diện cho 7 nhóm tình huống chiến thuật của `rollout.py`, tích hợp MCTS và benchmark.
  - `tictactoe/benchmark_rollout.py`: Script thực nghiệm đối đầu AI vs AI giữa `Random Rollout` và `Heuristic Rollout`.

---

## 1. Mục Tiêu, Vai Trò Trong Pha Simulation & Phân Biệt Với MCTS Tại Root

### 1.1. Vai trò của Heuristic Rollout trong Simulation
Trong thuật toán **Monte Carlo Tree Search (MCTS)**, mỗi vòng lặp (`_iterate`) gồm 4 giai đoạn: **Selection (UCT)** $\rightarrow$ **Expansion** $\rightarrow$ **Simulation (Rollout)** $\rightarrow$ **Backpropagation**.
- Ở giai đoạn **Simulation**, nếu sử dụng `random_rollout` (chọn ngẫu nhiên đồng đều trên toàn bàn cờ), các ván chơi giả lập trên bàn cờ lớn ($10\times 10, 15\times 15$) thường đi vào các ô xa vô nghĩa, bỏ qua cả nước thắng ngay trước mắt lẫn nước chặn đối thủ, khiến kết quả thắng/thua (`reward` $\in \{1.0, 0.5, 0.0\}$) truyền ngược về cây bị nhiễu.
- `heuristic_rollout` thay thế việc chọn ngẫu nhiên hoàn toàn bằng một bộ quy tắc đánh giá thế cờ cục bộ đơn giản, giúp quá trình mô phỏng diễn ra hợp lý hơn và cung cấp ước lượng xác suất thắng ($W/N$) sát thực tế hơn cho cây tìm kiếm MCTS.
- **Liên hệ lý thuyết (Tiền thân của Value Network):** Heuristic Rollout có sự tương đồng về mục tiêu với **Value Network** trong các hệ thống AI hiện đại (như AlphaGo/AlphaZero) ở chỗ cả hai đều nhằm cung cấp thông tin đánh giá chất lượng trạng thái cờ cho quá trình tìm kiếm cây, dù một bên là hàm heuristic thiết kế thủ công dẫn hướng mô phỏng còn một bên là mạng nơ-ron được huấn luyện để dự đoán trực tiếp giá trị trạng thái.

### 1.2. Phân biệt Rollout Policy với quyết định nước đi thực tế tại Root của MCTS
- **Nước đi trong Simulation (`heuristic_rollout`):** Chỉ được gọi ở Pha 3 (`Simulation`) trên bản sao bàn cờ giả lập `sim` để dẫn dắt ván cờ mô phỏng tới khi kết thúc (`sim.is_terminal()`).
- **Nước đi thực tế trên bàn cờ (`MCTSAgent.get_move`):** Sau khi hoàn thành `iterations` vòng lặp mô phỏng và cập nhật thống kê $(W, N)$ qua Backpropagation, MCTS chọn nước đi thực tế tại **Root** dựa trên nhánh con có **số lần thăm $N$ lớn nhất** (`max_n`, kết hợp ưu tiên nước thắng/chặn trực tiếp tại Root). Heuristic Rollout không trực tiếp quyết định nước đi cuối cùng của MCTS tại Root và không đảm bảo MCTS luôn chọn nước đi tối ưu tuyệt đối trong mọi thế cờ.

---

## 2. Thiết Kế Thuật Toán Heuristic Rollout Đơn Giản Hóa

### 2.1. Ba tiêu chí đánh giá
1. **Position (Vị trí – `_get_candidate_moves`):**
   - Bàn cờ nhỏ ($\le 3 \times 3$): xét toàn bộ các ô trống hợp lệ.
   - Bàn cờ trống: chọn ô trung tâm `(board_size // 2, board_size // 2)`.
   - Bàn cờ lớn ($> 3 \times 3$): chỉ xét các ô trống nằm sát các quân đã có trong bán kính Chebyshev `radius = 1` để loại bỏ các nước đi rời rạc ở xa khu vực giao tranh.
2. **Attack (Tấn công):** Đánh giá khả năng nối dài chuỗi quân của người chơi hiện tại (`player`) trên 4 trục (ngang, dọc, chéo chính, chéo phụ) dựa trên tổng số quân liên tiếp tạo thành (`count`) và số đầu mở (`open_ends`).
3. **Defense (Phòng thủ):** Đánh giá mức độ nguy hiểm nếu đối thủ (`opponent = -player`) đặt quân vào ô ứng viên đó. Thuật toán giả định đối thủ đặt quân tại ô ứng viên, sau đó đếm độ dài chuỗi đối thủ tạo ra (`def_count`) và số đầu mở còn lại của chuỗi mới (`def_open`).

### 2.2. Quy trình 5 bước của `heuristic_rollout(game)`
1. **Bước 1 – Lọc ứng viên (Position):** Gọi `candidates = _get_candidate_moves(game)`.
2. **Bước 2 – Thắng ngay:** Kiểm tra `_is_winning_move(game, move, player)` trên `candidates`. Nếu có nước thắng ngay, trả về `random.choice(winning_moves)`.
3. **Bước 3 – Chặn đối thủ thắng ngay:** Kiểm tra `_is_winning_move(game, move, opponent)`. Nếu có 1 ô chặn thì trả về ngay; nếu có nhiều ô chặn thì chấm điểm `_evaluate_move` trên tập ô chặn để chọn ô có điểm cao nhất (chọn ngẫu nhiên nếu đồng điểm).
4. **Bước 4 – Đánh giá chiến thuật (Attack + Defense):** Nếu chưa có nước thắng/chặn ngay, chấm điểm từng ô trong `candidates` bằng `_evaluate_move(game, move, player)` trên 4 hướng (không cộng điểm cho hướng hoàn toàn trống và không dùng `center_bonus`).
5. **Bước 5 – Chọn nước đi (Tie-breaking ngẫu nhiên):** Thu thập tất cả ứng viên cùng đạt điểm cao nhất `best_score` vào `best_moves` và trả về `random.choice(best_moves)` (yếu tố ngẫu nhiên chỉ xảy ra khi có từ 2 ứng viên đồng điểm cao nhất trở lên).

### 2.3. Giải thích `count`, `open_ends` và Bảng trọng số 6 mức
Trong `_analyze_direction(game, row, col, dr, dc, target_player)`:
- `count = 1 + forward_count + backward_count`: Tổng số quân liên tiếp cùng loại với `target_player` trên trục `(dr, dc)` khi giả định đặt thêm 1 quân tại `(row, col)`.
- `open_ends = forward_open + backward_open` ($\in \{0, 1, 2\}$): Số ô ngay sát hai đầu chuỗi liên tiếp đó nằm trong biên bàn cờ và đang trống (`EMPTY`). Ô nằm ngoài biên bàn cờ hoặc đang có quân đối phương không được tính là đầu mở.

| Mẫu hình chuỗi quân | Khóa `(count, open_ends)` | Điểm Tấn công (`ATTACK_WEIGHTS`) | Điểm Phòng thủ (`DEFENSE_WEIGHTS`) | Ý nghĩa chiến thuật |
|---|:---:|---:|---:|---|
| Tứ mở 2 đầu | `(4, 2)` | `10,000` | `8,000` | Chuỗi 4 quân còn trống cả 2 đầu |
| Tứ mở 1 đầu | `(4, 1)` | `2,000` | `1,500` | Chuỗi 4 quân mở 1 đầu (đe dọa thắng ở lượt kế / hạn chế chuỗi 3 đối thủ lên 4) |
| Tam mở 2 đầu | `(3, 2)` | `800` | `600` | Chuỗi 3 quân mở 2 đầu có tiềm năng phát triển mạnh |
| Tam mở 1 đầu | `(3, 1)` | `200` | `120` | Bước đệm phát triển chuỗi 2 quân lên 3 quân mở 1 đầu |
| Nhị mở 2 đầu | `(2, 2)` | `60` | `30` | Khởi tạo chuỗi 2 quân thông thoáng 2 đầu |
| Nhị mở 1 đầu | `(2, 1)` | `15` | `10` | Liên kết 2 quân cơ bản bị chặn 1 đầu |
| Chuỗi 1 quân / Bị bịt 2 đầu | `count < 2` hoặc `open_ends == 0` | `0` | `0` | Không tạo chuỗi hoặc không còn đầu mở để phát triển |

### 2.4. Ví dụ minh họa tính điểm trên bàn cờ 5×5 và cân bằng Công – Thủ
- **Ví dụ bàn cờ 5×5 ($k=5$, đến lượt X):** X có 2 quân tại `(2, 1), (2, 2)`; O có 2 quân tại `(3, 2), (3, 3)`:
  - **Ứng viên A = `(2, 3)`:** Hướng Ngang nối X thành `(3, 2) = 800` (Attack); hướng Dọc cản O tại `(3, 3)` thành `(2, 2) = 30` (Defense); hướng Chéo phụ cản O tại `(3, 2)` thành `(2, 2) = 30` (Defense); hướng Chéo chính = `0` $\Rightarrow$ **Tổng điểm A = `860` điểm**.
  - **Ứng viên B = `(3, 1)`:** Hướng Ngang cản O thành `(3, 2) = 600` (Defense); hướng Dọc nối X tại `(2, 1)` thành `(2, 2) = 60` (Attack); hướng Chéo phụ nối X tại `(2, 2)` thành `(2, 2) = 60` (Attack); hướng Chéo chính = `0` $\Rightarrow$ **Tổng điểm B = `720` điểm**.
  - Khi O mới có 2 quân, heuristic chọn **A `(2, 3)` (`860 > 720`)** để phát triển chuỗi 2 → 3 cho X.
  - Khi O có 3 quân `(3, 1), (3, 2), (3, 3)`: hai đầu trống là **C `(3, 0)`** và **D `(3, 4)`**. Khi giả định O đặt vào `(3, 0)` (hoặc `(3, 4)`), chuỗi 4 quân mới chạm biên bàn cờ 5×5 ở một phía nên chỉ còn 1 đầu mở $\rightarrow$ mẫu Defense `(4, 1) = 1,500` điểm. Tại C `(3, 0)` có thêm `+15` điểm Attack chéo phụ `(2, 1)` với `(2, 1)` nên đạt **`1,515` điểm** (cao hơn D `(3, 4) = 1,500` và A `(2, 3) = 860`).
- **Tình huống cân bằng Công – Thủ trên bàn 10×10:**
  - *Phát triển 2 → 3 thắng cụm quân lẻ:* X có `(5, 3), (5, 4)` (chặn trái bởi O ở `(5, 2)`), đồng thời có cụm quân lẻ O ở `(3, 6), (4, 7)` và X ở `(2, 7)`. Nước nối chuỗi `(5, 5)` đạt `(3, 1) = 200` điểm, cao hơn ô giao điểm cụm quân lẻ `(3, 7)` (`Attack dọc (2, 1) = 15` + `Defense dọc (2, 1) = 10` + `Defense ngang (2, 2) = 30` = `55` điểm).
  - *Chiếm đầu trống của chuỗi 3 quân đối thủ:* X có `(4, 4), (4, 5)` (nối lên `(3, 2)` được `800` điểm), còn O có 3 quân `(6, 4), (6, 5), (6, 6)` nằm giữa hai quân X ở `(6, 2)` và `(6, 8)`. Nước `(6, 3)` hoặc `(6, 7)` đạt `Defense (4, 1) = 1,500` + `Attack (2, 1) = 15` = `1,515` điểm (`> 800`), nên được ưu tiên theo trọng số heuristic.

### 2.5. Giới hạn của thiết kế Heuristic
- Bảng trọng số 6 mức từ chuỗi 2 đến chuỗi 4 được thiết kế cho luật thắng 5 quân (`win_length = 5`) và tương thích với bàn $3\times 3$ (`win_length = 3` thông qua `_is_winning_move` và mức chuỗi 2), không nhằm hỗ trợ tổng quát cho `win_length > 5`.
- Chỉ đếm chuỗi quân liên tiếp liền kề (không nhận diện mẫu đứt đoạn như `X_X` hay `XX_X` và không quét độ rộng hành lang xa).
- Khi đối thủ đã có $\ge 2$ điểm thắng ngay đồng thời mà người chơi hiện tại chưa có nước thắng ngay, mỗi lượt chỉ chặn được 1 đầu nên không thể đảm bảo cứu được ván đấu.

---

## 3. Kết Quả Kiểm Thử (Unit Tests) & Thực Nghiệm Benchmark

- **Unit Tests:** Chạy `python3 -m unittest -v tictactoe.test_rollout` đạt **23/23 tests PASS** (`0.587s`). Chạy toàn bộ test suite (`test_rollout`, `test_mcts`, `test_game`, `test_minimax`) đạt **102/102 tests PASS** (`OK, skipped=6`).
- **Benchmark AI vs AI:** Chạy thực tế qua `python3 -m tictactoe.benchmark_rollout --games 20 --seed 100` (330 ván đấu qua 18 cấu hình, $c = 1.4$, luân phiên X/O 50/50 theo cặp seed):

### 3.1. Nhóm 1: So sánh trong CÙNG SỐ VÒNG LẶP (Equal Iterations)

| Bàn cờ ($k$) | Số ván | Iterations (H / R) | Heuristic Thắng (W) | Hòa (D) | Heuristic Thua (L) | Chi tiết X / O (W-D-L) | Heuristic Win Rate | Random Win Rate | TG TB/nước H | TG TB/nước R | TG TB/ván |
|---|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| **3×3 ($k=3$)** | 20 | 50 / 50 | **7** | 10 | 3 | X: 7-2-1 \| O: 0-8-2 | **35.0%** | 15.0% | 0.0015s | 0.0008s | 0.009s |
| **3×3 ($k=3$)** | 20 | 100 / 100 | **7** | 12 | 1 | X: 6-4-0 \| O: 1-8-1 | **35.0%** | 5.0% | 0.0027s | 0.0015s | 0.016s |
| **3×3 ($k=3$)** | 20 | 200 / 200 | **2** | 18 | 0 | X: 2-8-0 \| O: 0-10-0 | **10.0%** | 0.0% | 0.0047s | 0.0029s | 0.030s |
| **5×5 ($k=5$)** | 20 | 50 / 50 | 0 | **20** | 0 | X: 0-10-0 \| O: 0-10-0 | 0.0% (Hòa 100%) | 0.0% | 0.0220s | 0.0025s | 0.294s |
| **5×5 ($k=5$)** | 20 | 100 / 100 | 0 | **20** | 0 | X: 0-10-0 \| O: 0-10-0 | 0.0% (Hòa 100%) | 0.0% | 0.0424s | 0.0049s | 0.568s |
| **10×10 ($k=5$)** | 20 | 30 / 30 | **19** | 0 | 1 | X: 9-0-1 \| O: 10-0-0 | **95.0%** | 5.0% | 0.1069s | 0.0100s | 2.883s |
| **10×10 ($k=5$)** | 20 | 50 / 50 | **20** | 0 | 0 | X: 10-0-0 \| O: 10-0-0 | **100.0%** | 0.0% | 0.1766s | 0.0168s | 3.705s |
| **10×10 ($k=5$)** | 20 | 100 / 100 | **19** | 0 | 1 | X: 10-0-0 \| O: 9-0-1 | **95.0%** | 5.0% | 0.3257s | 0.0341s | 5.831s |
| **15×15 ($k=5$)** | 10 | 30 / 30 | **8** | 0 | 2 | X: 5-0-0 \| O: 3-0-2 | **80.0%** | 20.0% | 0.1403s | 0.0302s | 4.152s |
| **15×15 ($k=5$)** | 10 | 50 / 50 | **10** | 0 | 0 | X: 5-0-0 \| O: 5-0-0 | **100.0%** | 0.0% | 0.2696s | 0.0530s | 7.908s |

### 3.2. Nhóm 2: Khảo sát GIẢM SỐ VÒNG LẶP của Heuristic so với Random (Bàn 10×10, $k=5$)

| Cấu hình đối đầu | Số ván | Heuristic Iter | Random Iter | Tỷ lệ giảm Iter | H_W | D | H_L | Chi tiết X / O (W-D-L) | Heuristic Win Rate | TG TB/nước H | TG TB/nước R |
|---|---:|---:|---:|---|---:|---:|---:|:---:|---:|---:|---:|
| Heuristic 25 vs Random 50 | 20 | **25** | 50 | Giảm 2× (50%) | **18** | 1 | 1 | X: 9-0-1 \| O: 9-1-0 | **90.0%** | 0.0925s | 0.0164s |
| Heuristic 25 vs Random 100 | 20 | **25** | 100 | Giảm 4× (75%) | **19** | 0 | 1 | X: 9-0-1 \| O: 10-0-0 | **95.0%** | 0.1018s | 0.0338s |
| Heuristic 25 vs Random 200 | 20 | **25** | 200 | Giảm 8× (87.5%) | **14** | 0 | 6 | X: 8-0-2 \| O: 6-0-4 | **70.0%** | 0.0852s | 0.0647s |
| Heuristic 50 vs Random 200 | 20 | **50** | 200 | Giảm 4× (75%) | **19** | 0 | 1 | X: 9-0-1 \| O: 10-0-0 | **95.0%** | 0.1574s | 0.0659s |

### 3.3. Nhóm 3: So sánh công bằng trong CÙNG NGÂN SÁCH THỜI GIAN mỗi nước đi (Equal Time Budget)

| Bàn cờ ($k$) | Ngân sách TG / nước | Số ván | Iter TB đạt được (H / R) | H_W | D | H_L | Chi tiết X / O (W-D-L) | Heuristic Win Rate | Random Win Rate | TG thực tế H | TG thực tế R |
|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|
| **5×5 ($k=5$)** | 0.030s / nước | 20 | 534.8 / 967.8 | 0 | **20** | 0 | X: 0-10-0 \| O: 0-10-0 | 0.0% (Hòa 100%) | 0.0% | 0.0303s | 0.0301s |
| **10×10 ($k=5$)** | 0.050s / nước | 20 | 79.7 / 160.5 | **17** | 0 | 3 | X: 10-0-0 \| O: 7-0-3 | **85.0%** | 15.0% | 0.0526s | 0.0502s |
| **10×10 ($k=5$)** | 0.100s / nước | 20 | 177.3 / 303.7 | **16** | 0 | 4 | X: 9-0-1 \| O: 7-0-3 | **80.0%** | 20.0% | 0.1033s | 0.1003s |
| **15×15 ($k=5$)** | 0.100s / nước | 10 | 96.8 / 116.8 | **10** | 0 | 0 | X: 5-0-0 \| O: 5-0-0 | **100.0%** | 0.0% | 0.1062s | 0.1006s |

---

## 4. Nhận Xét & Kết Luận

1. **Hiệu quả rõ rệt trên bàn cờ Caro rộng ($10\times 10$ và $15\times 15$, $k=5$):** Phiên bản Heuristic Rollout đơn giản hóa đạt tỷ lệ thắng **80% – 100%** khi cùng số vòng lặp và **80% – 100%** khi giới hạn trong cùng ngân sách thời gian mỗi nước đi (`0.05s` và `0.10s`).
2. **Duy trì ưu thế khi giảm số vòng lặp:** Trên bàn $10\times 10$, Heuristic Rollout chỉ dùng **25 iterations** vẫn thắng **90% – 95%** trước Random Rollout dùng **50 – 100 iterations**, và thắng **70%** trước Random Rollout dùng **200 iterations** (giảm 8 lần số vòng lặp).
3. **Trên bàn cờ nhỏ và hẹp ($3\times 3$ và $5\times 5$):** Trên bàn $3\times 3$ ($k=3$), Heuristic Rollout thắng nhiều hơn và chỉ thua 0–3 ván khi số vòng lặp thấp, tiến dần về hòa khi số vòng lặp tăng. Trên bàn $5\times 5$ ($k=5$), do kích thước bàn đúng bằng độ dài chuỗi thắng nên rất dễ bị khóa hòa, cả 3 cấu hình đều kết thúc với **100% ván hòa (20/20 ván hòa, 0 ván thua)**.
