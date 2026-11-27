# Báo Cáo Mục 5: Mở Rộng MCTS Với Heuristic Rollout

- **Người thực hiện:** Hồ Đức Toàn (Mục 5 – Heuristic Rollout)
- **Nhánh phát triển:** `feature/heuristic-rollout`
- **Các file triển khai:**
  - `tictactoe/rollout.py`: Triển khai `heuristic_rollout` và các hàm đánh giá chuỗi quân cục bộ, giữ nguyên `random_rollout` và `ROLLOUT_POLICIES`.
  - `tictactoe/test_rollout.py`: Bộ 16 unit test kiểm thử toàn diện cho `rollout.py`, tích hợp MCTS và benchmark.
  - `tictactoe/benchmark_rollout.py`: Script thực nghiệm đối đầu AI vs AI giữa `Random Rollout` và `Heuristic Rollout`.

---

## 1. Thiết Kế Thuật Toán Heuristic Rollout

Trong pha **Simulation (Rollout)** của MCTS (`MCTSAgent._iterate`), thay vì chọn nước đi ngẫu nhiên trên toàn bàn cờ (`random_rollout`), hàm `heuristic_rollout(game)` lựa chọn nước đi theo quy trình 3 bước ưu tiên mà **không làm biến đổi trạng thái `game` truyền vào** (Zero-Copy):

1. **Lọc miền ứng viên cục bộ (`_get_candidate_moves`):**
   - Bàn cờ nhỏ ($\le 3 \times 3$): xét toàn bộ các ô trống hợp lệ.
   - Bàn cờ rỗng: chọn ô trung tâm `(board_size // 2, board_size // 2)`.
   - Bàn cờ lớn ($> 3 \times 3$): chỉ xét các ô trống nằm trong bán kính Chebyshev `radius = 1` tiếp giáp với các quân cờ đã có trên bàn (vì mọi nước nối chuỗi $\ge 2$ quân hoặc chặn đầu chuỗi của đối thủ đều bắt buộc tiếp giáp với ít nhất một quân cờ trên bàn).
2. **Thứ tự ưu tiên chiến thuật:**
   - **Ưu tiên 1 – Thắng ngay lập tức (`_is_winning_move(game, move, player)`):** Nếu có nước đi tạo đủ $\ge k$ quân liên tiếp cho người chơi hiện tại, chọn ngay nước đó.
   - **Ưu tiên 2 – Chặn đối thủ thắng ngay (`_is_winning_move(game, move, opponent)`):** Nếu đối thủ có nước đi thắng ngay ở lượt kế tiếp, ưu tiên chặn ngay. Trường hợp đối thủ đồng thời có nhiều mối đe dọa thắng ngay (`len(blocking_moves) > 1`), thuật toán chấm điểm `_evaluate_move` cho từng ô trong `blocking_moves` và chọn ô chặn mang lại giá trị kết hợp (phản công / chắn giao điểm) cao nhất.
   - **Ưu tiên 3 – Tối đa hóa điểm chuỗi quân Tấn công & Phòng thủ (`_evaluate_move`):** Chấm điểm tất cả các ô ứng viên trên cả 4 hướng (ngang, dọc, 2 đường chéo) kèm điểm thưởng nhỏ cho vị trí gần tâm bàn cờ (`center_bonus` $\in [0, 3]$), sau đó chọn ngẫu nhiên trong nhóm các nước đi đạt điểm cao nhất.

### Bảng trọng số Tấn công (`ATTACK_WEIGHTS`) & Phòng thủ (`DEFENSE_WEIGHTS`) (chuẩn hóa cho $k = 5$)

| Mẫu hình chuỗi quân | Khóa `(count, open_ends)` | Điểm Tấn công (`player`) | Điểm Phòng thủ (`opponent`) | Ý nghĩa chiến thuật |
|---|:---:|---:|---:|---|
| Đủ chuỗi thắng ($\ge k$) | `count >= k` | `100,000` | `90,000` | Thắng ngay / Chặn thua ngay |
| Tứ mở 2 đầu (Open 4) | `(4, 2)` | `30,000` | `25,000` | Tạo 2 điểm thắng ở lượt kế tiếp |
| Tứ mở 1 đầu (Half-open 4) | `(4, 1)` | `3,500` | `2,800` | Nước chiếu trực tiếp buộc đối phương phải đỡ |
| Tam mở 2 đầu (Open 3) | `(3, 2)` | `1,500` | `1,200` | Đe dọa tạo Tứ mở 2 đầu ở lượt kế tiếp |
| Tam mở 1 đầu (Half-open 3) | `(3, 1)` | `150` | `120` | Chuỗi 3 quân có khả năng phát triển lên 4 |
| Nhị mở 2 đầu (Open 2) | `(2, 2)` | `50` | `40` | Khung phát triển chuỗi sớm |
| Nhị mở 1 đầu (Half-open 2) | `(2, 1)` | `10` | `8` | Liên kết 2 quân cơ bản |
| Đơn lẻ mở 2 phía | `(1, 2)` | `2` | `1` | Ô trống có không gian thoáng 2 bên |
| Bị bịt kín / Hành lang $< k$ | `open_ends == 0` | `0` | `0` | Chuỗi chết, không đủ không gian tạo $k$ quân |

*(Nhờ cộng tổng điểm trên cả 4 hướng đi qua ô ứng viên, các **giao điểm Đa đe dọa / Fork** như Đôi Tam mở 2 đầu đạt `1,500 + 1,500 = 3,000` điểm công hoặc `1,200 + 1,200 = 2,400` điểm thủ; trong khi nước phát triển hàng 3 lên hàng 4 mở 1 đầu `(4, 1) = 3,500` luôn cao hơn một hướng Tam mở 2 đầu đơn lẻ kèm các hướng nhỏ. Đồng thời tại nút gốc của `MCTSAgent.get_move()`, thuật toán cũng kiểm tra và ưu tiên trực tiếp nước thắng ngay hoặc chặn đối thủ thắng ngay trên toàn bộ `valid_moves`).*

---

## 2. Kết Quả Thực Nghiệm AI vs AI (`MCTS Heuristic Rollout` vs `MCTS Random Rollout`)

Thực nghiệm được chạy qua `python -m tictactoe.benchmark_rollout --games 20 --seed 100` (330 ván đấu qua 18 cấu hình, hệ số UCT $c = 1.4$, luân phiên cầm X/O công bằng 50/50 theo từng cặp ván cùng seed):

### 2.1. Nhóm 1: So sánh trong CÙNG SỐ VÒNG LẶP (Equal Iterations)

| Bàn cờ ($k$) | Số ván | Iterations (H / R) | Heuristic Thắng (W) | Hòa (D) | Heuristic Thua (L) | Chi tiết X / O (W-D-L) | Heuristic Win Rate | Random Win Rate | TG TB/nước H | TG TB/nước R | TG TB/ván |
|---|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| **3×3 ($k=3$)** | 20 | 50 / 50 | **9** | 9 | 2 | X: 8-2-0 \| O: 1-7-2 | **45.0%** | 10.0% | 0.0017s | 0.0008s | 0.009s |
| **3×3 ($k=3$)** | 20 | 100 / 100 | **3** | 16 | 1 | X: 2-8-0 \| O: 1-8-1 | **15.0%** | 5.0% | 0.0028s | 0.0014s | 0.017s |
| **3×3 ($k=3$)** | 20 | 200 / 200 | **4** | 15 | 1 | X: 4-6-0 \| O: 0-9-1 | **20.0%** | 5.0% | 0.0054s | 0.0029s | 0.031s |
| **5×5 ($k=5$)** | 20 | 50 / 50 | **5** | 14 | 1 | X: 4-6-0 \| O: 1-8-1 | **25.0%** | 5.0% | 0.0266s | 0.0026s | 0.311s |
| **5×5 ($k=5$)** | 20 | 100 / 100 | 1 | 16 | 3 | X: 1-8-1 \| O: 0-8-2 | 5.0% | **15.0%** | 0.0519s | 0.0052s | 0.634s |
| **10×10 ($k=5$)** | 20 | 30 / 30 | **20** | 0 | 0 | X: 10-0-0 \| O: 10-0-0 | **100.0%** | 0.0% | 0.1240s | 0.0093s | 2.762s |
| **10×10 ($k=5$)** | 20 | 50 / 50 | **20** | 0 | 0 | X: 10-0-0 \| O: 10-0-0 | **100.0%** | 0.0% | 0.1697s | 0.0149s | 4.238s |
| **10×10 ($k=5$)** | 20 | 100 / 100 | **20** | 0 | 0 | X: 10-0-0 \| O: 10-0-0 | **100.0%** | 0.0% | 0.3435s | 0.0316s | 6.962s |
| **15×15 ($k=5$)** | 10 | 30 / 30 | **10** | 0 | 0 | X: 5-0-0 \| O: 5-0-0 | **100.0%** | 0.0% | 0.1677s | 0.0285s | 5.419s |
| **15×15 ($k=5$)** | 10 | 50 / 50 | **10** | 0 | 0 | X: 5-0-0 \| O: 5-0-0 | **100.0%** | 0.0% | 0.3554s | 0.0501s | 9.667s |

### 2.2. Nhóm 2: Khảo sát GIẢM SỐ VÒNG LẶP của Heuristic so với Random (Bàn 10×10, $k=5$)

| Cấu hình đối đầu | Số ván | Heuristic Iter | Random Iter | Tỷ lệ giảm Iter | H_W | D | H_L | Chi tiết X / O (W-D-L) | Heuristic Win Rate | TG TB/nước H | TG TB/nước R |
|---|---:|---:|---:|---|---:|---:|---:|:---:|---:|---:|---:|
| Heuristic 25 vs Random 50 | 20 | **25** | 50 | Giảm 2× (50%) | **19** | 0 | 1 | X: 9-0-1 \| O: 10-0-0 | **95.0%** | 0.0934s | 0.0164s |
| Heuristic 25 vs Random 100 | 20 | **25** | 100 | Giảm 4× (75%) | **18** | 0 | 2 | X: 10-0-0 \| O: 8-0-2 | **90.0%** | 0.0951s | 0.0311s |
| Heuristic 25 vs Random 200 | 20 | **25** | 200 | Giảm 8× (87.5%) | **13** | 0 | 7 | X: 6-0-4 \| O: 7-0-3 | **65.0%** | 0.0947s | 0.0647s |
| Heuristic 50 vs Random 200 | 20 | **50** | 200 | Giảm 4× (75%) | **18** | 0 | 2 | X: 9-0-1 \| O: 9-0-1 | **90.0%** | 0.2044s | 0.0653s |

### 2.3. Nhóm 3: So sánh công bằng trong CÙNG NGÂN SÁCH THỜI GIAN mỗi nước đi (Equal Time Budget)

| Bàn cờ ($k$) | Ngân sách TG / nước | Số ván | Iter TB đạt được (H / R) | H_W | D | H_L | Chi tiết X / O (W-D-L) | Heuristic Win Rate | Random Win Rate | TG thực tế H | TG thực tế R |
|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|
| **5×5 ($k=5$)** | 0.030s / nước | 20 | 510.8 / 974.6 | 0 | 19 | 1 | X: 0-10-0 \| O: 0-9-1 | 0.0% (Hòa 95%) | **5.0%** | 0.0303s | 0.0300s |
| **10×10 ($k=5$)** | 0.050s / nước | 20 | 81.1 / 163.2 | **20** | 0 | 0 | X: 10-0-0 \| O: 10-0-0 | **100.0%** | 0.0% | 0.0534s | 0.0503s |
| **10×10 ($k=5$)** | 0.100s / nước | 20 | 190.7 / 315.5 | **17** | 0 | 3 | X: 10-0-0 \| O: 7-0-3 | **85.0%** | 15.0% | 0.1031s | 0.1003s |
| **15×15 ($k=5$)** | 0.100s / nước | 10 | 104.4 / 112.3 | **10** | 0 | 0 | X: 5-0-0 \| O: 5-0-0 | **100.0%** | 0.0% | 0.1080s | 0.1006s |

---

## 3. Nhận Xét & Kết Luận

1. **Cải thiện vượt trội tỷ lệ thắng trên bàn cờ Caro 5 quân (10×10 và 15×15):** Heuristic Rollout thắng tuyệt đối **100%** khi cùng số vòng lặp, và thắng **85% – 100%** khi giới hạn trong cùng ngân sách thời gian mỗi nước đi (`0.05s` và `0.10s`).
2. **Duy trì chất lượng chơi khi giảm sâu số vòng lặp:** Trên bàn 10×10, Heuristic Rollout chỉ cần **25 iterations** vẫn đạt tỷ lệ thắng **90% – 95%** trước Random Rollout dùng **50 – 100 iterations** (giảm 2–4 lần vòng lặp), và thắng **65%** ngay cả khi Random Rollout dùng **200 iterations** (giảm 8 lần vòng lặp).
3. **Đánh đổi giữa tốc độ và chất lượng tìm kiếm:** Một vòng lặp của Heuristic Rollout tốn thời gian hơn Random Rollout do phải phân tích 4 hướng xung quanh ô ứng viên; tuy nhiên nhờ kỹ thuật quét trực tiếp trên ma trận (Zero-Copy) và lọc ứng viên bán kính Chebyshev `radius = 1`, thời gian mỗi nước đi trên bàn 15×15 (30–50 iterations) chỉ mất `0.168s – 0.355s` (nhanh hơn ~50 lần so với bản khung ban đầu). Trên bàn cờ siêu nhỏ/hẹp như 5×5 ($k=5$) với số vòng lặp rất lớn ($\sim 500\text{–}1000$ iterations), 90%–95% ván đấu kết thúc hòa và Random Rollout có ưu điểm nhẹ nhờ tốc độ lấy mẫu nhanh và không bị thiên lệch (unbiased).
