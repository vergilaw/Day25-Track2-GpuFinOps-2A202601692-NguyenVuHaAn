# Write-up — Lab 25: GPU FinOps Optimization

**Họ và tên:** Nguyễn Vũ Hà An · **ID:** 2A202601692 · **Ngày:** 2026-08-27

Toàn bộ số liệu dưới đây được sinh ra bởi `python missions/run_all.py` và trùng khớp với
`outputs/report.md` + `outputs/savings.png`. Kiểm chứng: `python verify.py` → 11/11,
`pytest -q` → 40 passed (15 test có sẵn + 25 test tôi viết thêm cho phần extension).

---

## 1. Baseline vs. Optimized

| Chỉ số | Baseline | Optimized | Thay đổi |
|---|---|---|---|
| **Tổng chi phí fleet / tháng** | **$27,133** | **$14,422** | **−46.8%** (−$12,711/mo) |
| Inference $/1M token | $6.488 | $1.251 | −80.7% |
| Inference $/ngày | $48.87 | $9.42 | −80.7% |
| Purchasing (GPU fleet) / tháng | $25,667 | $15,879 | −38.1% |

Tổng tiết kiệm **46.8% (~$12,711/tháng, ~$152K/năm)**.

`$/1M-token` là con số tôi coi là chính, không phải tổng USD: nó **sống sót qua thay đổi quy mô**.
Nếu traffic tăng gấp đôi, tổng chi phí tăng gấp đôi và trông như một thất bại FinOps, trong khi
$6.488 → $1.251 vẫn đúng. Đây chính là lý do deck bắt chuyển đơn vị kế toán từ `$/GPU-hour`
sang `$/1M-token`.

**Một giới hạn tôi phải nói rõ:** `gpu_telemetry.csv` (11 GPU) và `workloads.csv` (8 job) là hai
*view* khác nhau, không được đối chiếu về cùng một fleet vật lý, nên 4 đòn bẩy được cộng như thể
độc lập. Thực tế chúng giao nhau (purchasing sẽ áp lên giá *sau khi* right-size). Vì vậy **con số
từng đòn bẩy là con số bảo vệ được, còn tổng 46.8% là giới hạn trên.** Tôi đã sửa một trường hợp
double-count cụ thể: `gpu-h100-5` vừa là GPU idle 8h/ngày vừa là ứng viên right-size, nên phần
right-size của nó chỉ được tính trên 16 active hours thay vì 24 (giảm $170/mo so với cách tính
ngây thơ).

---

## 2. Phân tích từng đòn bẩy

| Đòn bẩy | $/tháng | % tiết kiệm | Effort | Payback |
|---|---|---|---|---|
| **Purchasing (spot / reserved)** | **$9,788** | **77%** | 15 ngày | 1.8 tháng |
| Inference (cascade / cache / batch) | $1,183 | 9% | 10 ngày | 10.1 tháng |
| Right-size GPU nghẽn bandwidth | $1,140 | 9% | 30 ngày | 31.6 tháng |
| Tắt GPU idle | $600 | 5% | 2 ngày | 4.0 tháng |

**Purchasing thắng, vì hai lý do khác nhau về bản chất:**

1. **Cơ sở lớn hơn một bậc.** Fleet GPU là $25,667/mo; toàn bộ inference chỉ $1,466/mo. Cắt 38%
   của cái lớn ăn đứt cắt 81% của cái nhỏ. Bài học: *đòn bẩy lớn nhất là đòn bẩy nằm trên cơ sở
   lớn nhất* — phải đo cơ sở trước khi tối ưu.
2. **Giảm giá ở đây là cấu trúc, không phụ thuộc hiệu suất.** Spot/reserved cho **cùng một con
   silicon** với giá thấp hơn 34–60%; không cần cải thiện gì về kỹ thuật, chỉ cần checkpointing và
   một quyết định cam kết.

**Bên trong inference, cascade chiếm gần như toàn bộ.** Phân rã tăng dần theo thứ tự ship:
cascade $37.40/ngày, batch $1.82, cache $0.23 → **cascade là 95% của đòn bẩy**. Nếu đo từng cái
*áp dụng riêng lẻ* thì cascade $37.40, batch $8.24, cache $3.51 — batch và cache trông lớn hơn
nhiều. Khác biệt này là điểm quan trọng nhất của mission 2: **discount nhân với nhau trên một cơ
sở đang co lại**. Sau khi cascade đã đẩy 80% traffic sang model rẻ hơn 15×, batch và cache chỉ còn
rất ít để giảm. Hệ quả thực tế: **ship router trước, đàm phán discount sau** — không phải ngược
lại như phản xạ thường thấy.

---

## 3. GPU-Util Lie

**GPU nào bị lie:** `gpu-h100-4` (**98.2%** GPU-Util, MFU **0.194**) và `gpu-a10g-1` (96.9%,
MFU 0.268). Ngưỡng: util > 90% nhưng MFU < 30%.

**Cơ chế — hai con số đo hai thứ khác nhau.** `nvidia-smi utilization.gpu` là **duty-cycle
counter**: tỉ lệ sampling interval có *ít nhất một* kernel đang resident trên device. Nó không nói
gì về việc kernel đó dùng bao nhiêu phần của device — nên **một kernel nhỏ chiếm 1 SM suốt cả
interval vẫn báo 100%**. MFU là **throughput ratio**: FLOP/s đạt được trên FLOP/s đỉnh mà ta đã
trả tiền. Một câu hỏi là "GPU có đang bận?", câu kia là "GPU có đang làm việc tôi trả tiền cho?"
— và chỉ câu thứ hai có dấu đô-la trong đó.

**Tác động tài chính.** `gpu-h100-4` giao 192 / 990 TFLOP/s ở $2.50/hr. 81% của rate đó không mua
được gì: **~$1,451/tháng trên một GPU duy nhất** trả cho năng lực tính toán không bao giờ được
dùng. Device gần như luôn occupied và idle *bên trong* gần như mọi interval — đúng cái trạng thái
mà dashboard utilization không thể hiện được.

**Nó nghẽn ở đâu — và tôi không đồng ý với chẩn đoán mặc định.** MBU của nó là 0.207, tức cũng
không bão hoà HBM. Arithmetic intensity đạt được là **277 FLOP/byte** so với ridge point của H100
là **296 FLOP/byte** (990 TFLOP/s ÷ 3.35 TB/s). Nằm dưới *cả hai* roof cùng lúc là dấu hiệu của
kernel **latency/occupancy-bound**, không phải memory-bound: batch nhỏ, sequence ngắn, overhead
launch/sync để tensor core chờ giữa các burst. Nghĩa là **fix đầu tiên là fix cấu hình serving**
(batch lớn hơn, continuous batching, sequence packing) — *miễn phí* — rồi mới đến fix hardware.
*Caveat trung thực:* trên cả 11 GPU, MBU đi theo MFU trong khoảng 4–13%, nên telemetry này **không
tách sạch được** memory-bound khỏi occupancy-bound; phải xác nhận bằng Nsight (SM occupancy, DRAM
throughput, achieved warps) trước khi mua bất cứ thứ gì dựa trên nó.

**Bằng chứng mạnh nhất rằng utilization không thể định giá công việc.** `gpu-h100-3` báo **93.1%**
utilization — *thấp hơn* h100-4 — ở MFU 0.427, tức **2.2× lượng việc hữu ích với cùng $2.50/hr**.
Sắc hơn nữa: `gpu-a10g-1` và `gpu-a10g-0` là **cùng một loại part, cùng giá**: 96.9% so với 25.0%
utilization (**3.9×**) nhưng MFU 0.268 so với 0.218 (**chỉ 1.23×**). Một dashboard xếp hạng hai
GPU này theo utilization sẽ xếp **gần như ngược**.

**Hệ quả governance:** GPU-Util là **liveness signal** — hữu ích để phát hiện trainer đã crash, vô
dụng cho capacity planning. Efficiency review và chargeback phải chạy trên MFU/MBU và
$/1M-token, hai chỉ số **không thể bị game bằng cách giữ device bận trên danh nghĩa**.

---

## 4. Extension đã làm (5/5)

**1 — Purchasing policy định giá risk và commitment term.** `recommend_tier()` giờ chấm mọi tier
khả thi theo `$/useful-hour`: spot mang reclaim rate riêng theo loại GPU (H100 3% → A10G 12%) cộng
checkpoint overhead và rework kỳ vọng; **reservation bị tính đủ 24h/ngày vì đó là cách một cam kết
thực sự bill**; 1yr vs 3yr được gate theo số ngày job thật sự chạy.
*Kết quả:* policy duty-cycle gốc khai 39.1% tiết kiệm; định giá reservation 24/7 **điều chỉnh
xuống 38.1%** (+$252/mo). Đây là một chỉnh sửa **theo hướng xấu đi**, và tôi giữ nó — con số cũ
được tạo ra bởi một lỗi mô hình (tính reserved chỉ trên giờ dùng). *Insight:* break-even reclaim
rate của **mọi** GPU trong catalogue đều > 100%/h, nên spot **không** phải lựa chọn rủi ro cho
việc interruptible — **commitment bị dùng dưới công suất mới là rủi ro**.

**2 — Right-size theo bandwidth, không theo giá niêm yết.** Với mỗi GPU dưới 30% MFU, tìm part
catalogue rẻ nhất *vẫn* đạt bandwidth nó đang thực sự dùng + 25% headroom, và định giá cú swap
bằng `$/TB/s-hr` và `$/GB-VRAM-hr` thay vì `$/GPU-hr`.
*Kết quả:* 4/11 GPU đủ điều kiện, trị giá **$1,140/mo** (tính trên active hours). *Insight:* chọn
theo `$/GPU-hr` sẽ đề xuất L4 cho workload H100 — L4 **không kham nổi** memory traffic của nó.
Nghịch lý hữu ích: H100 **rẻ hơn** L4 tính theo $/TB/s ($0.746 vs $2.667/TB/s-hr,
tức 3.6×) dù đắt hơn 3.1× theo giờ ($2.50 vs $0.80).

**3 — Prompt caching bị gate bằng break-even của chính nó.** Cache write đắt thêm 25%, read tiết
kiệm 90% → break-even = 0.25/0.90 = **0.278 re-read**. Nhưng **TTL 5 phút chặn re-read ở mức
những gì đến trong cùng một window** (288 window/ngày), nên gate được đánh giá theo từng prefix
group `(team, route_tier)`.
*Kết quả:* **5/8 group** nhận dưới 1 request mỗi TTL window nên **lỗ** khi bật cache
(assistant/large, eval/large, eval/small, rag/large, search/large). Từ chối chúng nghĩa là bỏ
$28/mo "tiết kiệm" **vốn không hề có thật**. *Insight:* "bật caching ở mọi nơi" là default sai;
đúng metric là **re-read trên mỗi write trong một TTL window**, không phải hit-rate trên ngày.

**4 — Reasoning-token budget.** Tách spend và energy theo `is_reasoning`, rồi mô phỏng cap
reasoning về một tỉ lệ traffic (giữ lại các câu trả lời dài nhất, downgrade phần còn lại).
*Kết quả:* reasoning là **8.4% request nhưng 14.9% chi phí và 94.0% năng lượng** — 148.2 Wh/request
so với 0.86 Wh (**173×**). Cap về 5% traffic tiết kiệm $0.23/ngày (2.4%) và **7,880 Wh/ngày (25%
năng lượng serving)**. *Insight:* reasoning là **đòn bẩy sustainability trước, đòn bẩy chi phí
sau** — chênh lệch năng lượng lớn hơn chênh lệch tiền một bậc. Routing rule: chạy model rẻ trước,
escalate sang reasoning chỉ khi confidence dưới ngưỡng, **gate theo độ phức tạp task chứ không
theo default của team**.

**5 — Carbon-aware scheduling.** Chỉ đặt lại các job **interruptible** theo carbon intensity của
grid, và chấm cả 5 region đồng thời trên giá điện, carbon và latency (min-max normalise).
*Kết quả:* **1,657 kgCO2e/tháng (−92%, ~19.9 tấn/năm)** khi chuyển 5 job movable sang
europe-north1. **28% năng lượng fleet là latency-bound và không thể di chuyển.** *Insight:* ba
"region tốt nhất" là ba nơi khác nhau — sạch nhất europe-north1 (30 gCO2/kWh), điện rẻ nhất
us-east-wa ($0.055/kWh), cân bằng tốt nhất us-east-wa. Và phải trung thực về tiền: với neocloud,
giá điện **đã nằm trong** GPU-hour rate, nên region là **đòn bẩy carbon lớn, đòn bẩy chi phí nhỏ**
($308/mo) — chỉ vào P&L nếu tự host.

---

## 5. Ba hành động đầu tiên nếu tôi là FinOps lead của NimbusAI

**1. Đổi đơn vị đo trước khi tiêu một đồng nào — tuần 1, chi phí ~$0.**
Bỏ GPU-Util khỏi mọi dashboard capacity/efficiency (giữ lại đúng một chỗ: alert liveness). Thay
bằng **MFU/MBU per GPU** và **$/1M-token per team**. Bật **showback** ngay: tag coverage đang là
**91.8%**, đã vượt ngưỡng 80% để chargeback là hợp lý; export FOCUS đã có
(`outputs/focus_export.csv`). Lý do đây là hành động *đầu tiên* dù không tiết kiệm đồng nào: **mọi
quyết định trong ba mục dưới đây đều được định giá bằng những con số này**, và dashboard hiện tại
đang chủ động dẫn sai — `gpu-h100-4` trông như GPU được dùng tốt nhất fleet trong khi nó là GPU tệ
nhất. Không sửa thước đo thì mọi tiết kiệm sau đó đều không chứng minh được.

**2. Purchasing policy — cùng sprint đó. $9,788/mo, payback 1.8 tháng.**
5 job interruptible → spot (kèm checkpointing, đây là điều kiện tiên quyết chứ không phải
tuỳ chọn). Job steady → reserved, nhưng **chỉ khi duty cycle vượt break-even = 1 − discount tính
trên bill 24/7**, và term (1yr/3yr) phải khớp với thời gian job thật sự tồn tại. Đây là 77% tổng
tiết kiệm với effort thấp nhất tính theo $/engineer-day ($653/eng-day). Rủi ro cần quản: cam kết
3 năm là một **liability** nếu roadmap đổi — nên tôi cam kết ngắn hơn mức tối đa và ưu tiên spot.

**3. Hai thứ ship nhanh: tắt GPU idle + cascade router. $1,783/mo.**
Scheduler policy tắt GPU ngoài giờ làm ($600/mo, 2 ngày công) — chỉ cần kèm một đường
wake-on-demand để submission đêm không phải chờ đến sáng. Song song, ship **cascade router**
($1,122/mo trong tổng $1,183 của đòn bẩy inference) với một **eval gate trên escalation rate** để
chất lượng không tụt lặng lẽ. Batch API và prompt cache (đã gate) đi kèm miễn phí sau đó.

**Và một quyết định về việc *không* làm:** **hoãn right-size** ($1,140/mo) đến kỳ refresh hardware
kế tiếp. Payback là **31.6 tháng** so với chi phí engineer $1,200/ngày — dài hơn cả một năm thời
gian kỹ sư mà nó tiêu tốn. Đây là chỗ *lãng phí lộ rõ nhất* trên dashboard, nên nó sẽ là thứ mọi
người muốn làm trước; ROI nói ngược lại. Việc cần làm trước là **tăng batch size** trên
`gpu-h100-4` — miễn phí, và theo phân tích roofline ở mục 3 thì rất có thể nó **xoá luôn nhu cầu**
đổi hardware.

**Tôi sẽ đo thành công bằng gì:** `$/1M-token` theo tuần, MFU trung bình có trọng số của fleet, và
tỉ lệ chi phí đã được gán cho một owner. Ba chỉ số đó, không phải tổng hoá đơn — vì tổng hoá đơn
*nên* tăng khi công ty tăng trưởng.
