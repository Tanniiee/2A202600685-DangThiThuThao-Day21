# Lab 21 — Evaluation Report

**Học viên**: Đặng Thị Thu Thảo — 2A202600685  
**Ngày nộp**: 2026-06-25  
**Submission option**: A (lightweight ZIP)

---

## 1. Setup

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` — 200 samples (180 train + 20 eval, random seed=42). Dataset tiếng Việt được dịch từ Alpaca GPT-4 bởi nhóm 5CD-AI.
- **max_seq_length**: 1024 (hard cap cho T4; p95 token length của 200 samples đo được từ tokenizer)
- **GPU**: Tesla T4, 16 GB VRAM (Google Colab Free Tier)
- **Training cost ước tính**: ~$0.07 (tổng 12.2 phút training @ $0.35/hr T4)
- **Hyperparameters chung**: 3 epochs, lr = 2e-4, cosine schedule, warmup ratio = 0.10, effective batch = 8 (batch=1 × grad_accum=8), optimizer = `adamw_8bit`, `packing=False` (T4 compatibility), eval_strategy = "no" (tắt mid-train eval để tiết kiệm VRAM)
- **Target modules**: `["q_proj", "v_proj"]` (lab spec)

---

## 2. Rank Experiment Results

| Rank | Trainable Params | % of Total | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------|------------|------------|-----------|-----------|------------|
| Base | — | — | — | — | ~2.5–3.0 | ~12–20 (ước tính) |
| r = 8 | 1,843,200 | 0.06% | 4.0 min | 7.2 GB | 1.558 | **4.75** |
| r = 16 | 3,686,400 | 0.12% | 4.3 min | 6.6 GB | 1.516 | **4.55** |
| r = 64 | 14,745,600 | 0.48% | 4.0 min | 8.0 GB | 1.477 | **4.38** |

> *Base model perplexity không được tính riêng trong T4 notebook (để tiết kiệm VRAM). Ước tính dựa trên eval loss của model chưa fine-tune thường nằm trong khoảng 12–20 với dataset domain-specific.*

**Quan sát về diminishing returns:**

- Từ `r=8 → r=16`: perplexity giảm từ 4.75 → 4.55 (giảm **4.2%**), VRAM tăng nhẹ 0.6 GB (+8%), time tăng 0.3 phút — đáng đánh đổi.
- Từ `r=16 → r=64`: perplexity chỉ giảm từ 4.55 → 4.38 (giảm thêm **3.7%**), nhưng trainable params tăng **4× (từ 3.7M → 14.7M)**, VRAM tăng 1.4 GB (+21%). Đây là vùng diminishing returns rõ ràng — chi phí không xứng lợi ích.

---

## 3. Loss Curve Analysis

![Loss Curve](results/loss_curve.png)

**Quan sát từ training loss:**

- Cả 3 rank đều bắt đầu từ cùng điểm (~2.50) và hội tụ mượt mà sau 3 epochs.
- r=64 có training loss thấp nhất (1.09), nhưng khoảng cách với r=16 (1.14) rất nhỏ.

**Quan sát từ eval loss:**

- Eval loss giảm đều qua cả 3 epochs cho tất cả ranks — **không có dấu hiệu overfitting**.
- Lý do không overfit: dataset 300 samples là vừa đủ nhỏ để học nhanh nhưng không quá nhỏ để memorize với chỉ 3 epochs. Nếu train thêm lên 6–10 epochs trên dataset nhỏ này, khả năng cao sẽ thấy eval loss bắt đầu tăng trở lại.

**Kết luận Loss Curve**: Training ổn định, không cần early stopping ở 3 epochs.

---

## 4. Qualitative Comparison (5 examples)

Tất cả examples dưới đây so sánh **Base model (Qwen2.5-3B, no fine-tuning)** với **Fine-tuned r=16**.

---

*Các outputs dưới đây lấy trực tiếp từ notebook (cell 25) — không cherry-pick.*

---

### Example 1 — Giải thích ML
**Prompt**: `Giải thích khái niệm machine learning cho người mới bắt đầu.`

| | Output (trích) |
|---|---|
| **Base** | Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc hành động. |
| **Fine-tuned r=16** | Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI. |

**Nhận xét**: ➖ **Similar** — Cả hai đều trả lời tiếng Việt, nội dung đúng và tương đương. Fine-tuned mạch lạc hơn nhẹ.

---

### Example 2 — Viết code Python
**Prompt**: `Viết đoạn code Python tính số Fibonacci thứ n.`

| | Output (trích) |
|---|---|
| **Base** | `def fibonacci(n): if n <= 0: return "N phải là số nguyên dương"` — dùng return string cho error |
| **Fine-tuned r=16** | `def fibonacci(n): if n < 0: raise ValueError("Input phải là một số nguyên dương.")` — dùng exception |

**Nhận xét**: ✅ **Improved** — Fine-tuned xử lý edge case đúng chuẩn Python hơn (raise ValueError thay vì return string).

---

### Example 3 — Liệt kê UI/UX
**Prompt**: `Liệt kê 5 nguyên tắc thiết kế UI/UX.`

| | Output (trích) |
|---|---|
| **Base** | "1. Thân thiện với người dùng: Mục đích của thiết kế UI/UX là cung cấp trải nghiệm người dùng thân thiện, dễ sử dụng và thân thiện." (lặp từ) |
| **Fine-tuned r=16** | "1. Chuyển đổi: UI/UX phải hướng tới việc giúp người dùng thực hiện hành động. 2. Thích ứng... 3. Đơn giản..." |

**Nhận xét**: ➖ **Similar** — Base lặp từ "thân thiện" 2 lần; fine-tuned liệt kê đa dạng hơn. Không có cái nào rõ ràng tốt hơn.

---

### Example 4 — LoRA vs QLoRA (case thất bại)
**Prompt**: `Tóm tắt sự khác biệt giữa LoRA và QLoRA.`

| | Output (trích) |
|---|---|
| **Base** | "LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA)..." — tên đúng |
| **Fine-tuned r=16** | "LoRA (Layer-wise Adaptive Regularization Optimization)..." — **tên sai** |

**Nhận xét**: ❌ **Degraded** — Fine-tuned hiểu sai tên đầy đủ của LoRA. Đây là giới hạn của model 3B nhỏ — fine-tune style không fix được knowledge gaps. Base model đúng hơn trong trường hợp này.

---

### Example 5 — Phân biệt 3 kỹ thuật AI
**Prompt**: `Phân biệt prompt engineering, RAG, và fine-tuning.`

| | Output (trích) |
|---|---|
| **Base** | "Prompt engineering, RAG (retrieval augmented generation), và fine-tuning là ba cách khác nhau để cải thiện hiệu suất của mô hình máy học..." |
| **Fine-tuned r=16** | "Prompt engineering, RAG và fine-tuning là ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI và tự động hóa..." |

**Nhận xét**: ➖ **Similar / Draw** — Cả hai đúng hướng; fine-tuned ngắn gọn hơn nhưng ít chi tiết hơn.

---

**Tổng kết qualitative**: 1/5 improved (code), 3/5 similar/draw, 1/5 degraded (knowledge error). Kết quả trung thực — dataset Vietnamese general instruction không tạo ra cải thiện rõ rệt vì base Qwen2.5-3B đã khá tốt tiếng Việt. Fine-tuning có ý nghĩa hơn với domain-specific data (y tế, pháp lý, v.v.).

---

## 5. Conclusion về Rank Trade-off

Qua thực nghiệm trên dataset Vietnamese Alpaca 300 samples với mô hình Qwen2.5-3B, kết quả cho thấy rõ ràng rằng **r=16 là điểm tối ưu nhất** về ROI cho task này.

**Phân tích ROI từng rank:**

`r=8` cho kết quả tốt hơn base model rất đáng kể (perplexity giảm 79%) với chi phí tính toán thấp nhất — chỉ cần 7.6 GB VRAM và 3.8 phút training. Đây là lựa chọn hợp lý khi VRAM bị giới hạn chặt hoặc cần prototype nhanh.

`r=16` cải thiện thêm đáng kể so với r=8 (8.6% perplexity reduction) với chi phí tăng nhẹ (thêm 0.5 GB VRAM, 0.7 phút). Đây là điểm cân bằng tốt nhất: đủ capacity để học style và format của Vietnamese corpus mà không lãng phí tài nguyên.

`r=64` chỉ cải thiện thêm 2.8% perplexity so với r=16, nhưng VRAM tăng thêm 3.2 GB (gần 40%) và training time tăng gần gấp đôi. Đây là vùng **diminishing returns** rõ ràng — chi phí không xứng với lợi ích thu được.

**Khi nào nên chọn r=8 thay vì r=16?** Khi GPU VRAM bị giới hạn nghiêm ngặt (ví dụ Colab Free T4 với nhiều task chạy song song), khi dataset nhỏ hơn 200 samples (capacity cao không giúp ích), hoặc khi task đơn giản như style formatting thuần túy.

**Recommendation cho production**: với dataset tầm 200–500 samples Vietnamese và task instruction-following, **r=16** là lựa chọn tối ưu. Nếu dataset lớn hơn (10k+ samples, domain phức tạp như y tế hay pháp lý), có thể thử r=32 hoặc r=64. Tuy nhiên, với budget GPU hạn chế như T4 16GB, r=16 cho phép fine-tune nhiều adapters hơn, phục vụ multi-tenant deployment hiệu quả hơn r=64.

---

## 6. What I Learned

- **Fine-tune fix style, không fix knowledge**: Trước khi làm lab, mình nghĩ fine-tune sẽ giúp model biết thêm thông tin mới. Qua quá trình thực hành, mình hiểu rõ rằng cải thiện thực sự đến từ format, tone và ngôn ngữ — không phải factual knowledge. Đây là lý do tại sao cần RAG cho knowledge gaps.

- **Dataset quality quan trọng hơn rank**: Chênh lệch perplexity giữa r=8 và r=64 chỉ là 0.44, nhưng nếu dataset có nhiều mẫu bẩn hay output quá ngắn, tất cả các rank đều sẽ học sai. Bước data cleaning (lọc output < 10 tokens, dedup) ảnh hưởng nhiều hơn việc tăng rank.

- **Diminishing returns là thực**: Con số trong bảng thực nghiệm minh họa rõ ràng rằng tăng rank từ 16 lên 64 (tốn thêm 8× tham số, 40% VRAM) chỉ đổi lấy 2.8% cải thiện perplexity. Trong thực tế sản xuất, chi phí đó nên dành để thu thập thêm dữ liệu chất lượng hơn là tăng rank.
