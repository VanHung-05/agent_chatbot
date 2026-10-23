# 🤖 AI Agent Tra Cứu Hướng Dẫn Sử Dụng Máy Giặt Samsung

Hệ thống **AI Agent Toàn Diện (Tool-Augmented Autonomous Agent)** chạy hoàn toàn local trên máy tính cá nhân bằng các mô hình Deep Learning (Ollama Qwen2.5-7B, BGE-M3, PyTorch, Sentence-Transformers) để tra cứu tài liệu **Hướng dẫn sử dụng máy giặt cửa trên Samsung (WA12CG5886 / WA14CG5886)** (52 trang PDF).

---

## 🌟 Điểm Nổi Bật Của Hệ Thống

1. **AI Agent Thực Thụ (Function Routing & Multi-Tool Calling)**:
   - `lookup_error_code`: Tra cứu $O(1)$ mã lỗi từ bảng dữ liệu có cấu trúc (`data/structured_error_codes.json` gồm 1C, 4C, 5C, Ub, dC...).
   - `get_specifications`: Tra cứu tức thì bảng thông số kỹ thuật (công suất, kích thước, khối lượng, model WA12/WA14).
   - `search_manual_rag`: Tìm kiếm ngữ nghĩa Neural 2 giai đoạn trên toàn bộ 52 trang tài liệu.
2. **Xử Lý Văn Bản Tiếng Việt Chuẩn Xác**:
   - Sử dụng **PyMuPDF (`fitz`)** trích xuất toàn vẹn cấu trúc và font chữ.
   - Chuẩn hoá Unicode dựng sẵn với `unicodedata.normalize('NFC', text)` loại bỏ hoàn toàn hiện tượng lỗi vỡ font tiếng Việt.
3. **Query Rewriter Xử Lý Hội Thoại Nối Tiếp (Multi-Turn)**:
   - Sử dụng LLM tự động viết lại các câu hỏi phụ thuộc ngữ cảnh (Ví dụ: *"còn cách khắc phục?"* $\rightarrow$ *"Máy giặt Samsung báo lỗi 4C là không có nước cấp, còn cách khắc phục như thế nào?"*).
4. **Nguyên Tắc Chống Ảo Giác (Anti-Hallucination) Tuyệt Đối**:
   - System Prompt siết chặt: Khi tài liệu không đề cập (ví dụ hỏi về NASA hay thiết bị khác), Agent kiên quyết từ chối thay vì tự suy đoán.
5. **Giao Diện Chat Web Hiện Đại (Frontend)**:
   - Phong cách Samsung Dark Theme với Glassmorphism (`#080d1a`, `#1428a0`, `#00a9e0`).
   - Hiển thị thanh trạng thái gọi tool thời gian thực.
   - Streaming SSE (chữ chạy theo thời gian thực) và trích dẫn số trang tài liệu tham chiếu.

---

## 📂 Cấu Trúc Dự Án

```
agent_hdsd/
├── Hướng-dẫn-sử-dụng.pdf               # Tài liệu gốc Samsung (52 trang)
├── data/
│   ├── raw_text_by_page.json           # Text 52 trang đã chuẩn hóa NFC
│   ├── chunks.json                     # 52 chunks ngữ nghĩa
│   ├── structured_error_codes.json     # Cơ sở dữ liệu mã lỗi có cấu trúc
│   ├── benchmark_questions.json        # 40 câu hỏi trắc nghiệm benchmark
│   └── embeddings_cache_bge-m3.npy     # Cache vector 1024D của BGE-M3
├── evaluation/
│   ├── evaluate_embeddings.py          # Script benchmark so sánh đa mô hình
│   ├── evaluate_reranker.py            # Script benchmark Reranker Stage 2
│   └── finetune_embedding.py           # Huấn luyện MultipleNegativesRankingLoss
├── backend/
│   ├── pdf_processor.py                # Xử lý PDF với PyMuPDF
│   ├── chunker.py                      # Bộ chia đoạn thông minh
│   ├── tools/
│   │   ├── error_code_tool.py          # Tool tra mã lỗi
│   │   ├── specs_tool.py               # Tool tra thông số kỹ thuật
│   │   └── rag_tool.py                 # Tool tìm kiếm Neural BGE-M3
│   ├── agent.py                        # Lõi AI Agent Orchestrator & Rewriter
│   └── main.py                         # FastAPI Server + SSE Streaming
└── frontend/
    ├── index.html                      # Giao diện chat
    ├── style.css                       # Dark theme Samsung Blue
    └── app.js                          # Client Javascript xử lý SSE
```

---

## 🚀 Hướng Dẫn Khởi Động & Trải Nghiệm

### 1. Yêu cầu môi trường
Đảm bảo **Ollama** đang chạy và đã tải model:
```bash
ollama serve
ollama pull qwen2.5:7b
ollama pull bge-m3
```

### 2. Khởi động Web App
Chạy lệnh khởi động server FastAPI:
```bash
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Truy cập ngay trên trình duyệt: **[http://localhost:8000](http://localhost:8000)**

---

## 🔬 Các Script Đánh Giá & Huấn Luyện (Evaluation)

- **So sánh Pretrained vs Finetuned**:
  ```bash
  python3 evaluation/compare_finetune.py
  ```
- **So sánh Embedding Models (Recall@K / MRR)**:
  ```bash
  python3 evaluation/evaluate_embeddings.py
  ```
- **Kiểm tra Reranker Stage 2**:
  ```bash
  python3 evaluation/evaluate_reranker.py
  ```
- **Fine-tuning Embedding với MNRL**:
  ```bash
  python3 evaluation/finetune_embedding.py
  ```

---

## 📊 Bảng So Sánh Thực Nghiệm 3 Mô Hình Embedding

Đánh giá thực nghiệm khách quan trên tập **40 câu hỏi kiểm thử (Benchmark Ground-Truth)** từ sách hướng dẫn sử dụng máy giặt Samsung:

| Tiêu chí / Chỉ số | 1. BAAI/bge-m3 (Pretrained) | 2. BKAI Bi-Encoder (Gốc) | 3. Finetuned Model (Sau huấn luyện) | Mức độ cải thiện (Finetuned vs Gốc) |
|---|---|---|---|---|
| **Đơn vị phát triển** | BAAI | ĐH Bách Khoa Hà Nội | **Tự huấn luyện (Custom)** | - |
| **Kiến trúc mạng** | XLM-RoBERTa Large | PhoBERT / RoBERTa Base | **Vietnamese Bi-Encoder + MNRL** | - |
| **Số chiều vector ($d$)** | **1024 chiều** | 768 chiều | 768 chiều | - |
| **Hàm mất mát huấn luyện** | InfoNCE / Contrastive | Cross-Entropy / SimCSE | **MultipleNegativesRankingLoss** | - |
| **MRR (Mean Reciprocal Rank)** | **0.8304** 🥇 | 0.6609 | **0.7350** 🚀 | **+0.0741** |
| **Recall@1 (Top-1 trúng ngay)** | **72.5%** 🥇 | 52.5% | **62.5%** 🚀 | **+10.0%** |
| **Recall@3 (Top-3)** | **95.0%** 🥇 | 75.0% | **82.5%** 🚀 | **+7.5%** |
| **Recall@5 (Top-5)** | **95.0%** 🥇 | 82.5% | **90.0%** 🚀 | **+7.5%** |
| **Recall@10 (Top-10)** | **97.5%** | 95.0% | **97.5%** 🥇 | **+2.5%** |
| **Độ trễ trung bình / query** | **~24.0 ms** | **~6.7 ms** ⚡ | **~6.8 ms** ⚡ | +0.1 ms |

### 💡 Phân Tích Thực Nghiệm (Dành Cho Báo Cáo):
1. **BAAI/bge-m3**: Đạt chất lượng toàn diện cao nhất (**MRR: 0.8304, Recall@5: 95.0%**, Recall@20: 100%) nhờ không gian vector lớn 1024 chiều và tập dữ liệu huấn luyện đa ngôn ngữ khổng lồ.
2. **BKAI Bi-Encoder (Gốc)**: Tốc độ truy vấn siêu nhanh (~6.7 ms) nhưng điểm Recall@1 ban đầu chỉ đạt 52.5% do chưa được làm quen với từ vựng chuyên ngành máy giặt Samsung.
3. **Finetuned Model (Sau huấn luyện với MultipleNegativesRankingLoss)**: 
   - Sau khi huấn luyện đối kháng (Contrastive Learning) với các cặp dữ liệu `(câu hỏi, chunk)` sinh tự động từ Qwen2.5-7B, chất lượng tăng vọt: **Recall@1 tăng +10.0%** (lên 62.5%), **Recall@5 đạt 90.0%**, **MRR tăng +0.0741** (từ 0.6609 lên 0.7350).
   - Tốc độ truy vấn vẫn giữ nguyên mức siêu nhanh (~6.8 ms), nhanh hơn BGE-M3 gần gấp 4 lần.
   - Chứng minh kỹ thuật Domain Adaptation giúp mô hình chuyên biệt tiếng Việt đạt chất lượng vượt trội trên tài liệu kỹ thuật Samsung.
