"""
Đánh giá hiệu quả của 2-Stage Retrieval:
Stage 1: Dense Retrieval lấy Top-20 candidates
Stage 2: Cross-Encoder Reranker (BAAI/bge-reranker-v2-m3) xếp hạng lại và trích xuất Top-5
Đo lường sự cải thiện của MRR và Recall@5
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

def is_page_match(chunk_page, target_page):
    if isinstance(target_page, list):
        return chunk_page in target_page
    return chunk_page == target_page

def evaluate_reranker(dense_model_name="BAAI/bge-m3", reranker_name="BAAI/bge-reranker-v2-m3"):
    device = get_device()
    print(f"\n========================================================")
    print(f"[*] Benchmark 2-Stage Pipeline:")
    print(f"    Dense:    {dense_model_name}")
    print(f"    Reranker: {reranker_name}")
    print(f"    Device:   [{device}]")
    print(f"========================================================")

    with open(DATA_DIR / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(DATA_DIR / "benchmark_questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    # 1. Load Dense & Reranker models
    print("[*] Đang tải mô hình...")
    dense_model = SentenceTransformer(dense_model_name, device=device)
    reranker = CrossEncoder(reranker_name, device=device)
    print("[+] Tải các mô hình thành công.")

    # 2. Encode toàn bộ passages
    passage_texts = [c["text"] for c in chunks]
    passage_embeddings = dense_model.encode(passage_texts, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)

    # 3. Dense Retrieval Top-20
    query_texts = [q["query"] for q in questions]
    query_embeddings = dense_model.encode(query_texts, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)
    sim_matrix = torch.mm(query_embeddings, passage_embeddings.t()).cpu().numpy()

    top_20_per_query = [np.argsort(-sim_matrix[i])[:20] for i in range(len(questions))]

    # 4. Đo chỉ số TRƯỚC KHI RERANK (Chỉ dùng Dense Model)
    total_q = len(questions)
    dense_rr = []
    dense_recall_5 = 0
    dense_recall_20 = 0

    for i, q in enumerate(questions):
        target_page = q["target_page"]
        indices = top_20_per_query[i]
        
        hit_rank = None
        for rank, idx in enumerate(indices, start=1):
            if is_page_match(chunks[idx]["page"], target_page):
                hit_rank = rank
                break
        
        if hit_rank is not None:
            dense_rr.append(1.0 / hit_rank)
            if hit_rank <= 5:
                dense_recall_5 += 1
            if hit_rank <= 20:
                dense_recall_20 += 1
        else:
            dense_rr.append(0.0)

    # 5. RERANK TOP-20 XUỐNG TOP-5 BẰNG CROSS-ENCODER
    print("[*] Đang chạy Cross-Encoder Reranker trên Top-20...")
    rerank_rr = []
    rerank_recall_5 = 0

    start_rerank_time = time.time()
    for i, q in enumerate(questions):
        target_page = q["target_page"]
        query_text = q["query"]
        indices = top_20_per_query[i]

        # Tạo 20 cặp (query, chunk_text)
        pairs = [[query_text, chunks[idx]["text"]] for idx in indices]
        scores = reranker.predict(pairs)
        
        # Sắp xếp lại top 20 theo cross-encoder score
        reranked_order = np.argsort(-scores)
        reranked_indices = [indices[j] for j in reranked_order]

        hit_rank = None
        for rank, idx in enumerate(reranked_indices, start=1):
            if is_page_match(chunks[idx]["page"], target_page):
                hit_rank = rank
                break

        if hit_rank is not None:
            rerank_rr.append(1.0 / hit_rank)
            if hit_rank <= 5:
                rerank_recall_5 += 1
        else:
            rerank_rr.append(0.0)

    total_rerank_time = time.time() - start_rerank_time
    avg_rerank_latency = (total_rerank_time / total_q) * 1000

    results = {
        "dense_model": dense_model_name,
        "reranker_model": reranker_name,
        "before_rerank": {
            "mrr": float(np.mean(dense_rr)),
            "recall@5": dense_recall_5 / total_q,
            "recall@20": dense_recall_20 / total_q
        },
        "after_rerank": {
            "mrr": float(np.mean(rerank_rr)),
            "recall@5": rerank_recall_5 / total_q
        },
        "avg_rerank_latency_ms": avg_rerank_latency
    }

    print("\n------------------- KẾT QUẢ SO SÁNH -------------------")
    print(f"Trước Rerank (Dense Top-5) : MRR = {results['before_rerank']['mrr']:.4f} | Recall@5 = {results['before_rerank']['recall@5']*100:.1f}%")
    print(f"Sau Rerank (Top-20 -> Top-5): MRR = {results['after_rerank']['mrr']:.4f} | Recall@5 = {results['after_rerank']['recall@5']*100:.1f}%")
    print(f"Độ trễ trung bình Reranker: {avg_rerank_latency:.1f}ms / câu hỏi")

    output_path = DATA_DIR / "reranker_benchmark_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[+] Kết quả lưu tại: {output_path}")
    return results

if __name__ == "__main__":
    evaluate_reranker()
