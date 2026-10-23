"""
Script benchmark so sánh đa mô hình Embedding qua PyTorch + sentence-transformers.
Đo đạc Recall@K (K=1, 3, 5, 10, 20) và MRR@K trên 40 câu hỏi.
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

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

def evaluate_embedding_model(model_name: str, chunks, questions, prefix_query="", prefix_passage=""):
    device = get_device()
    print(f"\n========================================================")
    print(f"[*] Bắt đầu đánh giá Model: {model_name} trên thiết bị [{device}]")
    print(f"========================================================")

    start_load = time.time()
    model = SentenceTransformer(model_name, device=device)
    print(f"[+] Load model hoàn tất trong {time.time() - start_load:.2f}s")

    # 1. Encode toàn bộ passages/chunks
    passage_texts = [f"{prefix_passage}{c['text']}" for c in chunks]
    start_encode_p = time.time()
    passage_embeddings = model.encode(passage_texts, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)
    encode_p_time = time.time() - start_encode_p
    print(f"[+] Encode {len(chunks)} chunks hoàn tất trong {encode_p_time:.2f}s")

    # 2. Encode queries và tính toán tương đồng Cosine
    query_texts = [f"{prefix_query}{q['query']}" for q in questions]
    start_encode_q = time.time()
    query_embeddings = model.encode(query_texts, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)
    encode_q_time = time.time() - start_encode_q

    # Cosine similarity matrix: (num_queries, num_passages)
    sim_matrix = torch.mm(query_embeddings, passage_embeddings.t()).cpu().numpy()

    # 3. Tính toán các chỉ số Retrieval: Recall@K & MRR@K
    ks = [1, 3, 5, 10, 20]
    recall_at_k = {k: 0 for k in ks}
    rr_scores = []

    for i, q in enumerate(questions):
        target_page = q["target_page"]
        ranked_indices = np.argsort(-sim_matrix[i])  # Sắp xếp giảm dần điểm số

        first_hit_rank = None
        for rank, idx in enumerate(ranked_indices, start=1):
            chunk_page = chunks[idx]["page"]
            if is_page_match(chunk_page, target_page):
                first_hit_rank = rank
                break

        if first_hit_rank is not None:
            rr_scores.append(1.0 / first_hit_rank)
            for k in ks:
                if first_hit_rank <= k:
                    recall_at_k[k] += 1
        else:
            rr_scores.append(0.0)

    total_q = len(questions)
    results = {
        "model_name": model_name,
        "device": device,
        "num_questions": total_q,
        "mrr": float(np.mean(rr_scores)),
        "recall@1": recall_at_k[1] / total_q,
        "recall@3": recall_at_k[3] / total_q,
        "recall@5": recall_at_k[5] / total_q,
        "recall@10": recall_at_k[10] / total_q,
        "recall@20": recall_at_k[20] / total_q,
        "query_latency_ms": (encode_q_time / total_q) * 1000
    }

    print(f"\n--- KẾT QUẢ CHO {model_name} ---")
    print(f"MRR:        {results['mrr']:.4f}")
    print(f"Recall@1:   {results['recall@1']*100:.1f}%")
    print(f"Recall@3:   {results['recall@3']*100:.1f}%")
    print(f"Recall@5:   {results['recall@5']*100:.1f}%")
    print(f"Recall@10:  {results['recall@10']*100:.1f}%")
    print(f"Recall@20:  {results['recall@20']*100:.1f}%")
    print(f"Avg Latency/Query: {results['query_latency_ms']:.1f}ms")

    return results

def main():
    chunks_path = DATA_DIR / "chunks.json"
    questions_path = DATA_DIR / "benchmark_questions.json"

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    models_to_test = [
        {"name": "BAAI/bge-m3", "prefix_q": "", "prefix_p": ""},
        {"name": "intfloat/multilingual-e5-base", "prefix_q": "query: ", "prefix_p": "passage: "},
        {"name": "bkai-foundation-models/vietnamese-bi-encoder", "prefix_q": "", "prefix_p": ""}
    ]

    all_results = []
    for m in models_to_test:
        try:
            res = evaluate_embedding_model(m["name"], chunks, questions, m["prefix_q"], m["prefix_p"])
            all_results.append(res)
        except Exception as e:
            print(f"[!] Lỗi khi đánh giá model {m['name']}: {e}")

    report_path = DATA_DIR / "embedding_benchmark_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[+] Đã lưu báo cáo so sánh vào: {report_path}")

if __name__ == "__main__":
    main()
