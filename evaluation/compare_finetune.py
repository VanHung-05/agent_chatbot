"""
Script đánh giá so sánh trực tiếp cả 3 mô hình Embedding:
1. BAAI/bge-m3 (Pretrained 1024D)
2. bkai-foundation-models/vietnamese-bi-encoder (Gốc Pretrained 768D)
3. models/finetuned-vietnamese-embedding (Sau Fine-tuning với MNRL 768D)

Đo lường Recall@K (K=1, 3, 5, 10, 20), MRR và Độ trễ trên 40 câu hỏi benchmark.
Tự động xuất bảng Markdown và lưu kết quả vào data/finetune_comparison_results.json.
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

def get_device():
    # Sử dụng CPU để đảm bảo độ tương thích ổn định với PyTorch trên macOS
    return "cpu"

def is_page_match(chunk_page, target_page):
    if isinstance(target_page, list):
        return chunk_page in target_page
    return chunk_page == target_page

def evaluate_model(model_path_or_name: str, label: str, chunks, questions):
    device = get_device()
    print(f"\n[*] Đang đánh giá: {label} ({model_path_or_name}) trên [{device}]...")
    
    start_load = time.time()
    model = SentenceTransformer(model_path_or_name, device=device)
    load_time = time.time() - start_load

    # 1. Mã hoá chunks
    passage_texts = [c["text"] for c in chunks]
    p_embeddings = model.encode(passage_texts, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)

    # 2. Mã hoá câu hỏi
    query_texts = [q["query"] for q in questions]
    start_q = time.time()
    q_embeddings = model.encode(query_texts, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)
    avg_latency = ((time.time() - start_q) / len(questions)) * 1000

    # 3. Tính tương đồng Cosine
    sim_matrix = torch.mm(q_embeddings, p_embeddings.t()).cpu().numpy()

    ks = [1, 3, 5, 10, 20]
    recall_at_k = {k: 0 for k in ks}
    rr_scores = []

    for i, q in enumerate(questions):
        target_page = q["target_page"]
        ranked_indices = np.argsort(-sim_matrix[i])

        first_hit_rank = None
        for rank, idx in enumerate(ranked_indices, start=1):
            if is_page_match(chunks[idx]["page"], target_page):
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
    return {
        "label": label,
        "model": model_path_or_name,
        "mrr": float(np.mean(rr_scores)),
        "recall@1": recall_at_k[1] / total_q,
        "recall@3": recall_at_k[3] / total_q,
        "recall@5": recall_at_k[5] / total_q,
        "recall@10": recall_at_k[10] / total_q,
        "recall@20": recall_at_k[20] / total_q,
        "latency_ms": avg_latency
    }

def main():
    chunks_path = DATA_DIR / "chunks.json"
    questions_path = DATA_DIR / "benchmark_questions.json"
    ft_model_path = str(MODELS_DIR / "finetuned-vietnamese-embedding")

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # 1. Đánh giá BAAI/bge-m3
    res_bge = evaluate_model("BAAI/bge-m3", "BAAI/bge-m3 (Pretrained 1024D)", chunks, questions)

    # 2. Đánh giá Pretrained model gốc BKAI
    res_pretrained = evaluate_model("bkai-foundation-models/vietnamese-bi-encoder", "GỐC (Pretrained)", chunks, questions)

    # 3. Đánh giá Finetuned model
    res_finetuned = evaluate_model(ft_model_path, "SAU FINE-TUNE (MNRL)", chunks, questions)

    # In bảng so sánh trực quan cả 3 model
    print("\n" + "="*95)
    print("📊 BẢNG SO SÁNH HIỆU NĂNG THỰC NGHIỆM: CẢ 3 MÔ HÌNH EMBEDDING TRÊN 40 CÂU TEST")
    print("="*95)
    print(f"{'Chỉ số đánh giá':<20} | {'1. BGE-M3 (1024D)':<18} | {'2. BKAI Gốc (768D)':<18} | {'3. Finetuned (768D)':<18} | {'Cải thiện (3 vs 2)'}")
    print("-"*95)

    metrics = [
        ("MRR (Mean Reciprocal)", "mrr", "{:.4f}"),
        ("Recall@1 (Top-1)", "recall@1", "{:.1%}"),
        ("Recall@3 (Top-3)", "recall@3", "{:.1%}"),
        ("Recall@5 (Top-5)", "recall@5", "{:.1%}"),
        ("Recall@10 (Top-10)", "recall@10", "{:.1%}"),
        ("Recall@20 (Top-20)", "recall@20", "{:.1%}"),
        ("Độ trễ trung bình", "latency_ms", "{:.1f}ms"),
    ]

    for name, key, fmt in metrics:
        v_bge = res_bge[key]
        v_pre = res_pretrained[key]
        v_ft = res_finetuned[key]

        s_bge = fmt.format(v_bge)
        s_pre = fmt.format(v_pre)
        s_ft = fmt.format(v_ft)

        if "ms" in fmt:
            diff_str = f"{v_ft - v_pre:+.1f}ms"
        else:
            diff = v_ft - v_pre
            if "%" in fmt:
                diff_str = f"{diff*100:+.1f}%" + (" 🚀" if diff > 0 else "")
            else:
                diff_str = f"{diff:+.4f}" + (" 🚀" if diff > 0 else "")

        print(f"{name:<20} | {s_bge:<18} | {s_pre:<18} | {s_ft:<18} | {diff_str}")

    print("="*95)

    # Lưu kết quả toàn diện vào data/finetune_comparison_results.json
    comparison_file = DATA_DIR / "finetune_comparison_results.json"
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump({
            "bge_m3": res_bge,
            "bkai_pretrained": res_pretrained,
            "bkai_finetuned": res_finetuned
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[+] Đã lưu toàn bộ kết quả 3 model vào: {comparison_file}")

if __name__ == "__main__":
    main()
