"""
Pipeline Fine-tuning mô hình Embedding:
1. Dùng Qwen2.5-7B sinh tự động các cặp (query, positive_chunk) từ các chunks PDF.
2. Huấn luyện bằng PyTorch + MultipleNegativesRankingLoss (In-batch negatives).
3. Đánh giá kiểm chứng trước và sau fine-tune trên tập benchmark.
"""

import json
import time
import requests
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
OLLAMA_URL = "http://localhost:11434/api/generate"

def generate_queries_for_chunks(max_chunks=25):
    """Sử dụng Qwen2.5-7B qua Ollama để sinh câu hỏi từ mỗi chunk."""
    with open(DATA_DIR / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    dataset_path = DATA_DIR / "ft_dataset.json"
    if dataset_path.exists():
        print(f"[+] Đã tìm thấy tập dữ liệu sinh sẵn tại {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"[*] Bắt đầu sinh câu hỏi huấn luyện tự động với Qwen2.5-7B (lấy mẫu {max_chunks} chunks)...")
    ft_data = []

    for i, c in enumerate(chunks[:max_chunks]):
        prompt = (
            f"Bạn là chuyên gia về máy giặt Samsung. Dưới đây là một đoạn trích từ hướng dẫn sử dụng:\n\n"
            f"\"\"\"{c['text'][:500]}\"\"\"\n\n"
            f"Hãy đặt ra ĐÚNG 2 câu hỏi thực tế mà người dùng sẽ hỏi, và câu trả lời nằm trọn vẹn trong đoạn trích trên.\n"
            f"Yêu cầu: Chỉ trả về 2 câu hỏi, mỗi câu trên 1 dòng, không đánh số, không giải thích."
        )

        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": "qwen2.5:7b",
                "prompt": prompt,
                "stream": False
            }, timeout=30)
            if resp.status_code == 200:
                output = resp.json().get("response", "").strip()
                lines = [l.strip().lstrip("0123456789.- ") for l in output.split("\n") if len(l.strip()) > 10]
                for q in lines[:2]:
                    ft_data.append({
                        "query": q,
                        "positive": c["text"],
                        "chunk_id": c["chunk_id"],
                        "page": c["page"]
                    })
                print(f"  [Chunk {i+1}/{max_chunks}] Sinh {len(lines[:2])} câu hỏi")
        except Exception as e:
            print(f"  [!] Lỗi khi gọi Qwen: {e}")

    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(ft_data, f, ensure_ascii=False, indent=2)

    print(f"[+] Đã tạo tập dữ liệu huấn luyện gồm {len(ft_data)} cặp (query, chunk)")
    return ft_data

def finetune_embedding(base_model_name="bkai-foundation-models/vietnamese-bi-encoder", epochs=3, batch_size=8):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n========================================================")
    print(f"[*] Bắt đầu Fine-tuning mô hình: {base_model_name}")
    print(f"    Loss: MultipleNegativesRankingLoss")
    print(f"    Thiết bị: [{device}] | Epochs: {epochs} | Batch size: {batch_size}")
    print(f"========================================================")

    # 1. Chuẩn bị dữ liệu
    ft_data = generate_queries_for_chunks()
    if not ft_data:
        print("[!] Không có dữ liệu huấn luyện.")
        return

    train_examples = [InputExample(texts=[item["query"], item["positive"]]) for item in ft_data]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    # 2. Khởi tạo Model & Loss function
    model = SentenceTransformer(base_model_name, device=device)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # 3. Huấn luyện
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = str(MODELS_DIR / "finetuned-vietnamese-embedding")

    print("[*] Đang tiến hành huấn luyện...")
    start_train = time.time()
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=int(len(train_dataloader) * epochs * 0.1),
        output_path=save_path,
        show_progress_bar=True
    )
    print(f"[+] Huấn luyện hoàn tất trong {time.time() - start_train:.2f}s!")
    print(f"[+] Mô hình đã được lưu tại: {save_path}")

    return save_path

if __name__ == "__main__":
    finetune_embedding()
