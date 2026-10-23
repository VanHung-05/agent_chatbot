"""
Tool 3: Neural Retrieval hỗ trợ chuyển đổi linh hoạt mô hình embedding:
1. 'bge-m3': Pretrained BAAI/bge-m3 qua Ollama (Đa ngôn ngữ, 1024-dim)
2. 'finetuned': Mô hình tiếng Việt đã Fine-tune với MultipleNegativesRankingLoss (models/finetuned-vietnamese-embedding, 768-dim)
3. 'vietnamese-bi-encoder': Model gốc Bách Khoa Hà Nội trước khi fine-tune
"""

import json
import torch
import requests
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"

class RAGSearchTool:
    def __init__(self, model_type="bge-m3"):
        self.model_type = model_type
        self.device = "cpu"
        self.pt_model = None
        
        with open(DATA_DIR / "chunks.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self._setup_model(model_type)

    def _setup_model(self, model_type: str):
        self.model_type = model_type
        print(f"[*] Cấu hình RAGSearchTool với mô hình: [{self.model_type}]")

        cache_key = self.model_type.replace("/", "_")
        cache_path = DATA_DIR / f"embeddings_cache_{cache_key}.npy"

        if self.model_type == "finetuned":
            ft_path = MODELS_DIR / "finetuned-vietnamese-embedding"
            if ft_path.exists():
                print(f"[+] Đang tải mô hình đã Fine-tune từ: {ft_path}")
                self.pt_model = SentenceTransformer(str(ft_path), device=self.device)
            else:
                print(f"[!] Chưa tìm thấy thư mục {ft_path}, fallback sang bge-m3")
                self.model_type = "bge-m3"

        elif self.model_type == "vietnamese-bi-encoder":
            print(f"[+] Đang tải mô hình gốc BKAI từ HuggingFace...")
            self.pt_model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder", device=self.device)

        if cache_path.exists():
            print(f"[+] Tải vector embeddings đã cache từ: {cache_path}")
            self.passage_embeddings = np.load(str(cache_path))
        else:
            print(f"[*] Chưa có cache, đang mã hoá {len(self.chunks)} chunks bằng [{self.model_type}]...")
            self.passage_embeddings = self._encode_batch([c["text"] for c in self.chunks])
            np.save(str(cache_path), self.passage_embeddings)
            print(f"[+] Đã lưu cache embeddings tại: {cache_path}")

        print(f"[+] RAGSearchTool sẵn sàng phục vụ với model [{self.model_type}].")

    def _encode_batch(self, texts: list) -> np.ndarray:
        if self.model_type in ["finetuned", "vietnamese-bi-encoder"] and self.pt_model:
            embeddings = self.pt_model.encode(texts, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)
            return embeddings.cpu().numpy()

        # Mặc định dùng Ollama bge-m3
        vectors = []
        for i, t in enumerate(texts):
            vec = self._get_ollama_embedding(t)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
            if (i + 1) % 15 == 0 or (i + 1) == len(texts):
                print(f"  -> Đã mã hoá {i+1}/{len(texts)} chunks...")
        return np.array(vectors)

    def _get_ollama_embedding(self, text: str) -> np.ndarray:
        resp = requests.post(OLLAMA_EMBED_URL, json={
            "model": "bge-m3",
            "prompt": text
        }, timeout=30)
        if resp.status_code == 200:
            return np.array(resp.json()["embedding"], dtype=np.float32)
        else:
            raise RuntimeError(f"Ollama embedding error: {resp.text}")

    def search(self, query: str, top_k: int = 4) -> list:
        if self.model_type in ["finetuned", "vietnamese-bi-encoder"] and self.pt_model:
            q_emb = self.pt_model.encode([query], convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)
            query_vec = q_emb[0].cpu().numpy()
        else:
            query_vec = self._get_ollama_embedding(query)
            norm = np.linalg.norm(query_vec)
            if norm > 0:
                query_vec = query_vec / norm

        scores = np.dot(self.passage_embeddings, query_vec)
        top_indices = np.argsort(-scores)[:top_k]

        results = []
        for idx in top_indices:
            c = self.chunks[idx]
            results.append({
                "chunk_id": c["chunk_id"],
                "page": c["page"],
                "section": c["section"],
                "text": c["text"],
                "score": float(scores[idx])
            })
        return results
