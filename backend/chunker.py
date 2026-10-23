"""
Module phân đoạn văn bản thông minh (Chunking) theo cấu trúc và ngữ nghĩa.
Lưu metadata chi tiết (page, section, chunk_id).
"""

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def split_into_semantic_chunks(pages_data, max_chars=750, overlap=100):
    chunks = []
    chunk_counter = 0

    for page_obj in pages_data:
        page_num = page_obj["page"]
        section = page_obj["section"]
        text = page_obj["text"]

        if not text:
            continue

        # Nếu là trang thông số kỹ thuật (50) hoặc trang mã lỗi (47, 48, 49), giữ khối có cấu trúc
        if page_num == 50:
            chunk_counter += 1
            chunks.append({
                "chunk_id": f"chunk_{chunk_counter:03d}",
                "page": page_num,
                "section": section,
                "text": f"THÔNG SỐ KỸ THUẬT MÁY GIẶT CỬA TRÊN SAMSUNG WA12CG5886** & WA14CG5886**:\n{text}"
            })
            continue

        # Tách các đoạn dựa trên 2 dấu xuống dòng hoặc tiêu đề BƯỚC / CẢNH BÁO / KHUYẾN CÁO
        paragraphs = re.split(r'\n{2,}', text)
        current_chunk = ""

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            
            if len(current_chunk) + len(p) <= max_chars:
                current_chunk += ("\n\n" + p if current_chunk else p)
            else:
                if current_chunk:
                    chunk_counter += 1
                    chunks.append({
                        "chunk_id": f"chunk_{chunk_counter:03d}",
                        "page": page_num,
                        "section": section,
                        "text": current_chunk.strip()
                    })
                # Giữ overlap một phần cuối của chunk trước
                if len(current_chunk) > overlap:
                    overlap_text = current_chunk[-overlap:]
                    current_chunk = overlap_text + "\n" + p
                else:
                    current_chunk = p

        if current_chunk.strip():
            chunk_counter += 1
            chunks.append({
                "chunk_id": f"chunk_{chunk_counter:03d}",
                "page": page_num,
                "section": section,
                "text": current_chunk.strip()
            })

    output_file = DATA_DIR / "chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"[+] Đã tạo {len(chunks)} chunks ngữ nghĩa, lưu tại {output_file}")
    return chunks

if __name__ == "__main__":
    raw_file = DATA_DIR / "raw_text_by_page.json"
    with open(raw_file, "r", encoding="utf-8") as f:
        pages = json.load(f)
    split_into_semantic_chunks(pages)
