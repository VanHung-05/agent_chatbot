"""
Module trích xuất nội dung từ Hướng-dẫn-sử-dụng.pdf bằng PyMuPDF (fitz)
và xử lý font tiếng Việt với unicodedata.normalize('NFC', text)
"""

import fitz  # PyMuPDF
import re
import json
import unicodedata
from pathlib import Path

PDF_PATH = Path(__file__).resolve().parent.parent / "Hướng-dẫn-sử-dụng.pdf"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def clean_vietnamese_text(text: str) -> str:
    """Chuẩn hoá Unicode NFC và loại bỏ nhiễu font, khoảng trắng thừa."""
    if not text:
        return ""
    # Chuẩn hoá Unicode về chuẩn dựng sẵn (NFC)
    text = unicodedata.normalize("NFC", text)
    
    # Loại bỏ footer định dạng Untitled-3 ... ngày giờ
    text = re.sub(r'Untitled-\d+\s+\d+\s+\d{4}-\d{2}-\d{2}\s+[^\n]*', '', text)
    
    # Loại bỏ các dòng tiêu đề lặp lại ở đầu trang
    text = re.sub(r'^(Tiếng Việt\s*)+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^(Thông tin về an toàn|Lắp đặt|Trước khi bắt đầu|Bảo trì|Xử lý sự cố|Thông số kỹ thuật)\s*', '', text, flags=re.MULTILINE)
    
    # Chuẩn hoá khoảng trắng
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_pdf_pages():
    """Trích xuất từng trang PDF thành danh sách dict."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(PDF_PATH))
    pages_data = []

    print(f"[*] Bắt đầu trích xuất PDF: {PDF_PATH} ({len(doc)} trang)")
    
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        raw_text = page.get_text("text")
        cleaned_text = clean_vietnamese_text(raw_text)
        
        page_num = page_idx + 1
        
        # Xác định section dựa vào số trang và nội dung
        section = "Tổng quan"
        if 4 <= page_num <= 9:
            section = "Thông tin về an toàn"
        elif 10 <= page_num <= 19:
            section = "Lắp đặt máy giặt"
        elif 20 <= page_num <= 25:
            section = "Chất giặt tẩy & Đồ giặt"
        elif 26 <= page_num <= 35:
            section = "Vận hành & Chương trình giặt & SmartThings"
        elif 36 <= page_num <= 42:
            section = "Bảo trì & Vệ sinh"
        elif 43 <= page_num <= 49:
            section = "Xử lý sự cố & Mã lỗi"
        elif page_num >= 50:
            section = "Thông số kỹ thuật"

        pages_data.append({
            "page": page_num,
            "section": section,
            "text": cleaned_text
        })

    output_file = DATA_DIR / "raw_text_by_page.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pages_data, f, ensure_ascii=False, indent=2)
    
    print(f"[+] Đã lưu {len(pages_data)} trang vào {output_file}")
    return pages_data

if __name__ == "__main__":
    extract_pdf_pages()
