"""
Tool 1: Tra cứu mã lỗi có cấu trúc O(1) từ bảng mã lỗi trang 47-49.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

class ErrorCodeTool:
    def __init__(self):
        json_path = DATA_DIR / "structured_error_codes.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Lưu vào dict để tra cứu O(1) hoa/thường
        self.error_dict = {item["code"].upper(): item for item in data}

    def lookup(self, code: str) -> dict:
        code_clean = code.strip().upper()
        if code_clean in self.error_dict:
            info = self.error_dict[code_clean]
            return {
                "found": True,
                "code": info["code"],
                "issue": info["issue"],
                "action": info["action"],
                "note": info.get("note", ""),
                "page": info["page"]
            }
        
        # Thử tìm tương đối (ví dụ user hỏi lỗi 'DC' hoặc 'DC1')
        for k, v in self.error_dict.items():
            if k in code_clean or code_clean in k:
                return {
                    "found": True,
                    "code": v["code"],
                    "issue": v["issue"],
                    "action": v["action"],
                    "note": v.get("note", ""),
                    "page": v["page"]
                }

        return {
            "found": False,
            "message": f"Không tìm thấy mã lỗi '{code}' trong danh mục mã lỗi máy giặt Samsung."
        }

if __name__ == "__main__":
    tool = ErrorCodeTool()
    print("Test 4C:", tool.lookup("4C"))
    print("Test Ub:", tool.lookup("ub"))
