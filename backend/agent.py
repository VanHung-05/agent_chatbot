"""
AI Agent Orchestrator:
1. Multi-turn Query Rewriting: Tự động bổ sung ngữ cảnh cho câu hỏi nối tiếp bằng Qwen2.5.
2. Tool Calling Router: Phân tích ý định người dùng và tự động chọn Tool phù hợp.
3. Chống ảo giác (Anti-hallucination): System prompt nghiêm ngặt, từ chối trả lời nếu thiếu cơ sở trong tài liệu.
"""

import re
import json
import requests
from typing import List, Dict, Generator
from .tools import ErrorCodeTool, SpecsTool, RAGSearchTool

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:7b"

SYSTEM_PROMPT = """Bạn là Trợ lý Kỹ thuật AI chuyên nghiệp hỗ trợ tra cứu Sách Hướng dẫn sử dụng máy giặt cửa trên Samsung (WA12CG5886 / WA14CG5886).

QUY TẮC CỐT LÕI BẮT BUỘC:
1. Bạn CHỈ ĐƯỢC PHÉP trả lời dựa trên các thông tin được cung cấp trong phần NGỮ CẢNH HỖ TRỢ hoặc kết quả từ CÔNG CỤ bên dưới.
2. NGUYÊN TẮC CHỐNG BỊA ĐẶT (QUAN TRỌNG): Nếu ngữ cảnh hoặc công cụ KHÔNG chứa thông tin để trả lời câu hỏi, bạn BẮT BUỘC phải nói rõ: "Thông tin này không được tìm thấy trong tài liệu hướng dẫn sử dụng máy giặt Samsung", tuyệt đối KHÔNG tự suy đoán, KHÔNG bịa đặt kiến thức ngoài đời thực.
3. Trình bày rõ ràng:
   - Các quy trình lắp đặt, kiểm tra, bảo trì, vệ sinh phải định dạng theo dạng số thứ tự các bước: Bước 1, Bước 2...
   - Cảnh báo an toàn (cháy nổ, điện giật, ngập nước, đồ giặt cấm): PHẢI in đậm nổi bật.
4. LUÔN trích dẫn số trang tài liệu tương ứng ở cuối câu trả lời (Ví dụ: [Tham khảo: Trang 47]).
"""

class SamsungAgent:
    def __init__(self, embedding_model="finetuned"):
        print("[*] Đang khởi động AI Agent & các công cụ...")
        self.error_tool = ErrorCodeTool()
        self.specs_tool = SpecsTool()
        self.current_model = embedding_model
        self.rag_tool = RAGSearchTool(model_type=embedding_model)
        print("[+] Agent sẵn sàng phục vụ!")

    def set_embedding_model(self, model_name: str):
        if model_name != self.current_model:
            print(f"[*] Chuyển đổi mô hình Embedding sang: {model_name}")
            self.rag_tool = RAGSearchTool(model_type=model_name)
            self.current_model = model_name

    def rewrite_query_with_history(self, current_query: str, history: List[Dict]) -> str:
        """Nếu có lịch sử, dùng Qwen2.5 viết lại câu hỏi hoàn chỉnh nghĩa."""
        if not history or len(history) == 0:
            return current_query

        # Lấy tối đa 2 vòng hội thoại gần nhất
        recent_history = history[-4:]
        hist_text = ""
        for msg in recent_history:
            role = "Người dùng" if msg.get("role") == "user" else "Trợ lý"
            hist_text += f"{role}: {msg.get('content', '')}\n"

        prompt = f"""Dựa vào lịch sử hội thoại gần nhất và câu hỏi mới của người dùng, hãy viết lại thành MỘT câu hỏi độc lập duy nhất có đầy đủ chủ ngữ, vị ngữ và ngữ cảnh rõ ràng về máy giặt Samsung.
- Giữ nguyên các thuật ngữ, tên model, mã lỗi.
- Nếu câu hỏi đã đầy đủ nghĩa, hãy giữ nguyên.
- Chỉ xuất ra ĐÚNG 1 câu hỏi viết lại, không giải thích gì thêm.

Lịch sử:
{hist_text}

Câu hỏi mới: {current_query}
Câu hỏi viết lại:"""

        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            }, timeout=15)
            if resp.status_code == 200:
                rewritten = resp.json().get("response", "").strip()
                # Loại bỏ trích dẫn ngoặc kép nếu có
                rewritten = re.sub(r'^["\']|["\']$', '', rewritten).strip()
                if len(rewritten) > 5:
                    return rewritten
        except Exception as e:
            print(f"[!] Lỗi khi viết lại câu hỏi: {e}")

        return current_query

    def route_intent(self, query: str) -> Dict:
        """Quyết định công cụ nào cần được kích hoạt."""
        # 1. Regex kiểm tra nhanh mã lỗi (4C, 1C, 5C, Ub, dC, dC1, 9C1...)
        error_match = re.search(r'\b(1C|3C|4C|4C2|5C|8C|9C1|9C2|AC6|dC|dC1|HC|LC|OC|PC|PC1|Ub)\b', query, re.IGNORECASE)
        if error_match:
            code = error_match.group(1).upper()
            return {"tool": "error_code_tool", "arg": code}

        # 2. Kiểm tra thông số kỹ thuật
        specs_keywords = ["thông số", "kích thước", "trọng lượng", "khối lượng giặt", "công suất", "tốc độ vắt", "áp lực nước", "wa12", "wa14"]
        if any(k in query.lower() for k in specs_keywords) and ("bao nhiêu" in query.lower() or "là gì" in query.lower() or "thông số" in query.lower()):
            model_key = "14" if "14" in query.lower() else ("12" if "12" in query.lower() else "all")
            return {"tool": "specs_tool", "arg": model_key}

        # 3. Mặc định là tìm kiếm RAG trong sách HDSD
        return {"tool": "rag_tool", "arg": query}

    def run_agent(self, user_message: str, history: List[Dict] = None) -> Generator[Dict, None, None]:
        """
        Agent Pipeline:
        1. Query rewrite
        2. Tool call
        3. Stream generation
        """
        if history is None:
            history = []

        # 1. Viết lại câu hỏi nếu có ngữ cảnh
        standalone_query = self.rewrite_query_with_history(user_message, history)
        if standalone_query != user_message:
            yield {
                "type": "thought",
                "message": f"🔄 Đã hiểu theo ngữ cảnh: \"{standalone_query}\""
            }

        # 2. Phân tích & gọi Tool
        routing = self.route_intent(standalone_query)
        tool_name = routing["tool"]
        tool_arg = routing["arg"]

        context_text = ""
        sources = []

        if tool_name == "error_code_tool":
            yield {"type": "tool_start", "tool": "lookup_error_code", "arg": tool_arg}
            res = self.error_tool.lookup(tool_arg)
            if res.get("found"):
                context_text = (
                    f"BẢNG TRA CỨU MÃ LỖI CẤU TRÚC:\n"
                    f"- Mã lỗi: {res['code']}\n"
                    f"- Nguyên nhân: {res['issue']}\n"
                    f"- Hành động khắc phục: {res['action']}\n"
                    f"- Lưu ý: {res.get('note', '')}\n"
                    f"- Nguồn: Trang {res['page']}"
                )
                sources.append({"page": res["page"], "title": f"Mã lỗi {res['code']}"})
            else:
                context_text = res.get("message", "")
            yield {"type": "tool_end", "tool": "lookup_error_code"}

        elif tool_name == "specs_tool":
            yield {"type": "tool_start", "tool": "get_specifications", "arg": tool_arg}
            specs_data = self.specs_tool.get_specs(tool_arg)
            context_text = f"BẢNG THÔNG SỐ KỸ THUẬT CHÍNH THỨC (Trang 50):\n{json.dumps(specs_data, ensure_ascii=False, indent=2)}"
            sources.append({"page": 50, "title": "Bảng thông số kỹ thuật"})
            yield {"type": "tool_end", "tool": "get_specifications"}

        else:
            yield {"type": "tool_start", "tool": "search_manual_rag", "arg": tool_arg}
            rag_results = self.rag_tool.search(standalone_query, top_k=4)
            context_text = "CÁC ĐOẠN TRÍCH TỪ TÀI LIỆU HƯỚNG DẪN SỬ DỤNG:\n\n"
            for r in rag_results:
                context_text += f"[Trang {r['page']} - {r['section']}]:\n{r['text']}\n\n"
                sources.append({
                    "page": r["page"],
                    "section": r["section"],
                    "title": f"Trang {r['page']} - {r['section']}"
                })
            yield {"type": "tool_end", "tool": "search_manual_rag"}

        # 3. Gửi prompt đến LLM để sinh câu trả lời
        user_prompt = f"""NGỮ CẢNH HỖ TRỢ:
\"\"\"
{context_text}
\"\"\"

CÂU HỎI CỦA NGƯỜI DÙNG: {user_message}

Dựa vào các quy tắc trong System Prompt, hãy trả lời câu hỏi của người dùng một cách chính xác, mạch lạc và đầy đủ."""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        yield {"type": "sources", "sources": sources}

        # Gọi Ollama Chat streaming
        try:
            resp = requests.post(OLLAMA_CHAT_URL, json={
                "model": MODEL_NAME,
                "messages": messages,
                "stream": True,
                "options": {"temperature": 0.2}
            }, stream=True, timeout=60)

            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield {"type": "token", "content": content}
        except Exception as e:
            yield {"type": "error", "message": f"Lỗi sinh câu trả lời từ LLM: {str(e)}"}
