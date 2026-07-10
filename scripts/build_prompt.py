"""Bước 7 — Augmentation & Prompt: ghép các chunk truy xuất được (bước 6)
vào 1 prompt hoàn chỉnh, sẵn sàng gửi cho LLM ở bước 8 (Generation).

Cấu trúc prompt theo đúng mẫu đã thống nhất từ đầu dự án
(quy-trinh-rag-hoc-tap.md) và được xác nhận lại bởi tài liệu tham khảo
(Tài liêu.odt — mục Augmentation & Prompt):
- Context luôn đứng TRƯỚC câu hỏi.
- Instruction yêu cầu rõ: (1) chỉ dùng thông tin trong context, (2) nói
  "không tìm thấy" khi thiếu dữ liệu — chống hallucination, (3) trích dẫn
  nguồn (tên tài liệu + trang) cho mỗi ý quan trọng.

Chưa xử lý token budget phức tạp (nén/loại trùng) ở bước này vì quy mô nhỏ
(chunk trung vị 221 ký tự) — chỉ cắt bớt phòng hờ nếu 1 chunk quá lớn (vd
bảng dữ liệu dài) để tránh 1 chunk chiếm hết context window vô lý.
"""
import json
import sys
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).parent))
from retrieve import get_embed_model, retrieve  # noqa: E402

GOLDEN_PATH = Path(__file__).parent.parent / "data" / "golden_set.json"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "rag_hoc_tap"

MAX_CHUNK_CHARS = 4000  # cat bot neu 1 chunk qua lon (vd bang du lieu dai), phong hoi khong can nen ca context

PROMPT_TEMPLATE = """Bạn là trợ lý trả lời dựa trên tài liệu được cung cấp.
Chỉ sử dụng thông tin trong phần NGỮ CẢNH bên dưới.
Nếu không đủ thông tin để trả lời, hãy nói rõ là không tìm thấy.
Trích dẫn nguồn (tên tài liệu, trang) cho mỗi ý quan trọng.

NGỮ CẢNH:
{context}

CÂU HỎI: {question}

TRẢ LỜI:"""


def format_context(chunks: list[dict]) -> str:
    """Ghép các chunk thành 1 khối NGỮ CẢNH, mỗi chunk kèm nhãn nguồn ngay
    phía trước — để LLM có đủ thông tin trích dẫn đúng theo yêu cầu trong
    instruction. Giữ nguyên thứ tự relevance score mà Chroma đã trả về
    (không cần sắp xếp lại ở quy mô nhỏ này)."""
    blocks = []
    for c in chunks:
        meta = c["metadata"]
        text = c["text"]
        if len(text) > MAX_CHUNK_CHARS:
            text = text[:MAX_CHUNK_CHARS] + " [...]"
        label = f"[Nguồn: {meta['source']}, trang {meta['page']}]"
        blocks.append(f"{label}\n{text}")
    return "\n\n".join(blocks)


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Ghép context + câu hỏi vào PROMPT_TEMPLATE — đầu ra là prompt hoàn
    chỉnh sẵn sàng gửi cho LLM (bước 8, chưa làm ở đây)."""
    context = format_context(chunks)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def main():
    """Demo: lấy 1 câu hỏi thật từ golden set, retrieval top-5 (đã làm ở
    bước 6), rồi ghép thành prompt hoàn chỉnh để xem thử kết quả thật."""
    golden = json.load(open(GOLDEN_PATH, encoding="utf-8"))
    g = golden[0]

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    model = get_embed_model()

    chunks = retrieve(collection, model, g["question"], k=5)
    prompt = build_prompt(g["question"], chunks)

    print(prompt)


if __name__ == "__main__":
    main()
