"""Bước 8 — Generation: ghép nối toàn bộ pipeline (retrieval bước 6 + prompt
bước 7) rồi gọi LLM để sinh câu trả lời cuối cùng.

Quyết định kỹ thuật:
- Model: Google Gemini (`gemini-2.5-flash`) — đã dùng ở bước đổi tên tài
  liệu, miễn phí, nhất quán trong toàn dự án.
- Temperature=0.2 (thấp) — theo đúng khuyến nghị của tài liệu tham khảo và
  kế hoạch gốc: giảm suy đoán, bám sát context cho bài toán hỏi-đáp dựa
  trên tài liệu (khác với các tác vụ cần sáng tạo).
- Luôn in kèm cả câu trả lời VÀ danh sách chunk đã dùng — để tự kiểm tra
  bằng mắt xem câu trả lời có "grounded" (bám sát context) hay không, đúng
  gợi ý của kế hoạch gốc. Chưa làm Answer Verification bằng LLM thứ 2 hay
  structured output — để dành bước 10 nâng cao.
"""
import json
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

sys.path.insert(0, str(Path(__file__).parent))
from _common import extract_text  # noqa: E402
from build_prompt import build_prompt  # noqa: E402
from retrieve import get_embed_model, retrieve  # noqa: E402

CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "rag_hoc_tap"
LLM_MODEL = "gemini-flash-latest"  # cac model 2.5-flash/2.5-flash-lite/2.0-flash/2.0-flash-lite het quota, model nay con quota rieng tai thoi diem nay
TEMPERATURE = 0.2
DEFAULT_K = 5

load_dotenv(Path(__file__).parent.parent / ".env")

_LLM = None


def get_llm() -> ChatGoogleGenerativeAI:
    """Lazy-load LLM client — chỉ khởi tạo khi thực sự cần gọi generation."""
    global _LLM
    if _LLM is None:
        _LLM = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=TEMPERATURE)
    return _LLM


def answer_question(collection, embed_model, llm, question: str, k: int = DEFAULT_K, retrieve_fn=None) -> tuple[str, list[dict]]:
    """Chạy trọn pipeline: retrieval (bước 6) -> build prompt (bước 7) ->
    gọi LLM (bước 8). Trả về (câu trả lời, danh sách chunk đã dùng) để
    người dùng tự kiểm tra faithfulness.

    `retrieve_fn(collection, embed_model, question, k) -> list[dict]` cho
    phép cắm vào bất kỳ pipeline retrieval nào (dense-only mặc định, hoặc
    Hybrid+Rerank ở bước 10) mà không cần sửa hàm này — mặc định `None` sẽ
    dùng dense-only (`retrieve()`) để không phá vỡ các chỗ đang gọi hàm
    này (evaluate.py, verify.py demo cũ)."""
    if retrieve_fn is None:
        chunks = retrieve(collection, embed_model, question, k)
    else:
        chunks = retrieve_fn(collection, embed_model, question, k)
    prompt = build_prompt(question, chunks)
    response = llm.invoke(prompt)
    return extract_text(response), chunks


def print_result(question: str, answer: str, chunks: list[dict]):
    print(f"CÂU HỎI: {question}\n")
    print(f"TRẢ LỜI:\n{answer}\n")
    print("CHUNK ĐÃ DÙNG (để tự kiểm tra faithfulness):")
    for i, c in enumerate(chunks, start=1):
        meta = c["metadata"]
        print(f"  [{i}] {meta['source']} (trang {meta['page']}, khoảng cách {c['distance']:.3f})")
    print("\n" + "=" * 80 + "\n")


def main():
    """Demo: chạy vài câu hỏi thật (từ golden set + 1 câu ngoài corpus để
    kiểm tra khả năng từ chối trả lời khi thiếu context)."""
    golden_path = Path(__file__).parent.parent / "data" / "golden_set.json"
    golden = json.load(open(golden_path, encoding="utf-8"))

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    embed_model = get_embed_model()
    llm = get_llm()

    demo_questions = [golden[0]["question"], golden[33]["question"]]  # gs-01 (retrieval tot), gs-34 (biet la mieng chunk)
    demo_questions.append("Thủ đô của nước Pháp là gì?")  # cau hoi ngoai corpus, kiem tra tu choi tra loi

    for q in demo_questions:
        answer, chunks = answer_question(collection, embed_model, llm, q)
        print_result(q, answer, chunks)


if __name__ == "__main__":
    main()
