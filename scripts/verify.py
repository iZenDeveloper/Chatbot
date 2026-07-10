"""Bước 10 (nâng cao) — Answer Verification: sau khi sinh câu trả lời (bước
8), gọi LLM thêm 1 lần để kiểm tra lại Faithfulness (câu trả lời có bám
sát context, không bịa thêm) trước khi trả về người dùng cuối.

Khác với evaluate.py (bước 9) — nơi LLM-judge so sánh với `expected_answer`
(chỉ có trong golden set, dùng để ĐO LƯỜNG chất lượng hệ thống) — ở đây
verification chạy trong PRODUCTION, không có đáp án chuẩn để so sánh, nên
chỉ kiểm tra được Faithfulness (câu trả lời có khớp với context đã truy
xuất hay không), không kiểm tra được Correctness (đúng thực tế hay không).

Nền tảng: đã kiểm chứng hành vi LLM-judge ở bước 9 (đồng thuận 100% với
fuzzy match cho Correctness, bắt được cả trường hợp fuzzy bỏ sót ở
Faithfulness) — nên không cần đánh giá lại quy mô lớn, chỉ cần đóng gói
thành 1 bước trong pipeline.

Nếu verification FAIL: thử sinh lại 1 lần với prompt nhắc nhở chặt hơn.
Nếu vẫn FAIL: trả về câu trả lời kèm cảnh báo rõ ràng, KHÔNG che giấu vấn
đề — đúng tinh thần chống hallucination xuyên suốt dự án.
"""
import sys
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).parent))
from _common import extract_text  # noqa: E402
from build_prompt import build_prompt  # noqa: E402
from generate import answer_question, get_llm  # noqa: E402
from retrieve import get_embed_model  # noqa: E402

CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "rag_hoc_tap"
MAX_RETRIES = 1  # so lan thu sinh lai neu verification FAIL

VERIFY_PROMPT = """Bạn là giám khảo kiểm tra xem câu trả lời của 1 hệ thống hỏi-đáp có bám sát tài liệu hay không.

CÂU HỎI: {question}

NGỮ CẢNH (context) đã cung cấp cho hệ thống:
{context}

CÂU TRẢ LỜI hệ thống đã sinh ra:
{answer}

Câu trả lời có HOÀN TOÀN dựa trên NGỮ CẢNH ở trên không, không bịa thêm thông tin nào ngoài context?
Chỉ trả lời đúng định dạng sau, không giải thích gì thêm:
FAITHFULNESS: CÓ/KHÔNG"""

RETRY_SUFFIX = "\n\nLƯU Ý QUAN TRỌNG: Câu trả lời trước đó có thể chứa thông tin không có trong NGỮ CẢNH. Hãy trả lời lại, CHỈ dùng thông tin có trong NGỮ CẢNH, và nói rõ \"không tìm thấy\" cho phần nào context không hỗ trợ."


def verify_faithfulness(llm, question: str, answer: str, chunks: list[dict]) -> bool:
    """Gọi LLM 1 lần để chấm Faithfulness — chỉ CÓ/KHÔNG, không cần
    expected_answer (không có sẵn lúc chạy thật, khác evaluate.py bước 9)."""
    context_text = "\n\n".join(c["text"][:1000] for c in chunks)
    prompt = VERIFY_PROMPT.format(question=question, context=context_text, answer=answer)
    response = llm.invoke(prompt)
    return "FAITHFULNESS: CÓ" in extract_text(response).upper()


def answer_with_verification(collection, embed_model, llm, question: str, k: int = 5, retrieve_fn=None) -> dict:
    """Pipeline đầy đủ: sinh câu trả lời (bước 8) -> verify Faithfulness ->
    nếu FAIL thì thử sinh lại tối đa MAX_RETRIES lần với prompt nhắc nhở
    chặt hơn -> nếu vẫn FAIL thì trả về kèm cảnh báo rõ ràng thay vì che
    giấu vấn đề.

    `retrieve_fn` truyền thẳng xuống answer_question() — cho phép app dùng
    Hybrid+Rerank (bước 10) thay vì dense-only mặc định."""
    answer, chunks = answer_question(collection, embed_model, llm, question, k, retrieve_fn=retrieve_fn)
    verified = verify_faithfulness(llm, question, answer, chunks)
    attempts = 1

    while not verified and attempts <= MAX_RETRIES:
        prompt = build_prompt(question, chunks) + RETRY_SUFFIX
        response = llm.invoke(prompt)
        answer = extract_text(response)
        verified = verify_faithfulness(llm, question, answer, chunks)
        attempts += 1

    return {
        "answer": answer,
        "chunks": chunks,
        "verified": verified,
        "attempts": attempts,
        "warning": None if verified else "⚠️ Câu trả lời chưa được xác nhận bám sát hoàn toàn vào tài liệu — hãy tự kiểm tra lại với các chunk nguồn bên dưới.",
    }


def main():
    """Demo: chạy verification thật trên vài câu hỏi, in kết quả kèm số
    lần thử và cảnh báo (nếu có)."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    embed_model = get_embed_model()
    llm = get_llm()

    questions = [
        "Grammarly ra mắt vào năm nào và nó hỗ trợ người dùng những gì?",
        "RAGPerf tách quy trình RAG thành những thành phần mô-đun nào?",
    ]

    for q in questions:
        result = answer_with_verification(collection, embed_model, llm, q)
        print(f"CÂU HỎI: {q}")
        print(f"TRẢ LỜI ({result['attempts']} lần thử, verified={result['verified']}):\n{result['answer']}")
        if result["warning"]:
            print(result["warning"])
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
