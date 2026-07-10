"""Bước 9 — Evaluation: đo chất lượng Generation trên golden set bằng 2
cách, rồi so sánh mức độ đồng thuận giữa chúng.

Cách A — Fuzzy match (miễn phí, không giới hạn số câu):
- Answer Correctness: so khớp mờ theo từ giữa câu trả lời sinh ra và
  `expected_answer` có sẵn trong golden set.
- Faithfulness (proxy): so khớp mờ giữa câu trả lời và nội dung các chunk
  đã dùng làm context — kiểm tra câu trả lời có "bám" vào context không,
  mà không cần LLM chấm điểm.

Cách B — LLM-as-judge (theo gợi ý tài liệu tham khảo, nhưng tốn quota API
— chỉ chạy trên 1 tập con nhỏ để tránh hết quota giữa chừng, đã 2 lần gặp
RESOURCE_EXHAUSTED trong dự án này):
- Gọi Gemini, đưa vào (câu hỏi, context, câu trả lời, expected_answer),
  yêu cầu chấm Faithfulness và Correctness dạng có/không.

So sánh: 2 cách có đồng thuận với nhau không, trên cùng 1 tập câu hỏi.
"""
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).parent))
from _common import extract_text  # noqa: E402
from generate import answer_question, get_llm  # noqa: E402
from retrieve import get_embed_model  # noqa: E402

GOLDEN_PATH = Path(__file__).parent.parent / "data" / "golden_set.json"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "rag_hoc_tap"

START_INDEX = 46  # gs-47: 19 cau moi them cho cac tai lieu vua bo sung, chua duoc danh gia Generation lan nao
N_FUZZY = 65 - START_INDEX  # chay het 19 cau moi, uu tien Cach A (mien phi ve mat judge)
N_JUDGE = 3    # giu it lai de van co du lieu so sanh, uu tien tiet kiem quota cho Cach A chay duoc nhieu cau hon
MATCH_RATIO_THRESHOLD = 0.4  # nguong thap hon retrieve.py vi so cau tra loi (dien dat tu do) voi expected_answer (tom tat), khong phai trich nguyen van


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def fuzzy_ratio(a: str, b: str) -> float:
    """Ty le tu cua `a` xuat hien trong `b`, cong don moi doan khop (ky
    thuat da kiem chung o buoc 3/6)."""
    a_words, b_words = normalize(a).split(), normalize(b).split()
    if not a_words:
        return 1.0
    sm = SequenceMatcher(None, a_words, b_words, autojunk=False)
    matched = sum(block.size for block in sm.get_matching_blocks())
    return matched / len(a_words)


def fuzzy_evaluate(question: str, answer: str, expected_answer: str, chunks: list[dict]) -> dict:
    """Cách A: Answer Correctness (so voi expected_answer) va Faithfulness
    proxy (so voi context da dung), khong goi LLM."""
    correctness = fuzzy_ratio(expected_answer, answer)
    context_text = " ".join(c["text"] for c in chunks)
    faithfulness = fuzzy_ratio(answer, context_text)
    return {
        "correctness_score": correctness,
        "correctness_pass": correctness >= MATCH_RATIO_THRESHOLD,
        "faithfulness_score": faithfulness,
        "faithfulness_pass": faithfulness >= MATCH_RATIO_THRESHOLD,
    }


JUDGE_PROMPT = """Bạn là giám khảo đánh giá chất lượng câu trả lời của 1 hệ thống hỏi-đáp dựa trên tài liệu (RAG).

CÂU HỎI: {question}

NGỮ CẢNH (context) hệ thống đã dùng để trả lời:
{context}

CÂU TRẢ LỜI hệ thống đã sinh ra:
{answer}

ĐÁP ÁN THAM KHẢO (do con người soạn trước, có thể diễn đạt khác):
{expected_answer}

Hãy đánh giá 2 tiêu chí sau, mỗi tiêu chí trả lời CÓ hoặc KHÔNG:
1. FAITHFULNESS: câu trả lời có hoàn toàn dựa trên NGỮ CẢNH, không bịa thêm thông tin ngoài context không?
2. CORRECTNESS: nội dung câu trả lời có đúng/khớp với ĐÁP ÁN THAM KHẢO không (không cần giống hệt câu chữ)?

Chỉ trả lời đúng định dạng sau, không giải thích gì thêm:
FAITHFULNESS: CÓ/KHÔNG
CORRECTNESS: CÓ/KHÔNG"""


def llm_judge_evaluate(llm, question: str, answer: str, expected_answer: str, chunks: list[dict]) -> dict:
    """Cách B: nhờ Gemini chấm điểm Faithfulness/Correctness — tốn 1 lần
    gọi LLM mỗi câu, ngoài lần gọi để sinh câu trả lời."""
    context_text = "\n\n".join(c["text"][:1000] for c in chunks)
    prompt = JUDGE_PROMPT.format(question=question, context=context_text, answer=answer, expected_answer=expected_answer)
    response = llm.invoke(prompt)
    raw = extract_text(response)
    text = raw.upper()
    faithfulness = "FAITHFULNESS: CÓ" in text or "FAITHFULNESS:CÓ" in text
    correctness = "CORRECTNESS: CÓ" in text or "CORRECTNESS:CÓ" in text
    return {"faithfulness_pass": faithfulness, "correctness_pass": correctness, "raw": raw}


def main():
    golden = json.load(open(GOLDEN_PATH, encoding="utf-8"))[START_INDEX:START_INDEX + N_FUZZY]

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    embed_model = get_embed_model()
    llm = get_llm()

    fuzzy_results = []
    judge_results = []
    agree_count = 0
    judge_attempted = 0

    for i, g in enumerate(golden):
        try:
            answer, chunks = answer_question(collection, embed_model, llm, g["question"])
        except Exception as e:
            print(f"[{g['id']}] LOI khi sinh cau tra loi: {e}")
            continue

        fuzzy = fuzzy_evaluate(g["question"], answer, g["expected_answer"], chunks)
        fuzzy_results.append(fuzzy)
        print(f"[{g['id']}] Fuzzy -> correctness={fuzzy['correctness_score']:.2f} "
              f"({'PASS' if fuzzy['correctness_pass'] else 'FAIL'}), "
              f"faithfulness={fuzzy['faithfulness_score']:.2f} ({'PASS' if fuzzy['faithfulness_pass'] else 'FAIL'})")

        if i < N_JUDGE:
            judge_attempted += 1
            try:
                judge = llm_judge_evaluate(llm, g["question"], answer, g["expected_answer"], chunks)
                judge_results.append(judge)
                print(f"       LLM-judge -> faithfulness={'PASS' if judge['faithfulness_pass'] else 'FAIL'}, "
                      f"correctness={'PASS' if judge['correctness_pass'] else 'FAIL'}")
                if judge["correctness_pass"] == fuzzy["correctness_pass"] and judge["faithfulness_pass"] == fuzzy["faithfulness_pass"]:
                    agree_count += 1
            except Exception as e:
                print(f"       [LOI LLM-judge] {e}")

    n = len(fuzzy_results)
    print(f"\n=== CÁCH A (Fuzzy match, {n} câu) ===")
    print(f"Answer Correctness: {sum(r['correctness_pass'] for r in fuzzy_results)}/{n}")
    print(f"Faithfulness (proxy): {sum(r['faithfulness_pass'] for r in fuzzy_results)}/{n}")

    if judge_results:
        m = len(judge_results)
        print(f"\n=== CÁCH B (LLM-as-judge, {m} câu) ===")
        print(f"Faithfulness: {sum(r['faithfulness_pass'] for r in judge_results)}/{m}")
        print(f"Correctness: {sum(r['correctness_pass'] for r in judge_results)}/{m}")
        print(f"\nMức đồng thuận giữa 2 cách (cả 2 tiêu chí khớp nhau): {agree_count}/{judge_attempted}")


if __name__ == "__main__":
    main()
