"""Bước 10 (nâng cao, ưu tiên 3) — Multi-query: nhờ LLM sinh thêm vài cách
diễn đạt lại câu hỏi gốc, retrieval riêng cho từng biến thể rồi hợp nhất
bằng Reciprocal Rank Fusion (giống cơ chế đã dùng ở Hybrid Search).

Lý do ưu tiên kỹ thuật này (trong nhóm cần LLM): chỉ tốn **1 lần gọi LLM/
câu hỏi** để sinh biến thể — rẻ hơn nhiều so với Generation+LLM-judge (2
lần/câu) đã làm cạn quota ở bước 9. Trực tiếp giải quyết vấn đề đã ghi
nhận ở bước 8: câu hỏi người dùng diễn đạt khác với câu chữ trong tài liệu
gốc khiến dense retrieval bỏ sót.

Xoay vòng nhiều model Gemini khi 1 model hết quota (đã học từ các bước
trước — quota free tier tính riêng theo từng model), để tối đa số câu
đánh giá được trong ngày.
"""
import json
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

sys.path.insert(0, str(Path(__file__).parent))
from _common import extract_text  # noqa: E402
from hybrid_retrieve import chunk_key  # noqa: E402
from retrieve import K_VALUES, evaluate_recall_at_k, get_embed_model, retrieve  # noqa: E402

GOLDEN_PATH = Path(__file__).parent.parent / "data" / "golden_set.json"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "rag_hoc_tap"

# Xoay vong theo thu tu nay khi mot model het quota (da kiem tra con quota luc viet script)
CANDIDATE_MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-flash-latest"]
N_VARIANTS = 2  # so cau hoi dien dat lai them (ngoai cau goc), giu nho de tiet kiem quota
RRF_K = 60
PER_QUERY_K = 15  # so ket qua dense lay cho MOI bien the truoc khi hop nhat

load_dotenv(Path(__file__).parent.parent / ".env")

_LLM = None
_LLM_MODEL_USED = None


def get_llm_with_fallback():
    """Trả về (llm, ten_model) — thử từng model trong CANDIDATE_MODELS,
    dùng lại model đầu tiên còn quota cho các lần gọi sau (không thử lại
    từ đầu mỗi câu hỏi, tốn thời gian)."""
    global _LLM, _LLM_MODEL_USED
    if _LLM is not None:
        return _LLM, _LLM_MODEL_USED
    for m in CANDIDATE_MODELS:
        try:
            llm = ChatGoogleGenerativeAI(model=m, temperature=0.3)
            llm.invoke("test")  # xac nhan con quota that, khong chi khoi tao thanh cong
            _LLM, _LLM_MODEL_USED = llm, m
            return llm, m
        except Exception:
            continue
    raise RuntimeError("Ca 3 model deu het quota, khong the dung Multi-query luc nay.")


QUERY_EXPANSION_PROMPT = """Cho câu hỏi sau, hãy viết lại thành {n} câu hỏi khác có CÙNG Ý NGHĨA nhưng dùng từ ngữ/cách diễn đạt khác (đồng nghĩa, đổi cấu trúc câu, thêm từ khoá kỹ thuật liên quan nếu phù hợp). Mục đích là tăng khả năng tìm đúng tài liệu khi câu hỏi gốc dùng từ ngữ khác với tài liệu.

CÂU HỎI GỐC: {question}

Chỉ trả lời đúng {n} dòng, mỗi dòng 1 câu hỏi diễn đạt lại, không đánh số, không giải thích."""


def expand_query(llm, question: str, n: int = N_VARIANTS) -> list[str]:
    """Sinh n câu hỏi diễn đạt lại. Nếu LLM lỗi/hết quota, trả về danh sách
    rỗng — hàm gọi (multi_query_retrieve) sẽ tự fallback về chỉ dùng câu
    hỏi gốc, không làm crash cả pipeline."""
    try:
        prompt = QUERY_EXPANSION_PROMPT.format(n=n, question=question)
        response = llm.invoke(prompt)
        variants = [line.strip() for line in extract_text(response).strip().split("\n") if line.strip()]
        return variants[:n]
    except Exception:
        return []


def multi_query_retrieve(collection, embed_model, llm, question: str, k: int) -> list[dict]:
    """Retrieval trên câu hỏi gốc + các biến thể do LLM sinh, hợp nhất
    bằng RRF (cùng cơ chế hybrid_retrieve.py) — chunk được nhiều biến thể
    câu hỏi cùng tìm ra sẽ được đẩy lên đầu."""
    queries = [question] + expand_query(llm, question)

    rrf_scores: dict[tuple, float] = {}
    chunk_by_key: dict[tuple, dict] = {}

    for q in queries:
        results = retrieve(collection, embed_model, q, PER_QUERY_K)
        for rank, r in enumerate(results, start=1):
            key = chunk_key(r)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            chunk_by_key.setdefault(key, r)

    ranked_keys = sorted(rrf_scores, key=lambda key: rrf_scores[key], reverse=True)
    return [chunk_by_key[key] for key in ranked_keys[:k]]


def main():
    golden = json.load(open(GOLDEN_PATH, encoding="utf-8"))
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    embed_model = get_embed_model()

    llm, model_used = get_llm_with_fallback()
    print(f"Dung model: {model_used}\n")

    dense_fn = lambda q, k: retrieve(collection, embed_model, q, k)  # noqa: E731
    multi_query_fn = lambda q, k: multi_query_retrieve(collection, embed_model, llm, q, k)  # noqa: E731

    print(f"Danh gia tren {len(golden)} cau hoi...")
    dense_result = evaluate_recall_at_k(dense_fn, golden, K_VALUES)
    mq_result = evaluate_recall_at_k(multi_query_fn, golden, K_VALUES)

    print("\n=== DENSE-ONLY (baseline) ===")
    for k in K_VALUES:
        print(f"  Recall@{k}: {dense_result['counts'][k]}/{dense_result['total']} ({dense_result['recall'][k]*100:.1f}%)")
    print(f"  MRR: {dense_result['mrr']:.3f}")

    print("\n=== MULTI-QUERY ===")
    for k in K_VALUES:
        print(f"  Recall@{k}: {mq_result['counts'][k]}/{mq_result['total']} ({mq_result['recall'][k]*100:.1f}%)")
    print(f"  MRR: {mq_result['mrr']:.3f}")


if __name__ == "__main__":
    main()
