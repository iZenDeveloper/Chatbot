"""Bước 10 (nâng cao) — so sánh Recall@k/Precision@k/MRR giữa dense-only
(bước 6) và hybrid+rerank (bước 10), trên CÙNG golden set, dùng chung hàm
đánh giá `evaluate_recall_at_k` đã tổng quát hoá trong retrieve.py.

Hoàn toàn miễn phí (BM25 + cross-encoder local), không cần gọi LLM nên
chạy được trên toàn bộ golden set mà không lo hết quota.
"""
import json
import sys
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).parent))
from hybrid_retrieve import hybrid_retrieve  # noqa: E402
from rerank import rerank  # noqa: E402
from retrieve import K_VALUES, evaluate_recall_at_k, get_embed_model, retrieve  # noqa: E402

GOLDEN_PATH = Path(__file__).parent.parent / "data" / "golden_set.json"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "rag_hoc_tap"
CANDIDATES_K = 30  # so ung vien lay tu hybrid truoc khi rerank (khop voi rerank.py)


def make_hybrid_rerank_fn(collection, embed_model):
    """Trả về hàm retrieve_fn(question, k) dùng cho evaluate_recall_at_k:
    lấy top-CANDIDATES_K từ hybrid search 1 LẦN, rerank TOÀN BỘ (không chỉ
    top-k), để evaluate_recall_at_k có thể cắt lấy bao nhiêu k cũng được mà
    không phải chạy lại cross-encoder (đắt) nhiều lần."""
    def fn(question: str, k: int) -> list[dict]:
        candidates = hybrid_retrieve(collection, embed_model, question, CANDIDATES_K)
        return rerank(question, candidates, k=len(candidates))[:k]
    return fn


def print_result(name: str, result: dict):
    print(f"\n=== {name} ===")
    for k in K_VALUES:
        print(f"  Recall@{k}: {result['counts'][k]}/{result['total']} ({result['recall'][k]*100:.1f}%)")
    for k in K_VALUES:
        print(f"  Precision@{k}: {result['precision'][k]:.3f}")
    print(f"  MRR: {result['mrr']:.3f}")


def main():
    golden = json.load(open(GOLDEN_PATH, encoding="utf-8"))
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    embed_model = get_embed_model()

    print(f"So sanh tren {len(golden)} cau hoi, {collection.count()} chunk trong Chroma")

    dense_fn = lambda q, k: retrieve(collection, embed_model, q, k)  # noqa: E731
    dense_result = evaluate_recall_at_k(dense_fn, golden, K_VALUES)
    print_result("DENSE-ONLY (bước 6, baseline)", dense_result)

    hybrid_rerank_fn = make_hybrid_rerank_fn(collection, embed_model)
    hybrid_result = evaluate_recall_at_k(hybrid_rerank_fn, golden, K_VALUES)
    print_result("HYBRID + RERANK (bước 10)", hybrid_result)

    print("\n=== CHÊNH LỆCH (Hybrid+Rerank so với Dense-only) ===")
    for k in K_VALUES:
        d_recall = (hybrid_result["recall"][k] - dense_result["recall"][k]) * 100
        d_prec = hybrid_result["precision"][k] - dense_result["precision"][k]
        print(f"  Recall@{k}: {d_recall:+.1f} điểm % | Precision@{k}: {d_prec:+.3f}")
    print(f"  MRR: {hybrid_result['mrr'] - dense_result['mrr']:+.3f}")


if __name__ == "__main__":
    main()
