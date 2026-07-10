"""Bước 10 (nâng cao) — Reranker: chấm điểm lại các chunk ứng viên (lấy từ
Hybrid Search) bằng cross-encoder local, chọn ra top-k chính xác nhất.

Lý do dùng cross-encoder LOCAL (BAAI/bge-reranker-v2-m3) thay vì LLM-based
reranking: giữ đúng triết lý "miễn phí, không phụ thuộc quota" xuyên suốt
dự án — đặc biệt quan trọng sau khi đã hết quota Gemini nhiều lần ở bước 9.
Cùng họ BGE với model embedding (bge-m3) đã dùng, quen thuộc và đáng tin.

Khác với embedding (encode câu hỏi và chunk riêng biệt rồi so cosine),
cross-encoder đọc CẢ CẶP (câu hỏi, chunk) CÙNG LÚC qua model — chậm hơn
nhiều (không thể tính trước/cache như embedding) nhưng chính xác hơn hẳn vì
model "thấy" được tương tác trực tiếp giữa 2 đoạn text, đúng lý do tài liệu
tham khảo gọi đây là kiến trúc "Retriever ưu tiên tốc độ, Reranker ưu tiên
độ chính xác".

Pipeline đầy đủ: Hybrid Search lấy top-30 (rẻ, nhanh) -> Reranker chấm lại
từng cặp (query, chunk) trong 30 ứng viên đó -> chọn top-5 cuối cùng gửi
cho Generation (bước 7-8).
"""
import sys
from pathlib import Path

import chromadb
from sentence_transformers import CrossEncoder

sys.path.insert(0, str(Path(__file__).parent))
from hybrid_retrieve import hybrid_retrieve  # noqa: E402
from retrieve import get_embed_model  # noqa: E402

CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "rag_hoc_tap"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
CANDIDATES_K = 30  # so ung vien lay tu hybrid search truoc khi rerank
FINAL_K = 5        # so ket qua cuoi cung sau rerank

_RERANKER = None


def get_reranker() -> CrossEncoder:
    """Lazy-load cross-encoder — chỉ load khi thực sự cần rerank."""
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoder(RERANKER_MODEL, device="cpu")
    return _RERANKER


def rerank(query: str, candidates: list[dict], k: int = FINAL_K) -> list[dict]:
    """Chấm điểm từng (query, chunk) bằng cross-encoder, sắp xếp giảm dần,
    trả về top-k. Điểm reranker không phải xác suất chuẩn hoá, chỉ dùng để
    XẾP HẠNG tương đối giữa các ứng viên, không so sánh được giữa các câu
    hỏi khác nhau."""
    if not candidates:
        return []
    model = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [{**c, "rerank_score": float(score)} for c, score in ranked[:k]]


def hybrid_retrieve_and_rerank(collection, embed_model, query: str, k: int = FINAL_K) -> list[dict]:
    """Pipeline đầy đủ: Hybrid Search (top-CANDIDATES_K, rẻ) -> Reranker
    (chấm lại, chậm hơn nhưng chỉ chạy trên CANDIDATES_K ứng viên chứ không
    phải toàn bộ corpus) -> top-k cuối cùng."""
    candidates = hybrid_retrieve(collection, embed_model, query, CANDIDATES_K)
    return rerank(query, candidates, k)


def main():
    """Demo: so sánh top-5 của dense-only, hybrid, và hybrid+rerank cho
    cùng 1 câu hỏi thật để thấy rõ thứ tự kết quả thay đổi thế nào qua
    từng tầng."""
    from retrieve import retrieve

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    embed_model = get_embed_model()

    query = "Grammarly ra mắt vào năm nào và nó hỗ trợ người dùng những gì?"

    print("=== DENSE-ONLY top-5 ===")
    for r in retrieve(collection, embed_model, query, 5):
        print(f"  {r['metadata']['source']} (trang {r['metadata']['page']}, distance={r['distance']:.3f})")

    print("\n=== HYBRID + RERANK top-5 ===")
    for r in hybrid_retrieve_and_rerank(collection, embed_model, query, 5):
        print(f"  {r['metadata']['source']} (trang {r['metadata']['page']}, rerank_score={r['rerank_score']:.3f})")


if __name__ == "__main__":
    main()
