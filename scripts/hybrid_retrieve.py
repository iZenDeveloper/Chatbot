"""Bước 10 (nâng cao) — Hybrid Search: kết hợp BM25 (keyword) với Dense
retrieval (bước 6) bằng Reciprocal Rank Fusion (RRF).

Lý do ưu tiên kỹ thuật này trước (đã thống nhất với người dùng): hoàn toàn
miễn phí, chạy local (rank_bm25 thuần Python, không cần LLM) — tránh vấn đề
hết quota Gemini đã gặp nhiều lần ở các bước trước.

Bằng chứng từ bước 6/9 cho thấy dense retrieval một mình có Precision@k
trung bình (0.452-0.645) dù MRR cao (0.875) — nhiều chunk "gần đúng chủ đề"
gây nhiễu. BM25 bổ sung tín hiệu từ khoá chính xác (vd tên riêng, thuật
ngữ kỹ thuật) mà embedding đôi khi bỏ qua vì quá chú trọng ngữ nghĩa tổng
quát.
"""
import json
import re
import sys
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).parent))
from retrieve import get_embed_model, retrieve  # noqa: E402

CHUNKS_PATH = Path(__file__).parent.parent / "data" / "processed" / "chunks.jsonl"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "rag_hoc_tap"

DENSE_K = 30   # so ket qua dense lay ra de fusion (rong hon k cuoi cung de RRF co du lieu chon loc)
BM25_K = 30    # tuong tu, cho BM25
RRF_K = 60     # hang so lam min trong cong thuc RRF: 1/(RRF_K + rank) — gia tri pho bien trong tai lieu ve RRF

_BM25_INDEX = None
_BM25_CHUNKS = None


def tokenize(text: str) -> list[str]:
    """Tách từ đơn giản cho BM25: chữ thường, chỉ giữ chữ/số. Không cần
    stemming/stopword phức tạp vì BM25 vẫn hoạt động tốt với tokenize thô,
    và corpus song ngữ Việt-Anh khó áp dụng 1 bộ stemmer chung."""
    return re.findall(r"\w+", text.lower())


def get_bm25_index():
    """Xây (hoặc tái sử dụng) chỉ mục BM25 trên toàn bộ chunks.jsonl. Lazy
    -load 1 lần vì tokenize + xây chỉ mục cho ~7000+ chunk tốn vài giây,
    không cần làm lại mỗi lần gọi hybrid_retrieve()."""
    global _BM25_INDEX, _BM25_CHUNKS
    if _BM25_INDEX is None:
        chunks = [json.loads(line) for line in open(CHUNKS_PATH, encoding="utf-8")]
        tokenized = [tokenize(c["text"]) for c in chunks]
        _BM25_INDEX = BM25Okapi(tokenized)
        _BM25_CHUNKS = chunks
    return _BM25_INDEX, _BM25_CHUNKS


def bm25_search(query: str, k: int) -> list[dict]:
    """Tìm top-k chunk theo BM25 (khớp từ khoá), trả về cùng format với
    retrieve() (dense) để 2 danh sách gộp được với nhau."""
    index, chunks = get_bm25_index()
    scores = index.get_scores(tokenize(query))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [{"text": chunks[i]["text"], "metadata": chunks[i]["metadata"], "bm25_score": scores[i]} for i in top_idx]


def chunk_key(chunk: dict) -> tuple:
    """Khoá định danh 1 chunk để so khớp giữa 2 danh sách dense/BM25 (vốn
    không có ID chung) — dùng (source, page, 80 ký tự đầu) làm proxy vì đủ
    phân biệt trong corpus hiện tại."""
    meta = chunk["metadata"]
    return (meta["source"], meta["page"], chunk["text"][:80])


def hybrid_retrieve(collection, embed_model, query: str, k: int = 5) -> list[dict]:
    """Lấy top-DENSE_K từ dense retrieval + top-BM25_K từ BM25, gộp bằng
    Reciprocal Rank Fusion: mỗi chunk được cộng điểm 1/(RRF_K + hạng) từ
    MỖI danh sách nó xuất hiện — chunk được cả 2 phương pháp đồng thuận xếp
    cao sẽ được đẩy lên đầu, mạnh hơn chỉ dùng 1 tín hiệu riêng lẻ."""
    dense_results = retrieve(collection, embed_model, query, DENSE_K)
    bm25_results = bm25_search(query, BM25_K)

    rrf_scores: dict[tuple, float] = {}
    chunk_by_key: dict[tuple, dict] = {}

    for rank, r in enumerate(dense_results, start=1):
        key = chunk_key(r)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
        chunk_by_key[key] = r

    for rank, r in enumerate(bm25_results, start=1):
        key = chunk_key(r)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
        chunk_by_key.setdefault(key, r)

    ranked_keys = sorted(rrf_scores, key=lambda key: rrf_scores[key], reverse=True)
    return [{**chunk_by_key[key], "rrf_score": rrf_scores[key]} for key in ranked_keys[:k]]


def main():
    """Demo: so sánh top-5 dense-only vs hybrid cho 1 câu hỏi thật, để xem
    thứ tự kết quả có đổi không."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    embed_model = get_embed_model()

    query = "Grammarly ra mắt vào năm nào và nó hỗ trợ người dùng những gì?"

    print("=== DENSE-ONLY top-5 ===")
    for r in retrieve(collection, embed_model, query, 5):
        print(f"  {r['metadata']['source']} (trang {r['metadata']['page']}, distance={r['distance']:.3f})")

    print("\n=== HYBRID (BM25+Dense, RRF) top-5 ===")
    for r in hybrid_retrieve(collection, embed_model, query, 5):
        print(f"  {r['metadata']['source']} (trang {r['metadata']['page']}, rrf_score={r['rrf_score']:.4f})")


if __name__ == "__main__":
    main()
