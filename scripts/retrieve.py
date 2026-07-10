"""Bước 6 — Retrieval cơ bản: dense search bằng Chroma + đánh giá Recall@k
trên golden set.

Chiến lược (đã thống nhất với người dùng: "cơ bản trước, nâng cao sau" —
hybrid BM25+dense, reranker, MMR, HyDE để dành cho bước 10 nâng cao):
1. Embed câu hỏi bằng ĐÚNG model đã dùng để index (BAAI/bge-m3) — bắt buộc
   nhất quán giữa index và query (đã nhấn mạnh ở bước 4).
2. Query Chroma lấy top-k, so sánh k = 3, 5, 10 (theo gợi ý kế hoạch gốc).
3. Đánh giá bằng Recall@k thật trên golden_set.json: với mỗi câu hỏi, kiểm
   tra xem trong top-k có chunk nào (a) đến từ đúng `source` VÀ (b) chứa
   đủ nội dung `context_snippet` (so khớp mờ theo từ, giống cách đã dùng
   để so sánh chunking ở bước 3) hay không.
"""
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

GOLDEN_PATH = Path(__file__).parent.parent / "data" / "golden_set.json"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "rag_hoc_tap"
EMBED_MODEL_NAME = "BAAI/bge-m3"
K_VALUES = [3, 5, 10]
MATCH_RATIO_THRESHOLD = 0.7

_EMBED_MODEL = None


def get_embed_model() -> SentenceTransformer:
    """Lazy-load model — dùng device="cpu" giống bước 4 để tránh CUDA OOM
    đã gặp trên GPU nhỏ của máy này."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    return _EMBED_MODEL


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def snippet_parts(snippet: str) -> list[str]:
    """Tách context_snippet theo "..." — phần bị lược bớt không dùng để so khớp."""
    return [normalize(p) for p in snippet.split("...") if normalize(p)]


def fuzzy_contains(part: str, chunk_text: str) -> bool:
    """So khớp theo từ, cộng dồn mọi đoạn khớp (không chỉ đoạn dài nhất) —
    bền với sai khác định dạng nhỏ. Đã kiểm chứng cách này ở bước 3 (so
    sánh chunking) sau khi phát hiện SequenceMatcher theo ký tự cho kết
    quả sai trên văn bản dài.

    Fallback bỏ khoảng trắng: một số PDF tiếng Việt (phát hiện khi debug
    bước 6) bị Docling trích xuất với khoảng trắng chèn lạc giữa các chữ
    có dấu (vd "tiến hành" → "tiế n hành") do lỗi kerning font trong PDF
    gốc — làm gãy so khớp theo từ dù nội dung ký tự giống hệt nhau. Nếu
    so khớp theo từ thất bại, thử lại sau khi bỏ hết khoảng trắng ở cả 2
    bên trước khi kết luận không khớp."""
    if not part:
        return True
    part_words = part.split()
    chunk_words = chunk_text.split()
    if not part_words:
        return True
    sm = SequenceMatcher(None, part_words, chunk_words, autojunk=False)
    matched = sum(block.size for block in sm.get_matching_blocks())
    if (matched / len(part_words)) >= MATCH_RATIO_THRESHOLD:
        return True

    part_nospace = part.replace(" ", "")
    chunk_nospace = chunk_text.replace(" ", "")
    return part_nospace in chunk_nospace


def snippet_found_in_chunk(snippet: str, chunk_text: str) -> bool:
    parts = snippet_parts(snippet)
    chunk_text = normalize(chunk_text)
    return all(fuzzy_contains(part, chunk_text) for part in parts)


def retrieve(collection, model, query: str, k: int) -> list[dict]:
    """Embed câu hỏi rồi truy vấn Chroma, trả về top-k kèm text+metadata."""
    q_emb = model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=k)
    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ]


def evaluate_recall_at_k(retrieve_fn, golden: list[dict], k_values: list[int]) -> dict:
    """Đánh giá đầy đủ 3 chỉ số tầng Retrieval mà tài liệu tham khảo liệt kê
    (Recall@K, Precision@K, MRR — nDCG bỏ qua có chủ đích, xem ghi chú dưới),
    tất cả tính từ CÙNG 1 lần gọi retrieve_fn(question, k_max) mỗi câu hỏi.
    Không cần gọi LLM nên chạy được trên TOÀN BỘ golden set, không bị giới
    hạn bởi quota (khác với Generation ở bước 9).

    `retrieve_fn(question, k) -> list[dict]` được truyền vào thay vì gọi
    thẳng Chroma — nhờ đó hàm này dùng chung được cho cả dense-only (bước 6)
    lẫn hybrid/rerank (bước 10), chỉ cần đổi hàm truyền vào.

    - Recall@k (chặt, ở mức chunk): chunk phải đúng source VÀ chứa khớp
      context_snippet — đã dùng từ bước 6.
    - Precision@k, MRR (ở mức tài liệu): 1 chunk được coi là "liên quan"
      nếu đến từ đúng source, không cần khớp đúng snippet — theo đúng cách
      tài liệu tham khảo minh hoạ ví dụ Precision@5 (đếm "tài liệu đúng"
      trong top-k, không xét từng chunk có đúng câu trả lời không).
    - nDCG: bỏ qua vì cần biết TỔNG số chunk liên quan có trong toàn bộ
      Chroma cho mỗi câu hỏi để tính IDCG chuẩn hoá — golden set hiện chỉ
      có 1 (nguồn, trang) đúng cho mỗi câu, không đủ thông tin xếp hạng
      mức độ liên quan (graded relevance) mà nDCG cần, nên số liệu sẽ
      không phản ánh đúng ý nghĩa của chỉ số này nếu tính ẩu.
    """
    k_max = max(k_values)
    recall = {k: 0 for k in k_values}
    precision_sum = {k: 0.0 for k in k_values}
    mrr_sum = 0.0
    missed_at_max_k = []

    for g in golden:
        source_name = Path(g["source"]).name
        top_results = retrieve_fn(g["question"], k_max)

        is_relevant_source = [r["metadata"]["source"] == source_name for r in top_results]
        is_snippet_match = [
            r["metadata"]["source"] == source_name and snippet_found_in_chunk(g["context_snippet"], r["text"])
            for r in top_results
        ]

        for k in k_values:
            if any(is_snippet_match[:k]):
                recall[k] += 1
            precision_sum[k] += sum(is_relevant_source[:k]) / k

        first_relevant_rank = next((i + 1 for i, rel in enumerate(is_relevant_source) if rel), None)
        if first_relevant_rank:
            mrr_sum += 1.0 / first_relevant_rank

        if not any(is_snippet_match):
            missed_at_max_k.append(g["id"])

    n = len(golden)
    return {
        "recall": {k: recall[k] / n for k in k_values},
        "precision": {k: precision_sum[k] / n for k in k_values},
        "mrr": mrr_sum / n,
        "counts": {k: recall[k] for k in k_values},
        "total": n,
        "missed_at_max_k": missed_at_max_k,
    }


def main():
    golden = json.load(open(GOLDEN_PATH, encoding="utf-8"))
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    model = get_embed_model()

    print(f"Danh gia Recall@k tren {len(golden)} cau hoi, {collection.count()} chunk trong Chroma\n")

    result = evaluate_recall_at_k(lambda q, k: retrieve(collection, model, q, k), golden, K_VALUES)
    for k in K_VALUES:
        pct = result["recall"][k] * 100
        print(f"Recall@{k}: {result['counts'][k]}/{result['total']} ({pct:.1f}%)")
    print()
    for k in K_VALUES:
        print(f"Precision@{k}: {result['precision'][k]:.3f}")
    print(f"\nMRR: {result['mrr']:.3f}")

    if result["missed_at_max_k"]:
        print(f"\nCac cau khong tim thay du lieu ngay ca o k={max(K_VALUES)}: {result['missed_at_max_k']}")


if __name__ == "__main__":
    main()
