"""Sinh embedding cho data/processed/chunks.jsonl và lưu vào Chroma local
(data/chroma_db/), sẵn sàng cho bước Retrieval (bước 6).

Quyết định kỹ thuật (đã giải thích và thống nhất với người dùng qua ví dụ cụ
thể — xem GHI-CHU.md):
- Model: BAAI/bge-m3 (8192 token, đa ngôn ngữ, đã dùng ở bước chunking) —
  0/4415 chunk bị cắt cụt, trong khi các model giới hạn 512 token (e5,
  mpnet...) sẽ cắt mất 95 chunk (chủ yếu là bảng dữ liệu dài).
- Khoảng cách: cosine, chỉ định rõ qua configuration={"hnsw": {"space":
  "cosine"}} khi tạo collection — KHÔNG dựa vào mặc định của Chroma (đã
  kiểm chứng: Chroma mặc định dùng "l2", không phải cosine).
- Upsert theo ID ổn định (hash source+page+heading+text): chunk đã có trong
  Chroma thì bỏ qua, không embed lại — tránh lặp lại vấn đề "mỗi lần chạy
  lại tốn thời gian/tài nguyên" đã gặp ở bước extract/chunk. Chroma tự đóng
  vai trò cache bền vững cho embedding.
- Luôn dùng ĐÚNG model này (bge-m3) khi embed câu hỏi ở bước Retrieval sau
  này — bắt buộc phải nhất quán giữa index và query.
"""
import hashlib
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path(__file__).parent.parent / "data" / "processed" / "chunks.jsonl"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "rag_hoc_tap"
EMBED_MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 100

_EMBED_MODEL = None


def get_embed_model() -> SentenceTransformer:
    """Lazy-load model embedding — chỉ load khi thực sự có chunk mới cần
    embed (tránh tải model nếu chạy lại mà không có gì thay đổi).

    Ép chạy trên CPU (device="cpu") thay vì để tự động chọn GPU: máy này có
    GPU dung lượng nhỏ (3.68GB), từng bị CUDA OutOfMemoryError khi encode
    batch chứa chunk bảng rất lớn (tới ~24.000 ký tự). Đây là batch job
    không cần tốc độ real-time nên đánh đổi lấy sự ổn định là hợp lý."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    return _EMBED_MODEL


def chunk_id(chunk: dict) -> str:
    """Sinh ID ổn định cho 1 chunk từ nội dung + vị trí của nó (source, page,
    heading, text). Cùng 1 chunk sẽ luôn ra cùng 1 ID giữa các lần chạy,
    nên dùng được cho upsert — chunk không đổi thì ID không đổi, Chroma
    nhận ra đã có sẵn và bỏ qua, không embed lại."""
    m = chunk["metadata"]
    raw = f"{m['source']}|{m['page']}|{m.get('heading')}|{chunk['text']}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def clean_metadata(metadata: dict) -> dict:
    """Chroma không chấp nhận giá trị None trong metadata — thay bằng
    chuỗi rỗng. Ép kiểu page về int để nhất quán (Docling đôi khi trả về
    kiểu khác cho các định dạng không phân trang)."""
    return {
        "source": metadata["source"],
        "page": int(metadata["page"]),
        "type": metadata["type"],
        "label": metadata.get("label", "text"),
        "heading": metadata.get("heading") or "",
    }


def get_or_create_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
    )


def main():
    """Đọc chunks.jsonl, tính ID ổn định cho từng chunk, rồi:
    - Bỏ qua chunk đã có sẵn trong Chroma (theo ID) — không embed lại.
    - Embed + thêm mới các chunk chưa có, theo từng batch để không load
      hết vào RAM cùng lúc khi corpus lớn dần.
    - Xoá khỏi Chroma các chunk không còn tồn tại trong chunks.jsonl nữa
      (vd tài liệu đã bị xoá khỏi data/raw), để Chroma luôn khớp corpus
      hiện tại.
    """
    chunks = [json.loads(line) for line in open(CHUNKS_PATH, encoding="utf-8")]
    collection = get_or_create_collection()

    # Mot so chunk (vd manh vun 1 ky tu nhu so thu tu chu thich chan trang)
    # co the trung het source+page+heading+text, khien chunk_id() ra cung
    # 1 gia tri. Them so thu tu lan gap lai (-2, -3...) de dam bao ID duy
    # nhat, van on dinh giua cac lan chay vi thu tu chunk trong file la
    # deterministic.
    seen_count: dict[str, int] = {}
    ids = []
    for c in chunks:
        base_id = chunk_id(c)
        n = seen_count.get(base_id, 0)
        seen_count[base_id] = n + 1
        ids.append(base_id if n == 0 else f"{base_id}-{n}")

    existing_ids = set(collection.get(ids=ids)["ids"]) if chunks else set()

    new_indices = [i for i, cid in enumerate(ids) if cid not in existing_ids]

    if new_indices:
        model = get_embed_model()
        for start in range(0, len(new_indices), BATCH_SIZE):
            batch_idx = new_indices[start:start + BATCH_SIZE]
            texts = [chunks[i]["text"] for i in batch_idx]
            embeddings = model.encode(texts, normalize_embeddings=True).tolist()
            collection.add(
                ids=[ids[i] for i in batch_idx],
                embeddings=embeddings,
                documents=texts,
                metadatas=[clean_metadata(chunks[i]["metadata"]) for i in batch_idx],
            )
            print(f"  Da embed va them {min(start + BATCH_SIZE, len(new_indices))}/{len(new_indices)} chunk moi")

    # Don khoi Chroma cac chunk khong con trong chunks.jsonl (vd tai lieu da bi xoa)
    all_db_ids = set(collection.get()["ids"])
    stale_ids = all_db_ids - set(ids)
    if stale_ids:
        collection.delete(ids=list(stale_ids))

    print(f"\nHoan tat: {len(new_indices)} chunk moi duoc embed, "
          f"{len(chunks) - len(new_indices)} chunk tai su dung (da co san trong Chroma), "
          f"{len(stale_ids)} chunk cu da bi xoa. Tong so chunk trong Chroma: {collection.count()}")


if __name__ == "__main__":
    main()
