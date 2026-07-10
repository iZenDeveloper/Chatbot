"""Chunk data/processed/documents.jsonl thành các chunk sẵn sàng embedding,
ghi ra data/processed/chunks.jsonl.

Chiến lược (đã thống nhất với người dùng qua ví dụ cụ thể, xem GHI-CHU.md):
1. Section-based: gộp các bản ghi liên tiếp cùng `heading` thành 1 "section".
2. Document-aware: bảng/code (`label` "table"/"code") luôn tách thành chunk
   riêng, không bao giờ bị gộp chung với text xung quanh hay bị cắt ngang.
3. Nếu section ngắn (≤ SIZE_THRESHOLD ký tự) → giữ nguyên làm 1 chunk.
4. Nếu section quá dài (vd 1 chương sách không có heading con) → semantic
   chunking: tách câu, embed từng câu bằng model local (BAAI/bge-m3), cắt
   tại điểm độ tương đồng giữa 2 câu liên tiếp giảm mạnh (chuyển chủ đề),
   thay vì cắt cứng theo số ký tự cố định.

CACHE THEO TÀI LIỆU: chunking (đặc biệt bước semantic-split cần embed từng
câu) tốn thời gian, và việc chunk 1 tài liệu hoàn toàn độc lập với các tài
liệu khác (không có state dùng chung giữa các source). Vì vậy kết quả chunk
của mỗi tài liệu được cache theo MD5 (đọc từ extract_manifest.json mà
extract_text.py đã ghi) vào data/processed/cache_chunks/<md5>.jsonl — tài
liệu nào MD5 không đổi thì tái sử dụng ngay, không chunk lại.
"""
import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

IN_PATH = Path(__file__).parent.parent / "data" / "processed" / "documents.jsonl"
OUT_PATH = Path(__file__).parent.parent / "data" / "processed" / "chunks.jsonl"
CACHE_DIR = Path(__file__).parent.parent / "data" / "processed" / "cache_chunks"
EXTRACT_MANIFEST_PATH = Path(__file__).parent.parent / "data" / "processed" / "extract_manifest.json"

SIZE_THRESHOLD = 1800        # ky tu: section duoi nguong nay giu nguyen 1 chunk (~p90 do dai section thuc te)
SEMANTIC_MAX_CHUNK = 2000    # ky tu: hard cap cho 1 sub-chunk semantic, phong khi khong co diem chuyen chu de ro
BREAKPOINT_PERCENTILE = 25   # % thap nhat cua do tuong dong giua cau lien tiep duoc coi la diem chuyen chu de

_EMBED_MODEL = None


def get_embed_model() -> SentenceTransformer:
    """Lazy-load model embedding local BAAI/bge-m3 (chỉ 1 lần dùng chung
    cho cả script, vì load model tốn vài giây và chiếm RAM đáng kể). Nhờ
    cache theo tài liệu, nếu không có tài liệu nào mới cần semantic-split
    thì model này thậm chí không bao giờ được load.

    Ép chạy trên CPU (device="cpu"): máy này có GPU dung lượng nhỏ
    (3.68GB), từng bị CUDA OutOfMemoryError khi encode batch câu cho các
    section dài (cùng lỗi đã gặp và sửa ở embed_and_store.py bước 4)."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer("BAAI/bge-m3", device="cpu")
    return _EMBED_MODEL


def split_sentences(text: str) -> list[str]:
    """Tách text thành câu bằng regex đơn giản (dấu câu + khoảng trắng) —
    hoạt động được với cả tiếng Việt lẫn tiếng Anh vì không phụ thuộc từ
    điển ngôn ngữ cụ thể nào."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def semantic_split(sentence_page_pairs: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Cắt 1 danh sách câu (kèm số trang gốc) thành các sub-chunk theo điểm
    chuyển chủ đề, dựa trên cosine similarity giữa embedding của 2 câu liên
    tiếp — thay vì cắt cứng theo số ký tự cố định.

    Trả về list (text, page) cho từng sub-chunk; page lấy từ câu đầu tiên
    của sub-chunk đó.
    """
    if len(sentence_page_pairs) <= 1:
        return [("\n".join(s for s, _ in sentence_page_pairs), sentence_page_pairs[0][1])] if sentence_page_pairs else []

    sentences = [s for s, _ in sentence_page_pairs]
    model = get_embed_model()
    embeddings = model.encode(sentences, normalize_embeddings=True)

    sims = [float(np.dot(embeddings[i], embeddings[i + 1])) for i in range(len(embeddings) - 1)]
    threshold = np.percentile(sims, BREAKPOINT_PERCENTILE)

    groups = []
    current = [sentence_page_pairs[0]]
    current_len = len(sentence_page_pairs[0][0])
    for i, sim in enumerate(sims):
        next_pair = sentence_page_pairs[i + 1]
        should_break = sim <= threshold or current_len >= SEMANTIC_MAX_CHUNK
        if should_break:
            groups.append(current)
            current = [next_pair]
            current_len = len(next_pair[0])
        else:
            current.append(next_pair)
            current_len += len(next_pair[0])
    groups.append(current)

    return [("\n".join(s for s, _ in group), group[0][1]) for group in groups]


def flush_section(buffer: list[dict], source: str, doc_type: str, heading: str | None) -> list[dict]:
    """Chuyển 1 section (list các bản ghi cùng heading đã gom) thành 1 hoặc
    nhiều chunk: giữ nguyên nếu ngắn, semantic-split nếu quá dài."""
    if not buffer:
        return []

    joined_len = sum(len(r["text"]) for r in buffer)
    first_page = buffer[0]["metadata"]["page"]

    if joined_len <= SIZE_THRESHOLD:
        text = "\n".join(r["text"] for r in buffer)
        return [{
            "text": text,
            "metadata": {"source": source, "page": first_page, "type": doc_type, "heading": heading, "label": "text"},
        }]

    sentence_page_pairs = []
    for r in buffer:
        for s in split_sentences(r["text"]):
            sentence_page_pairs.append((s, r["metadata"]["page"]))

    sub_chunks = semantic_split(sentence_page_pairs)
    return [
        {
            "text": text,
            "metadata": {"source": source, "page": page, "type": doc_type, "heading": heading, "label": "text"},
        }
        for text, page in sub_chunks
    ]


def chunk_document(records: list[dict]) -> list[dict]:
    """Chunk toàn bộ bản ghi CỦA 1 TÀI LIỆU DUY NHẤT (đã lọc theo source từ
    trước). Tách riêng thành hàm này để có thể cache kết quả theo từng tài
    liệu độc lập ở main().

    Bảng/code luôn cắt đứt section đang gộp (flush ngay), để đảm bảo chúng
    không bao giờ bị trộn lẫn với đoạn text xung quanh trong cùng 1 chunk.
    """
    out_chunks = []
    buffer = []
    current_heading = None  # None nghia la "chua bat dau section nao"
    started = False

    def flush():
        if buffer:
            source = buffer[0]["metadata"]["source"]
            doc_type = buffer[0]["metadata"]["type"]
            out_chunks.extend(flush_section(buffer, source, doc_type, current_heading))
        buffer.clear()

    for r in records:
        label = r["metadata"]["label"]

        if label in ("table", "code"):
            flush()
            started = False
            out_chunks.append(r)  # bang/code giu nguyen, khong gop khong cat
            continue

        heading = r["metadata"]["heading"]
        if not started or heading != current_heading:
            flush()
            current_heading = heading
            started = True
        buffer.append(r)

    flush()
    return out_chunks


def load_extract_manifest() -> dict:
    """Đọc extract_manifest.json (md5 -> {source, type}) mà extract_text.py
    đã ghi, rồi đảo ngược thành source -> md5 để tra cứu nhanh theo tên file
    hiện tại. Nếu chưa chạy extract_text.py thì trả về rỗng (chunk_document
    vẫn chạy đúng, chỉ là không cache được)."""
    if not EXTRACT_MANIFEST_PATH.exists():
        return {}
    manifest = json.loads(EXTRACT_MANIFEST_PATH.read_text(encoding="utf-8"))
    return {info["source"]: md5 for md5, info in manifest.items()}


def main():
    """Đọc documents.jsonl, nhóm bản ghi theo source, rồi với mỗi tài liệu:
    - Nếu có cache (cache_chunks/<md5>.jsonl, tra theo extract_manifest.json)
      → tái sử dụng, bỏ qua hoàn toàn bước semantic-split tốn kém.
    - Nếu chưa có → chunk_document() rồi lưu cache cho lần sau.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in open(IN_PATH, encoding="utf-8")]
    source_to_hash = load_extract_manifest()

    # Gom ban ghi theo source, giu nguyen thu tu xuat hien trong file
    records_by_source: dict[str, list[dict]] = {}
    for r in records:
        records_by_source.setdefault(r["metadata"]["source"], []).append(r)

    all_chunks = []
    cache_hits = 0
    cache_misses = 0

    for source, source_records in records_by_source.items():
        file_hash = source_to_hash.get(source)
        cache_path = CACHE_DIR / f"{file_hash}.jsonl" if file_hash else None

        if cache_path and cache_path.exists():
            chunks = [json.loads(line) for line in open(cache_path, encoding="utf-8")]
            cache_hits += 1
        else:
            chunks = chunk_document(source_records)
            cache_misses += 1
            if cache_path:
                with open(cache_path, "w", encoding="utf-8") as f:
                    for c in chunks:
                        f.write(json.dumps(c, ensure_ascii=False) + "\n")

        # Patch lai source/type theo ten file HIEN TAI, phong khi file da bi
        # doi ten sau khi cache duoc tao voi ten cu (bug tung gay stale data
        # trong Chroma - chunk cu ton tai song song voi chunk moi cung noi
        # dung, chi khac ten). extract_text.py da lam dieu nay tu truoc,
        # chunk_text.py thieu buoc nay nen bi bo sot.
        doc_type = source_records[0]["metadata"]["type"]
        for c in chunks:
            c["metadata"]["source"] = source
            c["metadata"]["type"] = doc_type

        all_chunks.extend(chunks)

    # Don cache cua tai lieu khong con trong documents.jsonl nua
    current_hashes = {source_to_hash[s] for s in records_by_source if s in source_to_hash}
    for cache_file in CACHE_DIR.glob("*.jsonl"):
        if cache_file.stem not in current_hashes:
            cache_file.unlink()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Hoan tat: {cache_hits} tai lieu tai su dung cache, {cache_misses} tai lieu chunk moi, "
          f"{len(records)} ban ghi -> {len(all_chunks)} chunk -> {OUT_PATH}")


if __name__ == "__main__":
    main()
