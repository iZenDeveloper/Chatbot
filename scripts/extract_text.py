"""Trích xuất text từ data/raw/** ra data/processed/documents.jsonl.

Dùng Docling (https://github.com/DS4SD/docling) làm engine trích xuất chính
cho mọi định dạng (PDF, DOCX, PPTX, HTML, MD, CSV, ODT...) thay vì tự viết
1 hàm riêng cho từng đuôi file như bản trước.

Lý do đổi từ PyMuPDF/python-docx/odfpy/trafilatura tự viết sang Docling:
đã benchmark trên chính dữ liệu của dự án và thấy PyMuPDF/pdfplumber không
tái tạo được bảng không viền (borderless table) trong các paper khoa học —
trả về 0 bảng phát hiện được. Docling tái tạo đúng cấu trúc bảng thành
markdown, đồng thời tự nhận diện heading, giữ nguyên cấu trúc tài liệu —
điều mà cách trích xuất text phẳng cũ không làm được. Đánh đổi: chậm hơn
đáng kể (xem PdfPipelineOptions bên dưới).

CACHE THEO MD5: Docling chạy khá chậm (~0.15-1s/trang), nên nếu chạy lại
trên toàn bộ corpus mỗi khi chỉ thêm 1-2 tài liệu mới sẽ rất lãng phí thời
gian/tài nguyên. Kết quả trích xuất mỗi file được lưu vào
data/processed/cache_extract/<md5>.jsonl — lần chạy sau, file nào MD5 không
đổi (kể cả khi bị đổi tên) sẽ được tái sử dụng ngay, không chạy lại Docling.
Chỉ file thật sự mới (MD5 chưa từng thấy) mới tốn thời gian xử lý.
"""
import json
import sys
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from _common import file_md5

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_PATH = Path(__file__).parent.parent / "data" / "processed" / "documents.jsonl"
CACHE_DIR = Path(__file__).parent.parent / "data" / "processed" / "cache_extract"
MANIFEST_PATH = Path(__file__).parent.parent / "data" / "processed" / "extract_manifest.json"

# Tat OCR: da benchmark tren sach 301 trang, tat OCR giam thoi gian tu 272s
# xuong con 45s (nhanh gap 6 lan) ma chat luong bang/heading khong doi, vi
# toan bo tai lieu hien tai deu la PDF "digital-born" da co san text layer,
# khong phai anh scan can OCR moi doc duoc chu.
_pdf_opts = PdfPipelineOptions()
_pdf_opts.do_ocr = False
CONVERTER = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=_pdf_opts)})


def extract_with_docling(path: Path, doc_type: str) -> list[dict]:
    """Convert 1 file bằng Docling, trả về 1 bản ghi cho mỗi phần tử nội dung
    (đoạn văn, heading, bảng, caption...) mà Docling nhận diện được.

    - Bảng: dùng item.export_to_markdown(doc) để lấy đúng cấu trúc hàng/cột
      (kèm caption nếu có), thay vì text phẳng lộn xộn.
    - Ảnh (picture): bỏ qua, vì bản thân ảnh không có text hữu ích — caption
      của ảnh (nếu có) đã là 1 item "caption" riêng, vẫn được giữ lại.
    - heading gần nhất: theo dõi qua biến current_heading, gắn vào metadata
      của mọi đoạn văn/bảng đứng sau nó cho tới khi gặp heading tiếp theo.
    - page: lấy từ item.prov[0].page_no nếu có (PDF); với định dạng không
      phân trang (DOCX, HTML...) Docling không trả về prov, dùng page=1.
    """
    result = CONVERTER.convert(str(path))
    doc = result.document

    records = []
    current_heading = None
    for item, _level in doc.iterate_items():
        label = str(getattr(item, "label", "")).split(".")[-1].lower()

        if label == "picture":
            continue  # anh khong co text huu ich, caption da tach rieng

        if label == "table":
            text = item.export_to_markdown(doc)
        else:
            text = (getattr(item, "text", "") or "").strip()

        if not text:
            continue

        if label == "section_header":
            current_heading = text

        page_no = 1
        prov = getattr(item, "prov", None)
        if prov:
            page_no = prov[0].page_no

        records.append({
            "text": text,
            "metadata": {
                "source": path.name,
                "page": page_no,
                "type": doc_type,
                "label": label,
                "heading": current_heading,
            },
        })
    return records


# Docling tu nhan dien dinh dang qua noi dung/duoi file, nen chi can 1 ham
# duy nhat dung chung cho tat ca dinh dang no ho tro. Danh sach day du:
# pdf, docx, pptx, html, image, md, csv, xlsx, odt, ods, odp, email, epub...
SUPPORTED_SUFFIXES = {
    ".pdf", ".docx", ".pptx", ".html", ".htm", ".md", ".csv",
    ".xlsx", ".odt", ".ods", ".odp",
}


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: dict):
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    """Quét toàn bộ data/raw/{personal,public}/*, với mỗi file:
    - Nếu MD5 đã có trong cache_extract/ (đã xử lý ở lần chạy trước, kể cả
      khi bị đổi tên sau đó) → đọc lại cache, chỉ patch lại metadata
      source/type cho khớp tên/thư mục hiện tại (rẻ, không cần Docling).
    - Nếu MD5 chưa từng thấy → chạy Docling (đắt), lưu kết quả vào cache.
    Sau đó ghép tất cả bản ghi (theo đúng thứ tự file hiện có) ra
    documents.jsonl. Cache/manifest của file đã bị xoá khỏi data/raw cũng
    được dọn theo, tránh phình to vô hạn.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    all_records = []
    skipped = []
    cache_hits = 0
    cache_misses = 0
    seen_hashes = set()

    for type_dir in sorted(RAW_DIR.iterdir()):
        if not type_dir.is_dir():
            continue
        doc_type = type_dir.name  # "personal" hoac "public", lay tu ten thu muc cha
        for path in sorted(type_dir.glob("*")):
            if not path.is_file():
                # Bo qua thu muc con, vd thu muc "..._files" ma trinh duyet
                # tu tao khi luu trang web dang "Webpage, Complete".
                continue
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                print(f"[bo qua] chua ho tro dinh dang: {path.name}", file=sys.stderr)
                skipped.append(path.name)
                continue

            file_hash = file_md5(path)
            seen_hashes.add(file_hash)
            cache_path = CACHE_DIR / f"{file_hash}.jsonl"

            if cache_path.exists():
                records = [json.loads(line) for line in open(cache_path, encoding="utf-8")]
                cache_hits += 1
            else:
                print(f"Dang xu ly (moi): {path.name}")
                try:
                    records = extract_with_docling(path, doc_type)
                except Exception as e:
                    print(f"  [LOI] khong doc duoc {path.name}: {e}", file=sys.stderr)
                    skipped.append(path.name)
                    continue
                with open(cache_path, "w", encoding="utf-8") as f:
                    for r in records:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                cache_misses += 1

            # Patch lai source/type theo ten file/thu muc HIEN TAI, phong khi
            # file da bi doi ten (vd qua rename_new_docs.py) sau khi cache
            # duoc tao voi ten cu.
            for r in records:
                r["metadata"]["source"] = path.name
                r["metadata"]["type"] = doc_type
            all_records.extend(records)
            manifest[file_hash] = {"source": path.name, "type": doc_type}

    # Don cache/manifest cua file da bi xoa khoi data/raw, tranh phinh to vo han
    stale_hashes = set(manifest.keys()) - seen_hashes
    for h in stale_hashes:
        del manifest[h]
        (CACHE_DIR / f"{h}.jsonl").unlink(missing_ok=True)

    save_manifest(manifest)

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for r in all_records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nHoan tat: {cache_hits} file tai su dung cache, {cache_misses} file xu ly moi, "
          f"{len(all_records)} doan -> {OUT_PATH}")
    if skipped:
        print(f"Da bo qua {len(skipped)} file: {skipped}")
    if stale_hashes:
        print(f"Da don {len(stale_hashes)} cache cu (file khong con trong data/raw)")


if __name__ == "__main__":
    main()
