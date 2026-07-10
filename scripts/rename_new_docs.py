"""Tự động đổi tên tài liệu mới thả vào data/raw/{personal,public}/.

Cách dùng: thả file vào đúng thư mục (personal/public) rồi chạy:
    python scripts/rename_new_docs.py

Cơ chế:
- Theo dõi file đã xử lý qua MD5 trong data/raw/.rename_manifest.json,
  nên chạy lại nhiều lần không xử lý trùng lại file cũ.
- Với file chưa có trong manifest: trích 1-2 trang đầu, nhờ LLM (Google Gemini,
  free tier, đọc key từ .env) tóm tắt thành 1 tên file dạng slug (không dấu,
  kebab-case) dựa trên tiêu đề/tác giả thật, rồi đổi tên.
- Phát hiện file trùng MD5 thì báo (dù trùng với file khác đang quét cùng lượt,
  hay trùng với file đã đổi tên từ 1 lần chạy trước đó), không tự xoá.
"""
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import fitz  # PyMuPDF
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from _common import extract_text, file_md5  # noqa: E402
from extract_text import SUPPORTED_SUFFIXES  # noqa: E402

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
MANIFEST_PATH = RAW_DIR / ".rename_manifest.json"
PREVIEW_PAGES = 2
PREVIEW_CHARS = 3000

load_dotenv(Path(__file__).parent.parent / ".env")


def load_manifest() -> dict:
    """Đọc sổ theo dõi .rename_manifest.json (md5 -> thông tin file đã đổi
    tên). Nếu chưa có file (lần chạy đầu tiên) thì coi như sổ rỗng."""
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: dict):
    """Ghi lại sổ theo dõi sau khi xử lý xong, để lần chạy sau biết file nào
    đã được đổi tên rồi mà bỏ qua."""
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def preview_text(path: Path) -> str:
    """Trích nhanh 1-2 trang/đoạn đầu của file để đặt tên, KHÔNG dùng Docling
    (engine chính trong extract_text.py) vì Docling quá chậm cho việc này —
    convert cả 1 cuốn sách 300 trang chỉ để xem 2 trang đầu là lãng phí (đã
    benchmark: Docling ~45s/sách so với PyMuPDF gần như tức thời). Dùng lại
    các thư viện nhẹ (PyMuPDF, python-docx, trafilatura...) chỉ để lấy đủ
    text cho LLM nhận ra tiêu đề/tác giả, không cần chất lượng cấu trúc cao
    như bước trích xuất chính thức."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        with fitz.open(path) as pdf:
            text = "\n".join(p.get_text() for p in list(pdf)[:PREVIEW_PAGES])
    elif suffix == ".docx":
        import docx

        d = docx.Document(path)
        text = "\n".join(p.text for p in d.paragraphs[:60] if p.text.strip())
    elif suffix in (".html", ".htm"):
        import trafilatura

        html = path.read_text(encoding="utf-8", errors="ignore")
        text = trafilatura.extract(html) or ""
    elif suffix in (".txt", ".md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".csv":
        import pandas as pd

        text = pd.read_csv(path, nrows=5).to_string()
    elif suffix == ".odt":
        from odf import teletype
        from odf.opendocument import load
        from odf.text import P

        doc = load(path)
        paragraphs = doc.getElementsByType(P)[:60]
        text = "\n".join(teletype.extractText(p) for p in paragraphs)
    else:
        raise ValueError(f"chua ho tro dinh dang {suffix}")
    return text[:PREVIEW_CHARS]


def slugify_fallback(text: str) -> str:
    """Chuyển 1 chuỗi bất kỳ thành dạng slug an toàn cho tên file: bỏ dấu
    tiếng Việt, chỉ giữ chữ/số, nối bằng dấu gạch ngang. Dùng khi LLM trả về
    câu trả lời không đúng định dạng slug mong muốn (vd còn dấu, khoảng trắng)."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text[:80] or "tai-lieu"


def suggest_slug_via_llm(preview: str, original_name: str) -> str:
    """Gọi Gemini (qua GOOGLE_API_KEY trong .env), đưa vào đoạn preview text
    và tên file gốc, yêu cầu trả về 1 tên file slug dựa trên tiêu đề/tác giả
    thật. Nếu kết quả LLM không đúng định dạng kebab-case sạch, dùng
    slugify_fallback để làm sạch lại trước khi dùng làm tên file."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "Chua co GOOGLE_API_KEY trong .env. Hay copy .env.example -> .env va dien key "
            "(lay mien phi tai https://aistudio.google.com/apikey)."
        )
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
    prompt = f"""Đây là vài trang đầu của một tài liệu PDF (tên file gốc: "{original_name}"):

---
{preview}
---

Hãy đặt lại tên file theo tiêu đề/tác giả thật của tài liệu (không dùng tên file gốc).
Yêu cầu định dạng: chữ thường, không dấu, nối bằng dấu gạch ngang (kebab-case),
không có phần mở rộng file, tối đa khoảng 8-10 từ, dạng "tieu-de-ngan-gon_tac-gia".
Chỉ trả lời đúng 1 dòng là tên file, không giải thích gì thêm."""
    response = llm.invoke(prompt)
    slug = extract_text(response).strip().strip('"').strip()
    slug = re.sub(r"\.\w+$", "", slug)  # phong khi LLM lo them duoi file
    return slugify_fallback(slug) if not re.match(r"^[a-z0-9-_]+$", slug) else slug


def unique_path(directory: Path, stem: str, suffix: str, current_path: Path) -> Path:
    """Trả về đường dẫn chưa tồn tại trong thư mục, tự thêm hậu tố -2, -3...
    nếu tên slug bị trùng với file khác đã có sẵn.

    current_path là chính file đang được đổi tên — phải loại trừ nó khỏi
    việc kiểm tra tồn tại, nếu không file sẽ luôn "trùng với chính nó" (vì
    tại thời điểm kiểm tra, file gốc vẫn còn nằm ở đó), gây bug đổi tên nhầm
    thành "-2" ngay cả khi slug mới trùng khớp tên cũ vốn đã đúng chuẩn.
    """
    candidate = directory / f"{stem}{suffix}"
    n = 2
    while candidate.exists() and candidate != current_path:
        candidate = directory / f"{stem}-{n}{suffix}"
        n += 1
    return candidate


def main():
    """Quét data/raw/{personal,public}/, với mỗi file:
    1. Tính MD5 để biết đã xử lý chưa (qua manifest) và có trùng file khác
       đang quét trong cùng lượt này không (seen_this_run).
    2. Nếu là file mới, chưa hỗ trợ định dạng thì bỏ qua; nếu hỗ trợ thì lấy
       preview text, nhờ LLM đặt tên slug, rồi đổi tên file thật trên đĩa.
    3. Ghi nhận vào manifest để lần chạy sau không xử lý lại.
    Lỗi ở từng file (vd hết quota API) chỉ bỏ qua file đó, không dừng cả script.
    """
    manifest = load_manifest()
    known_hashes = set(manifest.keys())
    seen_this_run = {}  # md5 -> path, de bat trung lap ngay trong lan quet nay

    for type_dir in sorted(RAW_DIR.iterdir()):
        if not type_dir.is_dir():
            continue
        doc_type = type_dir.name
        for path in sorted(type_dir.glob("*")):
            if path.name.startswith("."):
                continue
            if not path.is_file():
                # Bo qua thu muc con, vd thu muc "..._files" ma trinh duyet
                # tu tao khi luu trang web dang "Webpage, Complete" kem file .html.
                continue
            file_hash = file_md5(path)

            if file_hash in seen_this_run:
                print(f"[TRUNG LAP] {path.name} == {seen_this_run[file_hash].name} -> bo qua, tu kiem tra va xoa thu cong")
                continue
            seen_this_run[file_hash] = path

            if file_hash in known_hashes:
                old_name = manifest[file_hash]["final_name"]
                if old_name != path.name:
                    print(f"[TRUNG LAP VOI FILE CU] {path.name} == {old_name} (da xu ly tu truoc) -> bo qua, tu kiem tra va xoa thu cong")
                continue  # da xu ly truoc do

            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                print(f"[BO QUA] chua ho tro dinh dang: {path.name}")
                continue

            print(f"Dang xu ly tai lieu moi: {path.name}")
            try:
                preview = preview_text(path)
                slug = suggest_slug_via_llm(preview, path.name)
            except Exception as e:
                print(f"  [LOI] {e} -> giu nguyen ten, se thu lai lan sau")
                continue

            new_path = unique_path(type_dir, slug, path.suffix.lower(), current_path=path)
            if new_path != path:
                path.rename(new_path)
                print(f"  -> doi ten thanh: {new_path.name}")
            else:
                print(f"  -> ten da chuan, giu nguyen: {path.name}")
            manifest[file_hash] = {
                "final_name": new_path.name,
                "original_name": path.name,
                "type": doc_type,
            }

    save_manifest(manifest)
    print("\nHoan tat. Nho chay lai scripts/extract_text.py de cap nhat data/processed/documents.jsonl.")


if __name__ == "__main__":
    main()
