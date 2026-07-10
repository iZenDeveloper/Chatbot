"""Hàm dùng chung cho các script trong thư mục scripts/."""
import hashlib
from pathlib import Path


def file_md5(path: Path) -> str:
    """Tính MD5 của nội dung file (đọc theo chunk 8KB để không load cả file
    lớn vào RAM cùng lúc). Dùng làm "danh tính" ổn định của file — không đổi
    kể cả khi file bị đổi tên — để cache kết quả xử lý theo nội dung thay vì
    theo tên file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_text(response) -> str:
    """Chuẩn hoá response.content của LangChain LLM về string thuần.

    Phát hiện khi debug bước 10 (verify.py): hầu hết model Gemini trả
    content dạng string, nhưng `gemini-flash-latest` trả về dạng list các
    block (vd [{"type": "text", "text": "...", "extras": {...}}]) — làm
    crash mọi code gọi .upper()/.split() thẳng trên response.content. Hàm
    này gộp lại thành 1 chuỗi để code phía sau dùng thống nhất, không cần
    biết model nào trả về định dạng gì."""
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)
