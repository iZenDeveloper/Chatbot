"""Bước 11 — Đóng gói thành demo cá nhân bằng Streamlit.

Ghép toàn bộ pipeline đã xây dựng qua các bước trước thành 1 giao diện chat
đơn giản:
- Retrieval: Hybrid Search (BM25+Dense, RRF) + Reranker cross-encoder local
  (bước 10) — đã chứng minh tốt hơn dense-only (+13 điểm % Recall@3).
- Augmentation: build_prompt() (bước 7).
- Generation: Gemini, temperature=0.2 (bước 8).
- Answer Verification: LLM kiểm tra lại Faithfulness trước khi hiển thị,
  cảnh báo rõ nếu không chắc chắn thay vì che giấu (bước 10).

Chạy: streamlit run app.py
"""
import sys
from pathlib import Path

import chromadb
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from generate import get_llm  # noqa: E402
from rerank import hybrid_retrieve_and_rerank  # noqa: E402
from retrieve import get_embed_model  # noqa: E402
from verify import answer_with_verification  # noqa: E402

CHROMA_DIR = Path(__file__).parent / "data" / "chroma_db"
COLLECTION_NAME = "rag_hoc_tap"
TOP_K = 5

st.set_page_config(page_title="RAG Học Tập — Demo", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Đang tải model & kết nối Chroma (chỉ chạy 1 lần)...")
def load_resources():
    """Cache theo session Streamlit — model embedding (bge-m3, ~2GB) và
    Chroma client chỉ load 1 lần, không load lại mỗi khi người dùng hỏi
    câu mới (Streamlit chạy lại toàn bộ script mỗi lần tương tác)."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    embed_model = get_embed_model()
    llm = get_llm()
    return collection, embed_model, llm


def retrieve_fn(collection, embed_model, question, k):
    """Wrapper để khớp chữ ký retrieve_fn mà answer_question()/
    answer_with_verification() mong đợi — bỏ qua tham số embed_model
    riêng vì hybrid_retrieve_and_rerank tự lấy model bên trong."""
    return hybrid_retrieve_and_rerank(collection, embed_model, question, k)


st.title("📚 RAG Học Tập — Demo Cá Nhân")
st.caption(
    "Pipeline: Hybrid Search (BM25+Dense) → Reranker → Prompt chống hallucination → "
    "Gemini → Answer Verification. Chỉ trả lời dựa trên tài liệu đã nạp, không bịa."
)

collection, embed_model, llm = load_resources()

with st.sidebar:
    st.header("Thông tin corpus")
    st.metric("Số chunk trong Chroma", collection.count())
    st.caption(
        "Xem chi tiết pipeline, benchmark từng bước (Recall@k, Precision@k, MRR...) "
        "trong `GHI-CHU.md` ở thư mục dự án."
    )

question = st.text_input("Đặt câu hỏi về nội dung tài liệu:", placeholder="Vd: Grammarly ra mắt vào năm nào?")

if st.button("Hỏi", type="primary") and question:
    with st.spinner("Đang tìm kiếm & sinh câu trả lời..."):
        try:
            result = answer_with_verification(collection, embed_model, llm, question, k=TOP_K, retrieve_fn=retrieve_fn)
        except Exception as e:
            st.error(f"Lỗi khi gọi LLM (có thể hết quota free tier hôm nay): {e}")
            st.stop()

    if result["verified"]:
        st.success("✅ Câu trả lời đã được xác nhận bám sát tài liệu (Answer Verification PASS).")
    else:
        st.warning(result["warning"])

    st.markdown("### Trả lời")
    st.markdown(result["answer"])

    with st.expander(f"📄 {len(result['chunks'])} đoạn tài liệu đã dùng làm ngữ cảnh (để tự kiểm tra)"):
        for i, c in enumerate(result["chunks"], start=1):
            meta = c["metadata"]
            st.markdown(f"**[{i}] {meta['source']}** — trang {meta['page']}" + (f" — *{meta['heading']}*" if meta.get("heading") else ""))
            st.text(c["text"][:500] + ("..." if len(c["text"]) > 500 else ""))
            st.divider()
