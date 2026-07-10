# RAG Học Tập Song Ngữ Việt–Anh

Hệ thống RAG (Retrieval-Augmented Generation) cá nhân, học tập, chạy hoàn toàn miễn phí (local + free-tier API). Trả lời câu hỏi dựa trên một kho tài liệu song ngữ Việt–Anh về AI trong giáo dục và kỹ thuật RAG, có trích dẫn nguồn và cơ chế chống hallucination.

**Báo cáo kỹ thuật đầy đủ:** xem [GHI-CHU.md](GHI-CHU.md) — kiến trúc, phương pháp, số liệu đánh giá (Retrieval + Generation), hạn chế và hướng cải thiện, kèm giải thích khái niệm/thuật ngữ.

**Cơ chế kỹ thuật chi tiết từng công nghệ:** xem [CONG-NGHE-CHI-TIET.md](CONG-NGHE-CHI-TIET.md).

## Kiến trúc

Trích xuất (Docling) → Chunking (section-based + document-aware + semantic) → Embedding (`BAAI/bge-m3`) → Vector DB (Chroma, HNSW) → Retrieval (Hybrid BM25+Dense → Reranker cross-encoder) → Prompt chống hallucination → Generation (Gemini/Groq) → Answer Verification.

## Kết quả

Đo trên golden set 65 câu hỏi, corpus 62 tài liệu / 7.044 chunk (chi tiết đầy đủ ở [GHI-CHU.md](GHI-CHU.md#4-kết-quả-cuối-cùng)):

| Retrieval (Hybrid+Rerank) | Generation (LLM-as-judge) |
|---|---|
| Recall@3 = 69.2%, Recall@10 = 78.5%, MRR = 0.867 | Correctness = 76.9%, Faithfulness = 81.5% |

## Cài đặt

```bash
python3 -m venv rag-env
source rag-env/bin/activate
pip install -r requirements.txt

cp .env.example .env
# điền GOOGLE_API_KEY (aistudio.google.com/apikey) và GROQ_API_KEY (console.groq.com) — cả hai đều miễn phí
```

## Dữ liệu

Thư mục `data/raw/` (tài liệu nguồn) và `data/processed/` (kết quả trích xuất/chunking) **không có sẵn trong repo** vì chứa toàn văn tài liệu có bản quyền của bên thứ ba. Để chạy dự án:

1. Tạo `data/raw/personal/` và `data/raw/public/`, thêm tài liệu PDF/HTML của riêng bạn vào đó.
2. Chạy pipeline xử lý:

```bash
python scripts/rename_new_docs.py    # đổi tên tự động theo tiêu đề/tác giả (cần GOOGLE_API_KEY)
python scripts/extract_text.py       # trích xuất → data/processed/documents.jsonl
python scripts/chunk_text.py         # chunking → data/processed/chunks.jsonl
python scripts/embed_and_store.py    # embedding → data/chroma_db/
```

3. (Tuỳ chọn) Soạn bộ câu hỏi đánh giá riêng theo đúng schema của `data/golden_set.json` có sẵn trong repo.

## Chạy demo

```bash
streamlit run app.py
```

## Đánh giá lại

```bash
python scripts/retrieve.py                    # Recall/Precision/MRR — dense-only
python scripts/evaluate_advanced_retrieval.py  # Hybrid+Rerank (chậm, rerank chạy CPU)
python scripts/evaluate_groq_batch.py          # Generation qua Groq (resumable — chạy lại nếu bị ngắt giữa chừng)
```

## Cấu trúc mã nguồn

Xem [Phụ lục A trong GHI-CHU.md](GHI-CHU.md#phụ-lục-a--cấu-trúc-mã-nguồn) để biết vai trò từng script.
