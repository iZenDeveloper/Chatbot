# Quy Trình Xây Dựng Dự Án RAG
### (Bản điều chỉnh cho dự án học tập/nghiên cứu cá nhân — chạy local, Python)

---

## Bối cảnh dự án đã chốt

| Hạng mục | Quyết định |
|---|---|
| Mục đích | Học tập & nghiên cứu cá nhân về RAG |
| Người dùng | Chính bạn |
| Nguồn dữ liệu | Hỗn hợp: tài liệu công khai có sẵn (kỹ thuật, giáo trình...) + PDF sách/bài báo khoa học của riêng bạn |
| Ngôn ngữ | Việt + Anh |
| Môi trường | Python, chạy local |
| Model | API cloud (OpenAI/Anthropic/Voyage...) |
| Vector DB | Chroma (local, nhẹ, không cần server riêng) |
| Phân quyền | Không cần |

---

## Mục lục

1. Chuẩn bị môi trường & phạm vi thử nghiệm
2. Thu thập & chuẩn bị dữ liệu
3. Chunking
4. Embedding
5. Vector Database (Chroma)
6. Retrieval
7. Augmentation & Prompt
8. Generation
9. Đánh giá (Evaluation)
10. Kỹ thuật nâng cao (mở rộng dần)
11. Đóng gói thành demo cá nhân
12. Checklist tổng hợp

---

## 1. Chuẩn bị môi trường & phạm vi thử nghiệm

### 1.1 Cài đặt môi trường Python local
```bash
python -m venv rag-env
source rag-env/bin/activate   # Windows: rag-env\Scripts\activate
pip install langchain langchain-community langchain-openai chromadb pypdf tiktoken python-dotenv
```

### 1.2 Chọn bộ dữ liệu khởi điểm (nhỏ, dễ kiểm soát)
- 5–10 PDF: bài báo khoa học/tài liệu nghiên cứu của bạn.
- 1–2 tài liệu kỹ thuật public (VD: docs của một thư viện Python) để có sự đa dạng cấu trúc (bảng, code block, references).
- **Mục tiêu:** đủ nhỏ để chạy lại toàn bộ pipeline trong vài phút, đủ đa dạng để thấy vấn đề thật (chunking sai chỗ, retrieval nhầm...).

### 1.3 Soạn "golden set" nhỏ để tự đánh giá
- Viết tay 10–15 câu hỏi bạn tự biết đáp án đúng dựa trên các tài liệu đã chọn.
- Ghi lại: câu hỏi → đoạn văn bản chứa đáp án đúng (trong tài liệu nào, đoạn nào) → câu trả lời mong đợi.
- Đây là "bộ đề kiểm tra" dùng lại mỗi khi bạn đổi chunking/embedding/retrieval để biết đang tốt lên hay tệ đi.

**Đầu ra bước này:** thư mục `data/raw/` chứa PDF, file `golden_set.json` chứa câu hỏi mẫu.

---

## 2. Thu thập & chuẩn bị dữ liệu

### 2.1 Đọc & trích xuất text từ PDF
```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/raw/paper1.pdf")
pages = loader.load()  # mỗi page là 1 Document với metadata (page number, source)
```

### 2.2 Làm sạch text
- Loại bỏ header/footer lặp lại, số trang rời rạc, ký tự lỗi do OCR.
- Với bài báo khoa học: cân nhắc loại phần References ra khỏi nội dung chính (thường gây nhiễu retrieval) hoặc tách thành nhóm riêng.
- Chuẩn hóa khoảng trắng, xuống dòng thừa.

### 2.3 Giữ metadata quan trọng
Với mỗi Document, đảm bảo lưu:
```python
metadata = {
    "source": "paper1.pdf",
    "page": 3,
    "title": "Tên bài báo",
    "type": "research_paper"  # hoặc "tech_doc"
}
```
Metadata này sẽ giúp bạn **trích dẫn nguồn** khi trả lời sau này, và debug khi retrieval sai.

---

## 3. Chunking

### 3.1 Bắt đầu đơn giản: Recursive Character Splitting
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # ~500 token, phù hợp bài báo khoa học
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " "]
)
chunks = splitter.split_documents(pages)
```

### 3.2 Thử nghiệm & so sánh (mục đích học tập)
Vì đây là dự án học, hãy **thử ít nhất 2 cách chunking** để tự thấy khác biệt:
- Fixed-size (baseline).
- Structure-aware: tách theo heading nếu tài liệu có (đặc biệt hợp với tech docs dạng Markdown/HTML).

So sánh kết quả retrieval trên golden set giữa 2 cách để rút kinh nghiệm thực tế — đây là bài học quan trọng nhất của RAG.

---

## 4. Embedding

### 4.1 Sinh embedding
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```
> Nếu câu hỏi/tài liệu tiếng Việt nhiều, nên thử thêm **BGE-M3** hoặc **Voyage AI multilingual** để so sánh chất lượng — đây cũng là một thử nghiệm học tập tốt.

### 4.2 Lưu ý chi phí khi học
- `text-embedding-3-small` rất rẻ, phù hợp thử nghiệm nhiều lần.
- Cache lại embedding đã tính (tránh gọi API lặp lại khi chạy thử lại pipeline).

---

## 5. Vector Database (Chroma – local)

```python
from langchain_community.vectorstores import Chroma

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="data/chroma_db"
)
vectordb.persist()
```

- Chroma lưu trực tiếp trên đĩa (`persist_directory`) — không cần server, phù hợp máy cá nhân.
- Khi thêm tài liệu mới, chỉ cần `vectordb.add_documents(new_chunks)` thay vì rebuild toàn bộ.

---

## 6. Retrieval

### 6.1 Truy vấn cơ bản
```python
retriever = vectordb.as_retriever(search_kwargs={"k": 5})
results = retriever.invoke("Câu hỏi của bạn ở đây")
```

### 6.2 Nâng cấp dần (mỗi bước là một bài học riêng)
- **Thêm reranking:** dùng Cohere Rerank hoặc `bge-reranker` để sắp xếp lại top-10 → top-3 tốt nhất trước khi đưa vào prompt.
- **Hybrid search:** kết hợp BM25 (`rank_bm25` hoặc `langchain BM25Retriever`) với vector search, hợp nhất bằng Reciprocal Rank Fusion.
- **So sánh k khác nhau** (k=3 vs k=5 vs k=10) trên golden set để thấy ảnh hưởng đến chất lượng câu trả lời.

---

## 7. Augmentation & Prompt

```python
prompt_template = """Bạn là trợ lý trả lời dựa trên tài liệu được cung cấp.
Chỉ sử dụng thông tin trong phần NGỮ CẢNH bên dưới.
Nếu không đủ thông tin để trả lời, hãy nói rõ là không tìm thấy.
Trích dẫn nguồn (tên tài liệu, trang) cho mỗi ý quan trọng.

NGỮ CẢNH:
{context}

CÂU HỎI: {question}

TRẢ LỜI:"""
```

- Ghép các chunk truy xuất được vào `{context}`, kèm theo `[nguồn: {source}, trang {page}]` trước mỗi chunk để LLM có thể trích dẫn.

---

## 8. Generation

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.2)
response = llm.invoke(prompt_template.format(context=context_text, question=query))
```

- Temperature thấp (0–0.3) để câu trả lời bám sát tài liệu, hạn chế "bịa".
- In ra cả câu trả lời lẫn các chunk đã dùng để bạn tự kiểm tra độ chính xác (rất hữu ích khi học).

---

## 9. Đánh giá (Evaluation)

### 9.1 Đánh giá thủ công bằng golden set
Với mỗi câu hỏi trong `golden_set.json`:
1. Chạy qua pipeline, ghi lại câu trả lời + chunk đã truy xuất.
2. Tự chấm: retrieval có lấy đúng đoạn chứa đáp án không? Câu trả lời có đúng và không bịa không?
3. Ghi vào bảng theo dõi (Excel/CSV) để so sánh qua các lần thay đổi pipeline.

### 9.2 Dùng framework (khi đã quen tay)
- Thử **RAGAS** để tự động tính Faithfulness, Answer Relevancy, Context Precision/Recall — giúp bạn học cách đo lường RAG một cách bài bản, không chỉ cảm tính.

```bash
pip install ragas
```

---

## 10. Kỹ thuật nâng cao (mở rộng dần sau khi pipeline cơ bản chạy ổn)

Gợi ý thứ tự học, từ dễ đến khó:
1. Reranking (Cohere Rerank / bge-reranker).
2. Hybrid search (BM25 + vector).
3. Query rewriting (LLM viết lại câu hỏi mơ hồ trước khi retrieval).
4. HyDE (sinh câu trả lời giả định để embed tìm kiếm).
5. Parent-child chunking (chunk nhỏ để tìm, trả ngữ cảnh lớn hơn).
6. Agentic RAG / multi-hop reasoning cho câu hỏi phức tạp cần nhiều bước suy luận.

Mỗi kỹ thuật nên được thử nghiệm **độc lập** trên golden set để thấy rõ nó cải thiện gì.

---

## 11. Đóng gói thành demo cá nhân (tuỳ chọn)

Khi pipeline đã ổn định, có thể bọc thành giao diện đơn giản để dùng thực tế:
- **Streamlit** hoặc **Gradio**: nhanh nhất để có UI chat local.
```bash
pip install streamlit
```
- Không cần Docker/CI-CD ở giai đoạn học — chỉ cần chạy `streamlit run app.py` là đủ.

---

## 12. Checklist tổng hợp

- [ ] Cài môi trường Python, các thư viện cần thiết
- [ ] Chọn 5-10 PDF cá nhân + 1-2 tài liệu public làm dữ liệu thử nghiệm
- [ ] Soạn golden set 10-15 câu hỏi + đáp án đúng
- [ ] Trích xuất & làm sạch text từ PDF, giữ metadata
- [ ] Thử nghiệm ít nhất 2 cách chunking, so sánh kết quả
- [ ] Sinh embedding, lưu vào Chroma local
- [ ] Retrieval cơ bản (top-k), sau đó thêm reranking/hybrid search
- [ ] Viết prompt template chống hallucination, yêu cầu trích dẫn
- [ ] Sinh câu trả lời bằng LLM, kiểm tra kết quả trên golden set
- [ ] Ghi chép lại so sánh giữa các phiên bản pipeline để rút kinh nghiệm
- [ ] (Tuỳ chọn) Bọc thành demo Streamlit/Gradio để dùng thực tế

---

*Bản quy trình này tối ưu cho việc **học từng bước bằng cách tự tay chạy và so sánh**, thay vì triển khai production quy mô lớn. Khi đã nắm vững, có thể mở rộng sang các phần "nâng cao" trong bản quy trình gốc (multi-tenant, phân quyền, CI/CD, monitoring...).*
