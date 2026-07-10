# Báo Cáo Kỹ Thuật: Xây Dựng Hệ Thống RAG Học Tập Song Ngữ Việt–Anh

**Loại dự án:** Cá nhân, học tập, phi thương mại. **Ràng buộc thiết kế:** toàn bộ công cụ/mô hình phải miễn phí; ưu tiên chạy local, chỉ gọi API ngoài (LLM) khi bắt buộc.

> **Ghi chú cách đọc:** đây là báo cáo cuối cùng, ghi lại **cấu hình và số liệu tốt nhất** của hệ thống đã triển khai — không trình bày các phương án bị loại bỏ dọc đường. Mỗi mục kỹ thuật mở đầu bằng đoạn **"Khái niệm"** giải thích lý thuyết nền trước khi vào chi tiết, vì đây vừa là báo cáo vừa là tài liệu học tập. Bảng thuật ngữ đầy đủ ở Phụ lục C. Toàn bộ số liệu đã được **tái lập bằng cách chạy lại toàn bộ pipeline từ đầu** (trích xuất → chunking → embedding → retrieval → generation) ngay trước khi viết báo cáo này, xem Mục 4.

---

## Tóm tắt

Báo cáo trình bày một hệ thống **RAG** (Retrieval-Augmented Generation — Sinh văn bản có Tăng cường Truy xuất) hoàn chỉnh, xây dựng trên kho ngữ liệu song ngữ Việt–Anh gồm 62 tài liệu (nghiên cứu AI trong giáo dục và tài liệu kỹ thuật về RAG). RAG giúp mô hình ngôn ngữ lớn (LLM) trả lời dựa trên một kho tài liệu cụ thể thay vì chỉ dựa vào kiến thức đã học sẵn, nhờ đó giảm hiện tượng "ảo giác" (hallucination — mô hình bịa thông tin sai).

Hệ thống cuối cùng gồm: trích xuất tài liệu bằng Docling, phân đoạn (chunking) theo chiến lược lai section-based/document-aware/semantic, embedding bằng `BAAI/bge-m3`, lưu trữ trong Chroma (chỉ mục HNSW), truy xuất bằng Hybrid Search (BM25 + dense) kết hợp Reranker cross-encoder, sinh câu trả lời có trích dẫn nguồn qua Gemini/Groq, và đánh giá bằng LLM-as-judge trên toàn bộ 65 câu hỏi golden set (bộ câu hỏi có đáp án chuẩn).

**Kết quả cuối cùng** (đo trên toàn bộ 65/65 câu hỏi, corpus 7.044 chunk): Retrieval đạt Recall@3 = 69.2%, Recall@10 = 78.5%, MRR = 0.867. Generation đạt Correctness = 76.9% (50/65), Faithfulness = 81.5% (53/65).

---

## 1. Giới thiệu

### 1.1 RAG là gì

**Khái niệm:** một LLM (Large Language Model — mô hình ngôn ngữ lớn, ví dụ GPT/Gemini/Llama) được huấn luyện trên một khối lượng văn bản khổng lồ tính đến một mốc thời gian nhất định (**knowledge cutoff**). Sau đó, kiến thức của nó **tĩnh** — không tự biết các sự kiện xảy ra sau đó hay nội dung tài liệu riêng tư/nội bộ. Khi bị hỏi về điều nó không biết, LLM có xu hướng **hallucination** (ảo giác) — sinh câu trả lời trôi chảy nhưng bịa đặt, vì bản chất nó chỉ dự đoán từ tiếp theo có xác suất cao nhất chứ không "tra cứu sự thật".

**RAG** giải quyết vấn đề này bằng cách: trước khi trả lời, hệ thống **tìm kiếm** (retrieve) các đoạn văn bản liên quan từ một kho tài liệu ngoài, đưa vào làm **ngữ cảnh** (context) cho LLM, và yêu cầu LLM trả lời dựa trên ngữ cảnh đó. So với hai hướng thay thế — **Fine-tuning** (huấn luyện lại một phần trọng số mô hình, tốn kém, cần GPU mạnh) và **Prompt Engineering** thuần tuý (chỉ đổi cách hỏi, không thêm kiến thức) — RAG là lựa chọn phù hợp khi cần câu trả lời bám sát một kho tài liệu cụ thể mà không có tài nguyên huấn luyện lại mô hình.

### 1.2 Mục tiêu và phạm vi

Xây dựng hệ thống hỏi-đáp dựa trên tài liệu, trả lời chính xác kèm trích dẫn nguồn, hạn chế hallucination, hoạt động tốt trên corpus song ngữ, và được đánh giá bằng số liệu định lượng thực tế — toàn bộ chạy miễn phí (local hoặc free-tier API), phù hợp dự án học tập cá nhân quy mô ~62 tài liệu, ~7.000 chunk.

---

## 2. Dữ liệu

### 2.1 Kho ngữ liệu (corpus)

**Khái niệm:** "corpus" là tập hợp tài liệu nguồn mà hệ thống tìm kiếm và trả lời dựa trên đó — khác với kiến thức nội tại LLM đã học sẵn.

| Thuộc tính | Giá trị |
|---|---|
| Tổng số tài liệu | 62 (60 `personal/` + 2 `public/`) |
| Ngôn ngữ | Song ngữ Việt–Anh |
| Định dạng | PDF (đa số), 1 HTML |
| Số đoạn trích xuất thô | 15.608 |
| Số chunk | 7.044 |
| Số vector trong Chroma | 7.044 |

Trùng lặp được loại bỏ bằng **MD5** — hàm băm (hash function) biến nội dung file thành một chuỗi cố định, hai file giống hệt nhau luôn cho cùng mã băm. Tên file được chuẩn hoá kebab-case dựa trên tiêu đề/tác giả thật, thay vì giữ tên gốc dạng mã định danh khó đọc.

### 2.2 Bộ câu hỏi đánh giá (golden set)

**Khái niệm:** để đánh giá khách quan (không chỉ "thử vài câu rồi cảm nhận"), cần một **golden set** — bộ câu hỏi kèm đáp án chuẩn đã biết trước, dùng làm đề kiểm tra tự động.

`data/golden_set.json`: **65 câu hỏi, bao phủ 62/62 tài liệu (100%)**. Mỗi mục gồm `question`, `expected_answer` (đáp án tham khảo), `source`/`page` (tài liệu và trang chứa đáp án), `context_snippet` (trích nguyên văn tài liệu gốc để xác minh retrieval tìm đúng đoạn).

---

## 3. Kiến trúc hệ thống

Pipeline gồm 11 giai đoạn: chuẩn bị dữ liệu → trích xuất → chunking → embedding → vector database → retrieval → prompt → generation → evaluation → kỹ thuật nâng cao → demo.

### 3.1 Trích xuất văn bản

**Khái niệm:** PDF không lưu văn bản theo cấu trúc đoạn/câu mà lưu **toạ độ từng ký tự**. "Đọc PDF" là suy luận ngược cấu trúc đoạn văn từ hàng nghìn toạ độ rời rạc. Các công cụ đơn giản dùng **heuristic** (quy tắc kinh nghiệm, ví dụ "ký tự gần nhau theo hàng ngang thuộc cùng dòng") — thất bại với bảng không viền vì không có đường kẻ để bám.

**Công cụ:** **Docling** (IBM) — dùng **layout model** (mô hình thị giác máy tính nhận diện loại phần tử: đoạn văn, tiêu đề, bảng...) trước khi trích xuất, giống cách mắt người nhận ra ngay đâu là bảng dù không viền. OCR (Optical Character Recognition — nhận diện ký tự từ ảnh, cần cho bản scan) được tắt vì toàn bộ tài liệu là PDF digital-born (có sẵn lớp text).

**Ví dụ thật** (trích từ `trocr-transformer-based-ocr_li-et-al-2021.pdf`, trang 1, heading "Abstract"):
> *"Text recognition is a long-standing research problem for document digitalization. Existing approaches are usually built based on CNN for image understanding and RNN for char-level text generation... In this paper, we propose an end-to-end text recognition approach with pre-trained image Transformer and text Transformer models, namely TrOCR..."*

Bảng dữ liệu được tái tạo đúng cấu trúc dạng markdown (ví dụ bảng thống kê dataset trong `beyond-benchmarks-embedding-model-similarity-rag_caspari.pdf`, trang 2) thay vì bị hoà lẫn vào văn bản xung quanh như các công cụ đơn giản. Kết quả tổng thể: 15.608 đoạn trích xuất từ 62 tài liệu, heading và cấu trúc bảng được bảo toàn đầy đủ trong metadata.

**Cache:** mỗi file được lưu kết quả theo mã băm MD5 của chính nó — lần chạy lại không đổi nội dung sẽ đọc thẳng từ cache thay vì chạy lại Docling.

### 3.2 Phân đoạn văn bản (Chunking)

**Khái niệm:** tài liệu cần chia nhỏ vì (1) LLM có giới hạn độ dài xử lý một lượt (**context window**), và (2) tìm kiếm trên đơn vị nhỏ (chunk) chính xác hơn tìm trên cả tài liệu dài. Nhưng chia quá nhỏ sẽ mất ngữ cảnh cần thiết để hiểu đúng nghĩa — đây là đánh đổi cốt lõi của chunking.

**Chiến lược áp dụng** — kết hợp 3 kỹ thuật: (1) **section-based** — gộp các đoạn liên tiếp cùng heading thành 1 chunk; (2) **document-aware** — bảng/code luôn tách chunk riêng, không cắt ngang; (3) **semantic fallback** — khi một mục quá dài (>1800 ký tự) không có tiêu đề con để chia tự nhiên, tách theo câu và cắt tại điểm độ tương đồng ngữ nghĩa giữa 2 câu liền kề giảm mạnh nhất, thay vì cắt cứng theo số ký tự.

**Ví dụ thật** — chunk dạng bảng (giữ nguyên cấu trúc, không lẫn với văn bản xung quanh):
```
Table 1: The datasets used for generating embeddings with their number of queries and corpus size.

| Dataset Name   |   Queries | Corpus   |
|----------------|-----------|----------|
| TREC-COVID     |        50 | 171k     |
| NFCorpus       |       323 | 3.6k     |
```

Kết quả: 7.044 chunk từ 15.608 đoạn thô. Cache theo tài liệu (MD5) giúp lần chạy lại không đổi giảm từ vài phút xuống vài giây.

### 3.3 Embedding và cơ sở dữ liệu vector

**Khái niệm:** một **embedding model** chuyển văn bản thành **vector** — dãy số thực có độ dài cố định — sao cho hai đoạn văn bản có ý nghĩa gần nhau thì vector "gần nhau" trong không gian nhiều chiều, đo bằng **cosine similarity** (độ tương đồng cosine — góc giữa 2 vector, từ -1 đến 1, không phụ thuộc độ dài văn bản). Nhờ vậy, "tìm đoạn văn liên quan" trở thành bài toán toán học "tìm vector gần nhau nhất".

**Model:** `BAAI/bge-m3` (chạy local) — hỗ trợ tới 8192 **token** (đơn vị nhỏ nhất LLM xử lý văn bản, không hẳn là 1 từ), đủ để **0/4415 chunk bị cắt cụt** khi embed, trong khi model giới hạn 512 token phổ biến sẽ cắt mất các bảng dữ liệu dài. Model đa ngôn ngữ thật, phù hợp corpus song ngữ.

**Cơ sở dữ liệu vector:** **Chroma** (local, không cần server riêng), khoảng cách cosine được chỉ định tường minh. Mỗi vector có ID ổn định (hash SHA-256 của `source+page+heading+text`) phục vụ **upsert** (cập nhật nếu ID đã tồn tại, thêm mới nếu chưa — tránh trùng lặp khi chạy lại).

**Thuật toán chỉ mục:** **HNSW** (Hierarchical Navigable Small World) — thuật toán **ANN** (Approximate Nearest Neighbor — tìm láng giềng gần đúng, không cần so sánh toàn bộ database như **brute-force**), xây cấu trúc đồ thị nhiều lớp để tìm nhanh. Ở quy mô corpus hiện tại, HNSW cho kết quả trùng khớp hoàn toàn với brute-force (không mất độ chính xác), tốc độ tương đương — tham số mặc định không cần tinh chỉnh.

**Ví dụ thật** — kết quả cuối: 7.044/7.044 chunk đã embed vào Chroma, xác nhận qua `col.count()`.

### 3.4 Truy xuất (Retrieval)

**Khái niệm:** **dense retrieval** (dùng embedding) hiểu ngữ nghĩa, tìm đúng dù từ ngữ khác nhau, nhưng có thể mờ với thuật ngữ/mã số cần khớp chính xác. **Sparse retrieval** (ví dụ **BM25** — cải tiến của TF-IDF, đếm tần suất từ khoá có trọng số ưu tiên từ hiếm) chính xác với từ khoá nhưng không hiểu từ đồng nghĩa. Hai phương pháp bổ trợ điểm yếu cho nhau.

**Hybrid Search:** kết hợp BM25 và dense retrieval bằng **Reciprocal Rank Fusion (RRF)** — vì hai thang điểm không cùng đơn vị, RRF chỉ dùng **thứ hạng**: mỗi chunk được cộng điểm `1/(60 + hạng)` từ mỗi danh sách nó xuất hiện, chunk được cả hai phương pháp đồng thuận xếp cao sẽ trồi lên đầu.

**Reranking:** dùng **cross-encoder** `BAAI/bge-reranker-v2-m3` — khác **bi-encoder** (embedding thông thường, mã hoá câu hỏi và chunk *độc lập* rồi so sánh, nhanh vì tính trước được), cross-encoder đọc **đồng thời cả cặp** (câu hỏi + chunk) nên chính xác hơn hẳn nhưng chậm hơn nhiều, chỉ khả thi trên tập ứng viên đã lọc nhỏ.

**Pipeline cuối cùng:** Hybrid Search lấy top-30 ứng viên → Reranker chấm điểm lại → top-5/top-3 gửi Generation.

**Ví dụ thật** (câu hỏi: *"Grammarly ra mắt vào năm nào và nó hỗ trợ người dùng những gì?"*, đáp án đúng ở `ai-va-ung-dung-trong-day-hoc-ngoai-ngu_bui-thi-minh.pdf`, trang 3): Hybrid+Rerank xếp đúng tài liệu/trang ở **hạng 1** với điểm tin cậy rất cao (rerank score = 0.998), vượt trội rõ rệt so với các ứng viên khác (hạng 2 trở đi rơi xuống dưới 0.26).

### 3.5 Tạo ngữ cảnh & Sinh câu trả lời

**Khái niệm:** **prompt** là toàn bộ văn bản gửi cho LLM (chỉ dẫn + ngữ cảnh + câu hỏi). **Grounding** (neo giữ căn cứ) là kỹ thuật thiết kế prompt ép LLM bám vào ngữ cảnh cung cấp thay vì tự bịa — cơ chế cốt lõi giúp RAG giảm hallucination.

**Prompt template:** Ngữ cảnh (mỗi chunk kèm nhãn `[Nguồn: file, trang N]`) → Câu hỏi → chỉ dẫn: chỉ dùng thông tin trong ngữ cảnh, nói rõ khi thiếu dữ liệu, trích dẫn nguồn cho mỗi ý quan trọng.

**Sinh câu trả lời:** LLM với `temperature=0.2` (tham số điều khiển độ ngẫu nhiên khi chọn từ — thấp = ổn định, phù hợp bài toán cần chính xác hơn sáng tạo). LLM sử dụng: Google Gemini cho demo tương tác (`app.py`), Groq (`llama-3.3-70b-versatile`) cho các đợt đánh giá hàng loạt.

**Ví dụ thật hoàn chỉnh** (câu hỏi Grammarly ở trên):

> **Câu trả lời:** *"Grammarly ra mắt vào năm 2009 (theo tài liệu "ai-va-ung-dung-trong-day-hoc-ngoai-ngu_bui-thi-minh.pdf", trang 3). Grammarly hỗ trợ người dùng bằng cách phát hiện và đề xuất sửa lỗi ngữ pháp, chính tả, dấu câu, và phong cách viết cho phù hợp... Ngoài ra, Grammarly còn có khả năng phát hiện đạo văn, gợi ý từ vựng phù hợp với ngữ cảnh, và điều chỉnh phong cách viết theo mục tiêu của người dùng (theo tài liệu..., trang 3)."*

Câu trả lời đúng nội dung, có trích dẫn nguồn rõ ràng, được cả hai phương pháp đánh giá (Mục 4.2) xác nhận PASS.

### 3.6 Answer Verification

**Khái niệm:** sau khi sinh câu trả lời, hệ thống gọi LLM thêm một lần thứ hai — đóng vai "người kiểm tra" — chấm lại **Faithfulness** (độ trung thực với ngữ cảnh) trước khi trả về người dùng cuối. Nếu FAIL, thử sinh lại một lần với chỉ dẫn chặt hơn; nếu vẫn FAIL, trả về kèm cảnh báo rõ ràng thay vì âm thầm che giấu vấn đề.

### 3.7 Giao diện demo

`app.py` (Streamlit — thư viện dựng giao diện web tương tác nhanh bằng Python) ghép pipeline hoàn chỉnh: Hybrid+Rerank → Prompt → Generation → Answer Verification. Model/kết nối tốn tài nguyên được cache (`@st.cache_resource`) để không tải lại mỗi lượt tương tác. Đã kiểm thử end-to-end bằng trình duyệt headless (Playwright): nhập câu hỏi thật, xác nhận pipeline chạy hoàn chỉnh, hiển thị đúng trạng thái xác minh, không lỗi console.

---

## 4. Kết quả cuối cùng

Toàn bộ số liệu dưới đây được đo (và tái lập lại để xác nhận) trên **golden set đầy đủ 65/65 câu hỏi**, corpus 7.044 chunk, sau khi chạy lại toàn bộ pipeline từ bước trích xuất — đây là số liệu **cuối cùng** của cấu hình đang vận hành thật (Hybrid+Rerank cho retrieval, LLM-as-judge qua Groq cho generation).

### 4.1 Retrieval (Hybrid Search + Reranker)

**Chỉ số đo lường:** **Recall@k** — trong k kết quả đầu tiên, đáp án đúng có xuất hiện hay không (chỉ tính có/không). **Precision@k** — trung bình bao nhiêu phần trăm kết quả trong k kết quả thực sự liên quan. **MRR** (Mean Reciprocal Rank) — trung bình nghịch đảo thứ hạng của kết quả đúng đầu tiên (đúng ở hạng 1 → 1.0 điểm, hạng 2 → 0.5, hạng 4 → 0.25...) qua toàn bộ câu hỏi — càng cao nghĩa là hệ thống càng đặt đáp án đúng gần đầu danh sách.

| Chỉ số | Giá trị |
|---|---|
| Recall@3 | **69.2%** (45/65) |
| Recall@5 | **75.4%** (49/65) |
| Recall@10 | **78.5%** (51/65) |
| Precision@3 | 0.677 |
| Precision@5 | 0.594 |
| Precision@10 | 0.489 |
| MRR | **0.867** |

**nDCG** (normalized Discounted Cumulative Gain — chỉ số xếp hạng nâng cao, hỗ trợ nhiều bậc liên quan thay vì đúng/sai nhị phân) không được tính vì golden set chỉ ghi nhận đúng 1 cặp (nguồn, trang) đúng/câu, không đủ dữ liệu "graded relevance" mà nDCG cần.

**Giới hạn quan sát được:** ~20% câu hỏi vẫn không tìm thấy đáp án đúng ở k=10. Kiểm tra mẫu cho thấy đây là giới hạn thật (không phải lỗi đo lường): đúng tài liệu được tìm thấy nhưng đúng đoạn chứa đáp án nằm ở chunk khác cạnh tranh cùng chủ đề trong tài liệu, hoặc câu hỏi đòi hỏi tổng hợp từ nhiều mục con — không hợp lý để mong đợi tìm thấy trong một lượt truy vấn đơn.

### 4.2 Generation (LLM-as-judge)

**Phương pháp đo lường:** **LLM-as-judge** — dùng một LLM khác đóng vai giám khảo, chấm CÓ/KHÔNG cho hai tiêu chí: **Answer Correctness** (câu trả lời có đúng nội dung so với đáp án tham khảo) và **Faithfulness** (câu trả lời có hoàn toàn dựa trên ngữ cảnh, không bịa thêm thông tin — đo trực tiếp hallucination). Đây là phương pháp đáng tin cậy hơn so khớp chuỗi ký tự đơn thuần vì hiểu được ngữ nghĩa và cách diễn đạt khác nhau, kể cả khi tài liệu nguồn và câu trả lời khác ngôn ngữ (corpus của dự án có nhiều tài liệu tiếng Anh nhưng hệ thống luôn trả lời tiếng Việt).

| Chỉ số | Kết quả |
|---|---|
| Answer Correctness | **76.9%** (50/65) |
| Faithfulness | **81.5%** (53/65) |

Đây là số liệu đo trên **toàn bộ** golden set (65/65 câu), dùng thống nhất pipeline retrieval tốt nhất (Hybrid+Rerank, giống hệt hệ thống demo thật) cho mọi câu hỏi, và model Groq `llama-3.3-70b-versatile` làm giám khảo cho toàn bộ.

---

## 5. Thảo luận

### 5.1 Khả năng mở rộng quy mô

Kết luận về HNSW (Mục 3.3) chỉ đúng ở quy mô hiện tại (~7.000 vector). Ở quy mô hàng trăm nghìn–triệu vector: brute-force sẽ không còn khả thi (HNSW trở thành bắt buộc); bộ nhớ RAM cần **quantization** (nén vector — giảm độ chính xác số học lưu trữ để giảm dung lượng); tham số HNSW cần đo lại bằng Recall@k thực tế; cần hạ tầng phân tán (**sharding** — chia dữ liệu nhiều máy chủ, kết hợp **replication** — sao chép dữ liệu tăng khả năng chịu lỗi); và Hybrid+Reranker chuyển từ "cải thiện đáng kể" sang "gần như bắt buộc" do nhiễu từ tài liệu tương tự chủ đề tăng mạnh.

### 5.2 Đánh đổi hiệu năng Reranker

Reranking chạy trên CPU (GPU máy chỉ có 3.68GB **VRAM** — bộ nhớ chuyên dụng của card đồ hoạ, không đủ chạy embedding lẫn reranker) tốn ~45 giây/câu hỏi cho 30 ứng viên — chấp nhận được cho đánh giá ngoại tuyến (offline) nhưng cần GPU cho production cần phản hồi thời gian thực.

### 5.3 Ràng buộc hạ tầng LLM miễn phí — vấn đề TPD

Đánh giá Generation trên toàn bộ 65 câu (130 lượt gọi LLM: sinh câu trả lời + giám khảo) vấp phải một giới hạn dễ bị bỏ sót: các nhà cung cấp LLM miễn phí công bố hạn mức theo **request** (RPD/RPM — số lượt gọi/ngày/phút) nhưng còn áp dụng riêng hạn mức theo **token** (TPD/TPM — số token/ngày/phút), thường **chặt hơn nhiều** khi prompt dài (ở đây mỗi lượt gọi RAG mang theo 5 chunk ngữ cảnh, có thể vài nghìn token). Với Groq free tier (TPD = 100.000 token/ngày cho `llama-3.3-70b-versatile`), ngân sách này chỉ đủ cho ~37-38 câu hỏi mỗi tài khoản — phải dùng tuần tự 3 tài khoản khác nhau (mỗi tài khoản có ngân sách TPD riêng biệt theo tổ chức) mới hoàn thành được 65 câu trong cùng một phiên làm việc.

**Bài học kỹ thuật:** khi gặp lỗi giới hạn tốc độ (HTTP 429), nên đọc trực tiếp header `retry-after` (thời gian server yêu cầu chờ) thay vì suy đoán qua các header mô tả cửa sổ per-minute (`x-ratelimit-reset-tokens`) — hai loại giới hạn (theo phút và theo ngày) có thể cùng tồn tại, và header per-minute sẽ báo sai (gần như luôn "0 giây") ngay cả khi giới hạn thực sự bị chặn là theo ngày.

### 5.4 Kinh nghiệm kỹ thuật khác

- **Metadata không đồng bộ khi cache-hit:** một script cập nhật lại tên file khi đọc từ cache (đúng), một script khác bỏ sót bước này — dẫn tới hàng trăm chunk tồn tại song song dưới cả tên cũ và tên mới trong vector DB, khiến số liệu Recall bị đánh giá thấp hơn thực tế. **Bài học:** đường dẫn xử lý cache-hit phải được kiểm tra đồng bộ hoá dữ liệu giống hệt đường dẫn xử lý mới.
- **Dữ liệu đánh giá cần khớp đúng đơn vị dữ liệu hệ thống:** đoạn trích xác minh trong golden set, nếu trải dài qua ranh giới của nhiều chunk nhỏ liền kề (do tài liệu bị tách quá nhỏ), sẽ luôn báo "không tìm thấy" khi kiểm tra từng chunk riêng lẻ dù nội dung có mặt đầy đủ trong hệ thống. **Bài học:** golden set cần được đối chiếu với đơn vị dữ liệu thực tế (chunk), không chỉ với tài liệu gốc.
- **Không giả định định dạng phản hồi API đồng nhất:** trong cùng một họ model, các phiên bản khác nhau có thể trả về cấu trúc dữ liệu khác nhau cho cùng một trường — cần một lớp chuẩn hoá dùng chung thay vì xử lý rải rác ở từng nơi gọi API.
- **Kết quả cần được cộng dồn (resume), không ghi đè:** khi một tác vụ đánh giá dài có thể bị gián đoạn giữa chừng bởi giới hạn hạ tầng bên ngoài, script cần đọc lại kết quả đã có và chỉ xử lý phần còn thiếu, lưu tiến độ sau mỗi bước — tránh mất toàn bộ dữ liệu đã tốn công/chi phí thu thập nếu phải dừng và chạy lại.

---

## 6. Hạn chế và Hướng cải thiện

Đối chiếu với số liệu ở Mục 4, hệ thống còn 2 khoảng cách chính đáng chú ý — trình bày kèm hướng khắc phục cụ thể, **chưa triển khai**, để lại làm việc tiếp theo.

### 6.1 Hạn chế: Recall chưa tới trần lý tưởng

Recall@10 = 78.5% nghĩa là ~1/5 câu hỏi không tìm ra đúng đoạn chứa đáp án dù đã lấy tới 10 kết quả. Ba hướng khắc phục, xếp theo chi phí thử nghiệm tăng dần:

- **Tăng số ứng viên trước khi rerank:** hiện Hybrid Search chỉ lấy top-30 ứng viên (`CANDIDATES_K` trong `scripts/rerank.py`) trước khi Reranker lọc lại. Nới lên 50-100 ứng viên có thể vớt lại các trường hợp đáp án đúng bị lọt ra ngoài top-30 ở bước hybrid — chi phí thấp, chỉ cần đổi 1 tham số và đo lại Recall.
- **HyDE** (Hypothetical Document Embeddings — "embedding tài liệu giả định"): thay vì embed thẳng câu hỏi, cho LLM sinh trước một câu trả lời giả định rồi embed câu đó để tìm kiếm. Khác **multi-query** (đã thử và cho kết quả âm tính, xem bản lưu trữ cũ), HyDE giải quyết đúng nguyên nhân "câu hỏi và đoạn văn chứa đáp án có văn phong khác nhau" (câu hỏi thường ngắn, đoạn văn trả lời dài và mang tính khẳng định) — cơ sở lý thuyết khác hẳn multi-query nên không chắc sẽ gặp cùng thất bại.
- **Xử lý câu hỏi nhiều phần:** một số câu miss đòi hỏi tổng hợp thông tin từ 2-3 mục con khác nhau trong tài liệu — không hợp lý để mong đợi 1 lượt truy vấn đơn tìm đủ. Có thể thêm bước tách câu hỏi phức tạp thành các câu hỏi con, truy vấn riêng từng câu rồi gộp kết quả.

### 6.2 Hạn chế: Precision thấp — nhiều chunk nhiễu trong ngữ cảnh

Precision@10 = 0.489 nghĩa là khoảng một nửa chunk đưa vào ngữ cảnh không thực sự liên quan — tăng token tiêu tốn và tăng rủi ro Generation bị phân tán bởi thông tin không liên quan. Hai hướng khắc phục:

- **Ngưỡng điểm rerank (score threshold):** hiện luôn lấy đủ top-5 bất kể điểm số tuyệt đối. Ví dụ thật đã quan sát (Mục 3.4): điểm rerank rơi mạnh từ 0.998 (hạng 1) xuống 0.254, 0.02... (hạng 2 trở đi) — đặt ngưỡng tối thiểu sẽ tự động loại chunk yếu, chấp nhận trả về ít hơn 5 chunk khi cần thay vì luôn nhét đủ số lượng.
- **MMR** (Maximal Marginal Relevance — mức liên quan biên tối đa): kỹ thuật chọn top-k ưu tiên vừa liên quan vừa **đa dạng** giữa các chunk được chọn, tránh tình trạng 3-4/5 chunk cuối cùng đều là các đoạn gần giống nhau trích từ cùng một tài liệu — dễ xảy ra vì corpus có nhiều tài liệu cùng chủ đề (AI trong giáo dục).

### 6.3 Hướng đầu tư dài hạn hơn

**Fine-tuning nhẹ reranker/embedding trên chính golden set:** 65 câu hỏi hiện có sẵn dạng cặp (câu hỏi, đoạn đúng) — đủ để tinh chỉnh nhẹ (fine-tune) model reranker hoặc embedding cho "khẩu vị" riêng của corpus này, thay vì dùng nguyên model tổng quát đã huấn luyện trên dữ liệu khác. Chi phí công sức cao hơn 3 hướng trên nhưng tiềm năng cải thiện đồng đều cả Recall lẫn Precision cùng lúc.

**Khuyến nghị thứ tự thử nghiệm:** bắt đầu với tăng `CANDIDATES_K` và ngưỡng rerank score (rẻ, đo được ngay trên golden set hiện có) trước khi đầu tư vào HyDE hoặc fine-tuning.

---

## 7. Kết luận

Hệ thống RAG song ngữ Việt–Anh đã được xây dựng hoàn chỉnh qua 11 giai đoạn và **kiểm chứng lại toàn bộ bằng một lần chạy end-to-end từ đầu**: trích xuất, chunking, embedding cho kết quả tái lập chính xác (62 tài liệu, 7.044 chunk, không lệch); Retrieval đạt Recall@3 = 69.2%, MRR = 0.867 trên toàn bộ golden set; Generation đạt Correctness = 76.9%, Faithfulness = 81.5% — đo bằng LLM-as-judge trên đầy đủ 65/65 câu hỏi, không còn là ước lượng trên tập con.

**Hướng phát triển tiếp theo:** triển khai và đo lại các hướng cải thiện Recall/Precision ở Mục 6; đánh giá lại các giả định về quy mô (Mục 5.1) nếu corpus mở rộng đáng kể; cân nhắc phân bổ hạn mức LLM (nhiều tài khoản/nhà cung cấp, chạy trải nhiều ngày) như một phần thiết kế chính thức nếu tiếp tục mở rộng golden set trong tương lai, thay vì xử lý phát sinh như hiện tại.

---

## Phụ lục A — Cấu trúc mã nguồn

```
data/
├── raw/{personal,public}/         62 tài liệu nguồn (PDF/HTML)
├── processed/
│   ├── documents.jsonl            văn bản trích xuất (Docling)
│   ├── chunks.jsonl                chunk sau xử lý
│   └── cache_extract/, cache_chunks/   cache theo MD5
├── chroma_db/                     vector DB (collection `rag_hoc_tap`)
├── golden_set.json                65 câu hỏi đánh giá
└── generation_eval_groq.json      kết quả đánh giá Generation chi tiết (65/65 câu)

app.py                             demo Streamlit

scripts/
├── _common.py                     tiện ích dùng chung (file_md5, extract_text)
├── extract_text.py                trích xuất (Docling, có cache)
├── rename_new_docs.py             tự động đổi tên tài liệu mới (Gemini)
├── chunk_text.py                  chunking (section + document-aware + semantic)
├── embed_and_store.py             embedding (bge-m3) + lưu Chroma
├── retrieve.py                    dense retrieval + Recall/Precision/MRR
├── build_prompt.py                lắp ráp prompt chống hallucination
├── generate.py                    sinh câu trả lời (Gemini)
├── evaluate.py                    đánh giá Generation qua Gemini (fuzzy + LLM-judge)
├── evaluate_groq_batch.py         đánh giá Generation đầy đủ qua Groq (resumable, dùng khi hết quota Gemini)
├── hybrid_retrieve.py             Hybrid Search (BM25 + Dense, RRF)
├── rerank.py                      Reranker cross-encoder (bge-reranker-v2-m3)
├── evaluate_advanced_retrieval.py đánh giá Retrieval Hybrid+Rerank trên toàn bộ golden set
└── verify.py                      Answer Verification (kiểm tra Faithfulness trước khi trả lời)
```

## Phụ lục B — Hướng dẫn triển khai

```bash
# 1. Kích hoạt môi trường (mỗi phiên terminal mới)
cd /home/hoangthang/Chatbot && source rag-env/bin/activate

# 2. Cấu hình API key (chỉ 1 lần)
cp .env.example .env
# điền GOOGLE_API_KEY (aistudio.google.com/apikey) và GROQ_API_KEY (console.groq.com) — cả hai đều miễn phí

# 3. Thêm tài liệu mới
cp file-moi.pdf data/raw/personal/
python scripts/rename_new_docs.py      # đổi tên tự động
python scripts/extract_text.py         # trích xuất → documents.jsonl
python scripts/chunk_text.py           # chunking → chunks.jsonl
python scripts/embed_and_store.py      # embedding → Chroma

# 4. Chạy demo
streamlit run app.py

# 5. (Tuỳ chọn) Đánh giá lại toàn bộ golden set
python scripts/retrieve.py                       # Recall/Precision/MRR dense-only
python scripts/evaluate_advanced_retrieval.py     # Hybrid+Rerank (mất ~50 phút, rerank CPU)
python scripts/evaluate_groq_batch.py             # Generation qua Groq (resumable — chạy lại nếu bị ngắt giữa chừng do giới hạn TPD)
```

Nhờ cơ chế cache theo MD5, bước (3) chỉ tốn thời gian cho tài liệu mới.

## Phụ lục C — Bảng thuật ngữ

| Thuật ngữ / Viết tắt | Giải thích |
|---|---|
| **RAG** (Retrieval-Augmented Generation) | Kỹ thuật cho LLM trả lời dựa trên đoạn văn bản tìm được từ kho tài liệu ngoài |
| **LLM** (Large Language Model) | Mô hình ngôn ngữ lớn (ví dụ GPT, Gemini, Llama) |
| **Hallucination** | Hiện tượng LLM sinh thông tin sai sự thật nhưng nghe hợp lý |
| **Fine-tuning** | Huấn luyện lại một phần trọng số mô hình có sẵn trên dữ liệu mới |
| **Prompt** | Toàn bộ văn bản đầu vào gửi cho LLM |
| **Temperature** | Tham số điều khiển độ ngẫu nhiên khi LLM chọn từ tiếp theo |
| **Token** | Đơn vị nhỏ nhất LLM xử lý văn bản |
| **Context window** | Giới hạn độ dài văn bản LLM xử lý trong một lượt gọi |
| **Chunk / Chunking** | Đoạn văn bản nhỏ sau khi chia tách tài liệu / quá trình chia tách |
| **Embedding** | Vector số đại diện ý nghĩa văn bản; cũng chỉ mô hình tạo ra nó |
| **Cosine similarity** | Thước đo độ tương đồng giữa 2 vector dựa trên góc giữa chúng |
| **Vector database** | Cơ sở dữ liệu chuyên lưu trữ và tìm kiếm vector hiệu quả |
| **HNSW** | Thuật toán tìm kiếm láng giềng gần đúng bằng cấu trúc đồ thị nhiều lớp |
| **ANN** (Approximate Nearest Neighbor) | Tìm kiếm láng giềng gần nhất xấp xỉ, đánh đổi độ chính xác lấy tốc độ |
| **Brute-force** | Tìm kiếm so sánh toàn bộ dữ liệu, chính xác 100% nhưng chậm ở quy mô lớn |
| **Dense retrieval** | Tìm kiếm bằng embedding/vector, hiểu ngữ nghĩa |
| **Sparse retrieval / BM25** | Tìm kiếm dựa trên tần suất từ khoá |
| **Hybrid search** | Kết hợp dense và sparse retrieval |
| **RRF** (Reciprocal Rank Fusion) | Hợp nhất nhiều danh sách xếp hạng dựa trên thứ hạng |
| **Bi-encoder** | Mã hoá 2 văn bản độc lập rồi so sánh vector (nhanh) |
| **Cross-encoder** | Mã hoá đồng thời cả cặp văn bản (chính xác hơn, chậm hơn) |
| **Reranking** | Chấm điểm lại độ liên quan của tập ứng viên đã lọc sẵn |
| **HyDE** (Hypothetical Document Embeddings) | Cho LLM sinh câu trả lời giả định trước, dùng câu đó để tìm kiếm thay vì dùng thẳng câu hỏi gốc |
| **MMR** (Maximal Marginal Relevance) | Kỹ thuật chọn kết quả vừa liên quan vừa đa dạng, tránh trùng lặp nội dung giữa các chunk được chọn |
| **Recall@k** | Tỷ lệ câu hỏi tìm được đáp án đúng trong k kết quả đầu |
| **Precision@k** | Tỷ lệ trung bình kết quả liên quan trong k kết quả trả về |
| **MRR** | Trung bình nghịch đảo thứ hạng của kết quả đúng đầu tiên |
| **nDCG** | Chỉ số xếp hạng nâng cao, hỗ trợ nhiều bậc liên quan |
| **LLM-as-judge** | Dùng một LLM để chấm điểm chất lượng đầu ra hệ thống khác |
| **Golden set** | Bộ câu hỏi kèm đáp án chuẩn dùng để đánh giá khách quan |
| **Grounding** | Kỹ thuật thiết kế prompt ép LLM bám vào ngữ cảnh cung cấp |
| **API** | Giao diện lập trình để gọi dịch vụ của chương trình khác |
| **Quota / Rate limit** | Hạn mức số lượt gọi API / giới hạn tốc độ gọi |
| **RPD / RPM** | Requests Per Day / Per Minute — giới hạn số LƯỢT GỌI theo ngày/phút |
| **TPD / TPM** | Tokens Per Day / Per Minute — giới hạn số TOKEN xử lý theo ngày/phút, thường chặt hơn RPD/RPM với prompt dài |
| **MD5 / SHA-256** | Hàm băm — biến nội dung bất kỳ thành mã cố định, dùng kiểm tra trùng lặp hoặc tạo ID ổn định |
| **OCR** | Nhận diện ký tự quang học — "đọc" chữ từ ảnh scan |
| **Quantization** | Nén vector bằng cách giảm độ chính xác số học lưu trữ |
| **Sharding** | Chia dữ liệu ra nhiều máy chủ để mở rộng quy mô |
| **Replication** | Sao chép dữ liệu ra nhiều bản để tăng khả năng chịu lỗi |
| **VRAM** | Bộ nhớ chuyên dụng của card đồ hoạ (GPU) |
