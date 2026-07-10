# Công Nghệ & Kỹ Thuật Chi Tiết Theo Từng Bước

> Mục đích: tài liệu học sâu, đi vào cơ chế hoạt động thật của từng công nghệ/kỹ thuật
> đang dùng trong dự án (không chỉ liệt kê tên). Mỗi bước gồm: **công nghệ gì**, **hoạt
> động ra sao** (khái niệm nền tảng), **vì sao chọn** (đánh đổi), và **hàm nào làm việc
> đó** trong code thật. Đọc song song với file script tương ứng để đối chiếu.
>
> Khác với `GHI-CHU.md` (nhật ký tiến độ, số liệu benchmark theo thời gian), file này
> đứng yên — tập trung giải thích *cơ chế*, không cập nhật số liệu chạy.

---

## Bước 2 — Trích xuất: Docling

**File:** [scripts/extract_text.py](scripts/extract_text.py)

### Công nghệ: Docling — document understanding model, không phải parser text thuần

Khác với PyMuPDF/pdfplumber (chỉ đọc toạ độ ký tự trên trang PDF và ghép lại thành
dòng/đoạn theo heuristic khoảng cách), Docling dùng **layout model** (thị giác máy tính
train trên dữ liệu tài liệu) để nhận diện *loại phần tử* (đoạn văn, heading, bảng, ảnh,
caption...) trước khi trích xuất text. Đó là lý do nó tái tạo được **bảng không viền** —
loại bảng mà PyMuPDF trả về "0 bảng phát hiện được" vì không có đường kẻ để dựa vào,
trong khi Docling nhận diện qua bố cục khoảng cách/căn chỉnh giống cách mắt người đọc.

```python
result = CONVERTER.convert(str(path))
doc = result.document
for item, _level in doc.iterate_items():
    label = str(getattr(item, "label", "")).split(".")[-1].lower()
```

`doc.iterate_items()` duyệt qua cây phần tử mà Docling đã dựng (không phải danh sách
ký tự phẳng) — mỗi `item` có `label` (`text`, `section_header`, `table`, `picture`...).
Với bảng, `item.export_to_markdown(doc)` gọi lại logic dựng bảng nội bộ của Docling để
xuất đúng cấu trúc hàng/cột dạng markdown — đây là phần "đắt" nhất về mặt tính toán.

### Khái niệm nền tảng: OCR pipeline và đánh đổi tốc độ/chất lượng

`PdfPipelineOptions.do_ocr` bật/tắt bước nhận diện ký tự quang học (OCR — đọc ảnh pixel
thành text) chạy trên **mọi trang**, kể cả trang đã có sẵn text layer (PDF "digital-born").
Vì corpus ở đây toàn PDF digital-born (không phải bản scan giấy), OCR là dư thừa — tắt đi
giảm 272s → 45s (nhanh 6 lần) trên sách 301 trang mà không mất chất lượng. Đây là ví dụ cụ
thể của nguyên tắc "đo trước khi tối ưu": quyết định dựa trên benchmark thật, không phải
suy đoán.

### Kỹ thuật cache: content-addressed caching theo MD5

```python
def file_md5(path: Path) -> str:  # scripts/_common.py
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
```

Đây là **content-addressed caching** (địa chỉ hoá theo nội dung, không theo tên) — khoá
cache là hash của *nội dung byte* file, không phải đường dẫn/tên file. Hệ quả quan trọng:
đổi tên file (vd qua `rename_new_docs.py`) không làm mất cache, vì MD5 không đổi. Đọc
theo khối 8KB (`f.read(8192)`) thay vì đọc cả file vào RAM cùng lúc — quan trọng khi có
file lớn (sách PDF hàng trăm trang).

`extract_with_docling()` chỉ chạy khi `cache_extract/<md5>.jsonl` chưa tồn tại; nếu có,
đọc thẳng cache và chỉ "patch" lại `source`/`type` trong metadata cho khớp tên/thư mục
hiện tại — tách biệt rõ "nội dung" (bất biến, cache được) khỏi "vị trí" (có thể đổi, patch
rẻ).

### Heading tracking: state machine đơn giản

```python
current_heading = None
for item, _level in doc.iterate_items():
    if label == "section_header":
        current_heading = text
    records.append({..., "metadata": {..., "heading": current_heading}})
```

Đây là một state machine 1 biến: mỗi khi gặp `section_header`, cập nhật "heading hiện
hành"; mọi phần tử sau đó (đoạn văn, bảng) được gắn heading này cho tới khi gặp heading
tiếp theo. Nhờ vậy, ở bước 3, các đoạn cùng heading gộp được với nhau mà không cần phân
tích cấu trúc tài liệu lại từ đầu.

---

## Bước 3 — Chunking: Section-based + Document-aware + Semantic (Hybrid 3 tầng)

**File:** [scripts/chunk_text.py](scripts/chunk_text.py)

### Vì sao không chunk theo số ký tự cố định (fixed-size chunking)?

Cách đơn giản nhất — cắt cứng mỗi N ký tự — có 2 vấn đề: (1) cắt ngang câu/bảng, phá vỡ
ngữ nghĩa; (2) không phân biệt được "1 đoạn văn ngắn tự nhiên" với "1 chương dài không có
heading con". Dự án dùng chiến lược 3 tầng, áp dụng theo thứ tự ưu tiên:

**Tầng 1 — Section-based (`chunk_document`):** gộp các bản ghi liên tiếp cùng `heading`
(từ bước 2) thành 1 "section". Đây là đơn vị ngữ nghĩa tự nhiên nhất — tác giả đã tự chia
theo heading rồi, nên tôn trọng ranh giới đó thay vì áp thêm quy tắc riêng.

**Tầng 2 — Document-aware:** bảng/code luôn `flush()` section đang gộp và tách thành
chunk riêng ngay lập tức — không bao giờ trộn với text xung quanh. Lý do: 1 bảng dữ liệu
mà bị cắt đôi hoặc lẫn với đoạn văn mô tả sẽ phá vỡ hoàn toàn cấu trúc hàng/cột khi LLM
đọc lại.

**Tầng 3 — Semantic fallback (`semantic_split`):** chỉ kích hoạt khi 1 section vượt quá
`SIZE_THRESHOLD = 1800` ký tự (vd 1 chương sách không chia heading con). Đây là phần kỹ
thuật đáng học nhất trong bước này:

```python
sentences = [s for s, _ in sentence_page_pairs]
embeddings = model.encode(sentences, normalize_embeddings=True)
sims = [float(np.dot(embeddings[i], embeddings[i + 1])) for i in range(len(embeddings) - 1)]
threshold = np.percentile(sims, BREAKPOINT_PERCENTILE)
```

### Khái niệm nền tảng: Cosine similarity giữa embedding câu liền kề

Mỗi câu được encode thành 1 vector (embedding) bằng model `bge-m3`. Vì
`normalize_embeddings=True` (vector đã chuẩn hoá về độ dài 1), tích vô hướng
`np.dot(embeddings[i], embeddings[i+1])` **chính là cosine similarity** giữa 2 câu liền
kề (công thức cosine similarity đầy đủ là `dot(a,b) / (|a| * |b|)`, nhưng khi `|a|=|b|=1`
thì mẫu số bằng 1, rút gọn về tích vô hướng thuần). Giá trị này đo "2 câu liền kề nói về
cùng 1 chủ đề đến mức nào" — gần 1 là rất giống nhau, gần 0 (hoặc âm) là rất khác.

`np.percentile(sims, 25)` lấy ngưỡng ở **percentile thứ 25** của toàn bộ độ tương đồng
trong section đó — tức là 25% các cặp câu liền kề có độ tương đồng thấp nhất so với ngưỡng
này được coi là "điểm chuyển chủ đề" (breakpoint) và bị cắt tại đó. Đây là kỹ thuật
**adaptive threshold** (ngưỡng thích nghi theo từng section) thay vì 1 số cố định — vì
"độ tương đồng cao/thấp" mang tính tương đối, phụ thuộc vào chính văn phong của đoạn văn
đang xét, không có 1 con số tuyệt đối đúng cho mọi tài liệu.

```python
should_break = sim <= threshold or current_len >= SEMANTIC_MAX_CHUNK
```

Có thêm điều kiện hard cap `SEMANTIC_MAX_CHUNK = 2000` ký tự — phòng trường hợp cả section
tương đồng đều đặn (không có điểm chuyển chủ đề rõ) thì vẫn phải cắt để tránh 1 chunk quá
lớn.

### Vì sao ép `device="cpu"` cho embedding model ở đây?

```python
_EMBED_MODEL = SentenceTransformer("BAAI/bge-m3", device="cpu")
```

Máy chạy dự án có GPU dung lượng nhỏ (3.68GB VRAM) — từng gặp `CUDA OutOfMemoryError` khi
encode 1 batch câu cho section rất dài. Đây là batch job chạy 1 lần (không cần real-time),
nên đánh đổi tốc độ lấy sự ổn định là hợp lý — không cần GPU cho tác vụ không yêu cầu
latency thấp.

### Cache theo tài liệu, không theo toàn corpus

`chunk_document()` chỉ nhận bản ghi của **1 tài liệu** — tách riêng hàm này (thay vì xử lý
cả corpus 1 lượt) để cache độc lập theo từng file (`cache_chunks/<md5>.jsonl`, tra cứu qua
`extract_manifest.json` mà bước 2 đã ghi). Nguyên lý: chunking 1 tài liệu không phụ thuộc
tài liệu khác, nên có thể cache/tái sử dụng ở đơn vị nhỏ nhất có thể — thêm 1 tài liệu mới
không kích hoạt chunk lại toàn bộ 62 tài liệu cũ.

---

## Bước 4 — Embedding & Vector DB: bge-m3 + ChromaDB

**File:** [scripts/embed_and_store.py](scripts/embed_and_store.py)

### Khái niệm nền tảng: Embedding là gì và vì sao chọn bge-m3

Embedding là hàm ánh xạ 1 đoạn text thành 1 vector số thực có số chiều cố định (bge-m3 ra
1024 chiều), sao cho *khoảng cách hình học* giữa 2 vector phản ánh *độ tương đồng ngữ
nghĩa* giữa 2 đoạn text — 2 câu nói cùng ý nhưng khác từ vẫn cho vector gần nhau. Đây là
nền tảng của "tìm kiếm theo ý nghĩa" (semantic search) thay vì chỉ khớp từ khoá.

`BAAI/bge-m3` được chọn vì 2 lý do đo được cụ thể: (1) hỗ trợ **8192 token** input — corpus
có bảng dữ liệu dài, các model giới hạn 512 token (e5, mpnet) sẽ cắt cụt mất 95 chunk (chủ
yếu là bảng); bge-m3 cắt cụt 0/4415 chunk; (2) **đa ngôn ngữ** — corpus song ngữ Việt-Anh,
cần 1 không gian vector chung biểu diễn được cả 2 ngôn ngữ để câu hỏi tiếng Việt vẫn tìm ra
được tài liệu tiếng Anh liên quan (và ngược lại).

### ChromaDB và HNSW index

```python
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    configuration={"hnsw": {"space": "cosine"}},
)
```

ChromaDB lưu vector và index chúng bằng **HNSW** (Hierarchical Navigable Small World) — 1
cấu trúc dữ liệu đồ thị nhiều tầng cho phép tìm "top-k vector gần nhất" trong tập hàng
nghìn/triệu vector mà **không cần so sánh tuần tự với từng vector** (approximate nearest
neighbor search — đánh đổi 1 chút độ chính xác lấy tốc độ gần như hằng số theo log(n)).

`space: "cosine"` chỉ định rõ HNSW đo khoảng cách bằng cosine — quan trọng vì **Chroma
mặc định dùng "l2" (khoảng cách Euclid)**, không phải cosine. Nếu không khai báo rõ, index
sẽ đo sai loại khoảng cách so với cách model bge-m3 được train/tối ưu (các model embedding
hiện đại hầu hết tối ưu cho cosine, vì độ dài vector không mang thông tin, chỉ hướng vector
mới quan trọng).

### Content-addressed upsert — áp dụng lại ý tưởng từ bước 2

```python
def chunk_id(chunk: dict) -> str:
    raw = f"{m['source']}|{m['page']}|{m.get('heading')}|{chunk['text']}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
```

Cùng nguyên lý content-addressing như `file_md5()` ở bước 2, nhưng áp dụng cho **chunk**
thay vì file: ID là hash của nội dung + vị trí chunk. Chunk không đổi → ID không đổi →
`collection.get(ids=ids)` phát hiện đã tồn tại → bỏ qua, không embed lại (embedding tốn
thời gian GPU/CPU, đặc biệt khi corpus lớn dần theo thời gian). ChromaDB ở đây đóng vai trò
**cache bền vững** (persistent cache) cho kết quả embedding, không chỉ là nơi lưu trữ.

Batch theo `BATCH_SIZE = 100` khi encode — tránh nạp toàn bộ hàng nghìn chunk vào RAM/VRAM
cùng lúc một khi corpus tiếp tục phình to.

**Nguyên tắc bắt buộc phải nhớ:** model dùng để embed *chunk* (lúc index) và model dùng để
embed *câu hỏi* (lúc query, bước 6) phải là **cùng 1 model, cùng cấu hình chuẩn hoá**.
Nếu lệch (vd index bằng bge-m3 nhưng query bằng model khác), 2 không gian vector không
tương thích — khoảng cách đo được vô nghĩa dù code không báo lỗi gì.

---

## Bước 6 — Retrieval: Dense Search + đo lường Recall@k / Precision@k / MRR

**File:** [scripts/retrieve.py](scripts/retrieve.py)

### Dense retrieval hoạt động thế nào

```python
def retrieve(collection, model, query: str, k: int) -> list[dict]:
    q_emb = model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=k)
```

"Dense" nghĩa là biểu diễn bằng vector số thực dày đặc (mọi chiều đều có giá trị khác 0),
đối lập với biểu diễn "sparse" như BM25 (đa số chiều bằng 0, chỉ các từ xuất hiện mới có
giá trị — xem bước 10). Câu hỏi được embed bằng đúng model đã index (bge-m3), rồi
`collection.query()` dùng HNSW để tìm k vector chunk gần nhất trong không gian cosine.

### Khái niệm nền tảng: 3 chỉ số đánh giá retrieval

```python
if any(is_snippet_match[:k]):
    recall[k] += 1
precision_sum[k] += sum(is_relevant_source[:k]) / k
first_relevant_rank = next((i + 1 for i, rel in enumerate(is_relevant_source) if rel), None)
if first_relevant_rank:
    mrr_sum += 1.0 / first_relevant_rank
```

- **Recall@k**: trong top-k trả về, có tồn tại ít nhất 1 chunk đúng hay không (nhị phân,
  tính trên MỖI câu hỏi rồi lấy trung bình). Trả lời câu hỏi "hệ thống có TÌM RA được đáp
  án đâu đó trong top-k không?" — không quan tâm thứ hạng chính xác, không quan tâm có bao
  nhiêu chunk sai lẫn vào.
- **Precision@k**: trong k chunk trả về, bao nhiêu phần trăm thực sự liên quan (`đúng
  source / k`). Trả lời câu hỏi "top-k có bị nhiễu bởi chunk sai nhiều không?" — Recall cao
  nhưng Precision thấp nghĩa là tìm ra đáp án đúng, nhưng lẫn nhiều rác xung quanh (chunk
  "gần đúng chủ đề" nhưng không thực sự liên quan) — đúng vấn đề mà Hybrid+Rerank ở bước 10
  giải quyết.
- **MRR (Mean Reciprocal Rank)**: trung bình của `1/hạng` của kết quả đúng ĐẦU TIÊN. Nếu
  kết quả đúng luôn nằm ở vị trí 1 → MRR = 1.0 (tối đa). Nếu luôn nằm ở vị trí 4 → MRR =
  0.25. Chỉ số này nhạy với **thứ hạng** hơn Recall — phạt nặng khi đáp án đúng bị đẩy
  xuống thấp dù vẫn nằm trong top-k.

Vì sao 3 chỉ số cùng cần thiết: Recall cao + MRR cao nhưng Precision trung bình (đúng hiện
trạng dự án ở bước 6/9: MRR 0.875 nhưng Precision@k chỉ 0.452-0.645) nghĩa là hệ thống
*tìm ra được* đáp án và thường xếp nó khá cao, nhưng đồng thời kéo theo khá nhiều chunk
không liên quan trong cùng top-k — 3 số cùng nhìn mới thấy được bức tranh đầy đủ, chỉ nhìn
1 số sẽ đánh giá sai vấn đề.

### Kỹ thuật fuzzy matching để tự động hoá việc chấm điểm

```python
sm = SequenceMatcher(None, part_words, chunk_words, autojunk=False)
matched = sum(block.size for block in sm.get_matching_blocks())
```

`difflib.SequenceMatcher` cài đặt thuật toán tìm **longest matching subsequence** (dãy con
khớp dài nhất) giữa 2 danh sách. Ở đây áp dụng trên danh sách **từ** (không phải ký tự) của
snippet đáp án đúng và text chunk — `get_matching_blocks()` trả về mọi đoạn khớp liên tục,
cộng dồn độ dài các đoạn đó lại rồi chia cho tổng số từ cần khớp, ra tỷ lệ khớp. Ngưỡng
`MATCH_RATIO_THRESHOLD = 0.7` quyết định "khớp đủ" hay không — cho phép sai khác nhỏ về
định dạng (dấu câu, khoảng trắng thừa) mà không cần khớp tuyệt đối từng ký tự, đồng thời
tránh false positive nếu chỉ dùng "substring khớp một phần" (đã kiểm chứng ở bước 3:
`SequenceMatcher` theo ký tự cho kết quả sai trên văn bản dài, phải đổi qua theo từ).

`retrieve_fn` được truyền vào `evaluate_recall_at_k()` như 1 **tham số hàm bậc cao**
(higher-order function) thay vì gọi thẳng Chroma bên trong — nhờ đó cùng 1 hàm đánh giá này
tái dùng được cho cả dense-only (bước 6), hybrid, hybrid+rerank, và multi-query (bước 10)
chỉ bằng cách đổi hàm truyền vào, không sửa logic tính điểm.

---

## Bước 7 — Prompt Engineering chống Hallucination

**File:** [scripts/build_prompt.py](scripts/build_prompt.py)

### Cấu trúc prompt và lý do từng phần

```python
PROMPT_TEMPLATE = """Bạn là trợ lý trả lời dựa trên tài liệu được cung cấp.
Chỉ sử dụng thông tin trong phần NGỮ CẢNH bên dưới.
Nếu không đủ thông tin để trả lời, hãy nói rõ là không tìm thấy.
Trích dẫn nguồn (tên tài liệu, trang) cho mỗi ý quan trọng.

NGỮ CẢNH:
{context}

CÂU HỎI: {question}

TRẢ LỜI:"""
```

Đây là kỹ thuật **grounding prompt** (neo câu trả lời vào ngữ cảnh cụ thể) — 3 chỉ dẫn
tường minh trong instruction đều nhắm trực tiếp vào vấn đề **hallucination** (LLM tự bịa
thông tin nghe có vẻ hợp lý nhưng sai/không có nguồn):

1. "Chỉ sử dụng thông tin trong NGỮ CẢNH" — giới hạn phạm vi kiến thức được phép dùng,
   thay vì để model tự do dùng kiến thức đã học trong quá trình training (vốn có thể lỗi
   thời hoặc sai lệch so với tài liệu cụ thể của người dùng).
2. "Nói rõ không tìm thấy khi thiếu dữ liệu" — cho phép model **từ chối trả lời** thay vì
   ép nó luôn phải sinh ra 1 câu trả lời (chính hành vi "luôn phải trả lời" là nguyên nhân
   gốc gây hallucination ở nhiều hệ thống).
3. "Trích dẫn nguồn" — không chỉ giúp người dùng tự kiểm chứng, mà còn **ép model bám vào
   context cụ thể** khi sinh câu trả lời, vì phải chỉ ra được câu đó lấy từ đâu.

Context luôn đặt **trước** câu hỏi trong template — cách sắp xếp này giúp model xử lý
thông tin nền trước khi thấy yêu cầu cụ thể, giống cấu trúc "đọc tài liệu rồi mới được hỏi"
tự nhiên hơn khi model đọc tuần tự token.

```python
label = f"[Nguồn: {meta['source']}, trang {meta['page']}]"
blocks.append(f"{label}\n{text}")
```

Mỗi chunk được gắn nhãn nguồn ngay phía trước nội dung — đây là điều kiện *cần* để yêu cầu
"trích dẫn nguồn" ở instruction khả thi được: model không thể trích dẫn đúng nếu context
không mang theo thông tin nguồn kèm nó.

---

## Bước 8 — Generation: LangChain + Gemini

**File:** [scripts/generate.py](scripts/generate.py)

### LangChain ở đây đóng vai trò gì

```python
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=TEMPERATURE)
response = llm.invoke(prompt)
```

`ChatGoogleGenerativeAI` là 1 lớp **wrapper chuẩn hoá** của LangChain quanh Gemini API —
cung cấp interface `.invoke(prompt) -> response` giống hệt cách gọi các LLM khác (OpenAI,
Anthropic...) mà LangChain hỗ trợ. Dự án dùng đúng 1 method (`.invoke`) cơ bản, không dùng
đến các tầng trừu tượng nâng cao hơn của LangChain (chain, agent, memory...) — nghĩa là
LangChain ở đây chỉ giữ vai trò **client Gemini có interface thống nhất**, không phải
framework điều phối phức tạp.

### Temperature — điều khiển độ "sáng tạo" của model

`TEMPERATURE = 0.2` (thấp, gần 0). Về mặt kỹ thuật, temperature điều chỉnh **phân phối xác
suất token đầu ra** trước khi model chọn token tiếp theo: temperature càng thấp, phân phối
càng "nhọn" quanh token có xác suất cao nhất (model gần như luôn chọn lựa chọn "an toàn
nhất" theo dữ liệu train + context) — càng cao, phân phối càng "phẳng" (model dễ chọn các
token ít khả năng hơn, sinh ra output đa dạng/sáng tạo hơn nhưng cũng dễ trôi khỏi context
hơn). Với bài toán hỏi-đáp dựa trên tài liệu (cần bám sát sự thật, không cần sáng tạo văn
phong), temperature thấp là lựa chọn đúng hướng — đã được minh chứng cụ thể ở bước 11 (2
lần chạy cùng 1 câu hỏi ra 2 câu trả lời hơi khác nhau vì 0.2 không phải 0 tuyệt đối, và
Answer Verification bắt được sự khác biệt đó).

### Vấn đề thực tế đã gặp: response.content không phải luôn là string

```python
# scripts/_common.py
def extract_text(response) -> str:
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return "".join(parts)
    return str(content)
```

Đây là bug thật đã phát hiện khi debug: hầu hết model Gemini trả `response.content` dạng
string thuần, nhưng `gemini-flash-latest` trả về dạng **list các block** (mỗi block là
dict có key `"text"`) — làm crash mọi chỗ gọi `.upper()`/`.split()` thẳng trên
`response.content` (ảnh hưởng 5 file: `verify.py`, `generate.py`, `multi_query.py`,
`evaluate.py`, `rename_new_docs.py`). Bài học kỹ thuật: **không được giả định cấu trúc dữ
liệu trả về của 1 API là cố định giữa các phiên bản/model khác nhau của cùng 1 nhà cung
cấp** — cần 1 lớp chuẩn hoá (adapter) ở 1 chỗ duy nhất thay vì xử lý rải rác ở từng nơi gọi.

### Model rotation — đối phó với giới hạn free tier

Free tier Gemini giới hạn ~20 request/ngày/model, và giới hạn được tính **riêng theo từng
tên model** — nên khi 1 model hết quota, đổi sang model khác (`gemini-2.5-flash` →
`gemini-2.5-flash-lite` → ... → `gemini-flash-latest`) vẫn còn quota riêng để tiếp tục
làm việc trong ngày. `multi_query.py` cài đặt việc này thành cơ chế tự động
(`get_llm_with_fallback()` thử lần lượt từng model trong `CANDIDATE_MODELS`, gọi thử
`llm.invoke("test")` để xác nhận THẬT SỰ còn quota chứ không chỉ khởi tạo client thành
công).

---

## Bước 9 — Evaluation: Fuzzy Match vs LLM-as-Judge

**File:** [scripts/evaluate.py](scripts/evaluate.py)

### Vì sao đo bằng 2 cách khác hẳn nhau

**Cách A (fuzzy, miễn phí):** dùng lại đúng kỹ thuật `SequenceMatcher` theo từ như bước 6,
nhưng áp dụng cho tầng Generation:
- *Correctness*: so khớp câu trả lời với `expected_answer` có sẵn trong golden set.
- *Faithfulness (proxy)*: so khớp câu trả lời với chính nội dung context đã dùng — không
  đo "đúng thực tế" mà đo "có bám vào context hay không" (bịa thêm từ ngoài context sẽ làm
  giảm tỷ lệ khớp).

Ưu điểm: không tốn quota LLM, chạy được trên toàn bộ golden set bất kể quota còn hay hết.
Nhược điểm: là **proxy** (đại diện gián tiếp) — câu trả lời diễn đạt đúng nhưng dùng từ
ngữ hoàn toàn khác `expected_answer` vẫn có thể bị chấm sai (điểm thấp), fuzzy match không
"hiểu nghĩa", chỉ đếm từ trùng.

**Cách B (LLM-as-judge, tốn quota):** đưa cả (câu hỏi, context, câu trả lời,
expected_answer) vào 1 prompt, nhờ chính Gemini chấm CÓ/KHÔNG cho từng tiêu chí:

```python
JUDGE_PROMPT = """...
1. FAITHFULNESS: câu trả lời có hoàn toàn dựa trên NGỮ CẢNH, không bịa thêm thông tin ngoài context không?
2. CORRECTNESS: nội dung câu trả lời có đúng/khớp với ĐÁP ÁN THAM KHẢO không (không cần giống hệt câu chữ)?
..."""
```

Đây là kỹ thuật LLM-as-judge — dùng chính khả năng hiểu ngôn ngữ tự nhiên của LLM để chấm
điểm "đúng nghĩa" thay vì "đúng từ", giải quyết đúng nhược điểm của Cách A. Đánh đổi: mỗi
câu tốn thêm 1 lần gọi LLM (ngoài lần gọi để sinh câu trả lời) — với quota ~20
request/ngày/model, LLM-judge trên toàn bộ golden set là không khả thi, nên chỉ chạy trên
`N_JUDGE = 3` câu để vẫn có dữ liệu so sánh, dành phần lớn quota còn lại cho Cách A (đo
rộng hơn dù thô hơn).

**Tại sao đo cả 2 cách rồi so sánh mức đồng thuận:** nếu Cách A (rẻ) và Cách B (đắt nhưng
chính xác hơn) *đồng thuận cao* trên tập mẫu nhỏ, có cơ sở tin tưởng Cách A đủ tốt để dùng
đo trên quy mô lớn khi quota hạn chế — đây là kỹ thuật **validation của 1 proxy metric
bằng 1 ground-truth metric đắt hơn**, phổ biến khi metric chính xác nhất lại tốn kém nhất
để tính trên quy mô lớn.

---

## Bước 10a — Hybrid Search: BM25 + Dense qua Reciprocal Rank Fusion

**File:** [scripts/hybrid_retrieve.py](scripts/hybrid_retrieve.py)

### Khái niệm nền tảng: BM25 là gì, vì sao bổ sung cho Dense

```python
from rank_bm25 import BM25Okapi
tokenized = [tokenize(c["text"]) for c in chunks]
_BM25_INDEX = BM25Okapi(tokenized)
scores = index.get_scores(tokenize(query))
```

BM25 (Best Matching 25) là thuật toán **sparse retrieval** kinh điển dựa trên khớp từ khoá
chính xác — về bản chất là 1 phiên bản cải tiến của TF-IDF: chấm điểm 1 chunk cao nếu nó
chứa nhiều từ trong câu hỏi, có trọng số theo:
- **TF (term frequency)**: từ khoá xuất hiện càng nhiều lần trong chunk, điểm càng cao,
  nhưng có hiệu ứng bão hoà (lần xuất hiện thứ 10 đóng góp ít hơn lần đầu).
  IDF (inverse document frequency): từ khoá càng hiếm gặp trong toàn bộ corpus, trọng số
  càng cao (từ phổ biến như "là", "và" gần như không đóng góp điểm).
- Chuẩn hoá theo độ dài document — tránh thiên vị chunk dài chỉ vì có nhiều từ hơn.

Điểm mạnh của BM25 đúng vào điểm yếu của dense embedding: embedding thiên về "ngữ nghĩa
tổng quát" nên đôi khi bỏ qua **tên riêng, mã số, thuật ngữ kỹ thuật hiếm gặp** (những thứ
mà bản thân embedding không "hiểu" là quan trọng về mặt ngữ nghĩa, nhưng lại là từ khoá
match chính xác quan trọng với người dùng). BM25 hoàn toàn local (`rank_bm25` thuần
Python), không cần LLM hay GPU — chi phí gần như bằng 0.

### Reciprocal Rank Fusion (RRF) — hợp nhất 2 danh sách xếp hạng khác bản chất

```python
for rank, r in enumerate(dense_results, start=1):
    rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
for rank, r in enumerate(bm25_results, start=1):
    rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
```

Vấn đề cần giải quyết: điểm dense (cosine similarity, khoảng 0-1) và điểm BM25 (không có
thang chuẩn, phụ thuộc độ dài corpus) **không cùng đơn vị**, không thể cộng/so sánh trực
tiếp. RRF né hoàn toàn vấn đề đó bằng cách **chỉ dùng thứ hạng (rank)**, bỏ qua giá trị
điểm số gốc: công thức `1/(RRF_K + rank)` cho điểm cao nếu 1 chunk đứng hạng cao (rank nhỏ)
ở bất kỳ danh sách nào, và **cộng dồn điểm từ cả 2 danh sách** — chunk được cả dense LẪN
BM25 cùng xếp hạng cao (đồng thuận giữa 2 phương pháp độc lập) sẽ có tổng điểm cao nhất,
được đẩy lên đầu danh sách hợp nhất. `RRF_K = 60` là hằng số làm mượt, giá trị phổ biến
trong tài liệu gốc về RRF — giảm chênh lệch điểm giữa hạng 1 và hạng 2 (nếu K nhỏ, chênh
lệch giữa hạng 1 và 2 sẽ quá lớn, gần như bỏ qua hoàn toàn kết quả từ hạng 2 trở đi).

`chunk_key()` dùng `(source, page, 80 ký tự đầu)` làm khoá định danh để so khớp 2 chunk từ
2 danh sách khác nguồn (dense trả `distance`, BM25 trả `score` — không có ID chung) —
đây là 1 giải pháp thực dụng (proxy key) thay vì cần 1 hệ thống ID tập trung.

---

## Bước 10b — Reranker: Cross-Encoder cục bộ

**File:** [scripts/rerank.py](scripts/rerank.py)

### Khái niệm nền tảng: Bi-encoder (embedding) vs Cross-encoder — khác nhau ở đâu

Đây là điểm quan trọng nhất để hiểu vì sao cần 1 tầng rerank riêng sau khi đã có
Hybrid Search:

- **Bi-encoder** (chính là cách bge-m3 hoạt động ở bước 4/6): encode câu hỏi và chunk
  **độc lập, riêng biệt**, mỗi thứ ra 1 vector, rồi so sánh 2 vector đó (cosine). Vì 2
  vector được tính tách rời, vector của mọi chunk **tính trước được 1 lần và lưu sẵn**
  (chính là lý do ChromaDB embed 1 lần rồi query nhanh nhiều lần) — rất nhanh ở thời điểm
  truy vấn, nhưng model không bao giờ "thấy" trực tiếp mối quan hệ giữa câu hỏi và chunk cụ
  thể, chỉ so sánh 2 điểm trong không gian vector đã cố định từ trước.

- **Cross-encoder** (`CrossEncoder`, model `bge-reranker-v2-m3`):

```python
pairs = [(query, c["text"]) for c in candidates]
scores = model.predict(pairs)
```

  Đưa **cả cặp (câu hỏi, chunk) vào cùng lúc** qua 1 model transformer duy nhất — model có
  thể "chú ý chéo" (cross-attention) giữa từng token của câu hỏi với từng token của chunk,
  nắm bắt được tương tác trực tiếp và tinh vi hơn nhiều so với so sánh 2 vector đã tính
  sẵn. Đánh đổi: **không thể tính trước/cache được** (vì kết quả phụ thuộc vào cặp cụ thể,
  không phải 1 chunk riêng lẻ) — phải chạy model 1 lần cho MỖI cặp (câu hỏi, chunk), chậm
  hơn nhiều so với bi-encoder.

### Kiến trúc 2 tầng: Retriever nhanh → Reranker chính xác

```python
candidates = hybrid_retrieve(collection, embed_model, query, CANDIDATES_K)  # 30 ứng viên, rẻ
return rerank(query, candidates, k)  # chấm lại 30 cặp, chọn top-5
```

Vì cross-encoder quá chậm để chạy trên toàn bộ ~7000 chunk mỗi câu hỏi, pipeline dùng
kiến trúc chuẩn trong retrieval hiện đại: tầng 1 (Hybrid Search, bi-encoder + BM25) rẻ và
nhanh, lọc từ toàn corpus xuống còn 30 ứng viên hợp lý; tầng 2 (cross-encoder) chỉ cần
chấm lại 30 cặp đó — đắt hơn trên mỗi cặp nhưng tổng chi phí vẫn nhỏ vì số lượng đã được
thu hẹp. Đây chính xác là mô tả "Retriever ưu tiên tốc độ (recall cao, lọc thô), Reranker
ưu tiên độ chính xác (precision cao, lọc tinh)".

Model `bge-reranker-v2-m3` cùng họ BGE với `bge-m3` (embedding) — cùng nhóm nghiên cứu, có
tính nhất quán và đã được kiểm chứng hoạt động tốt cùng nhau, cũng chạy local (không tốn
quota Gemini).

---

## Bước 10c — Multi-Query Expansion (đã thử, không dùng)

**File:** [scripts/multi_query.py](scripts/multi_query.py)

### Ý tưởng kỹ thuật

```python
QUERY_EXPANSION_PROMPT = """Cho câu hỏi sau, hãy viết lại thành {n} câu hỏi khác có CÙNG Ý NGHĨA..."""
queries = [question] + expand_query(llm, question)
```

Nhờ LLM sinh thêm N cách diễn đạt lại câu hỏi gốc, retrieval riêng cho từng biến thể, rồi
hợp nhất bằng **RRF** — tái sử dụng đúng cơ chế hợp nhất đã xây ở Hybrid Search (bước 10a),
minh hoạ cho việc 1 kỹ thuật nền tảng (RRF) áp dụng được cho nhiều bài toán hợp nhất danh
sách khác nhau, không chỉ riêng dense+BM25.

Động cơ: nếu người dùng hỏi bằng từ ngữ khác hẳn với từ ngữ trong tài liệu gốc (vd hỏi
"giá" nhưng tài liệu dùng từ "chi phí"), dense retrieval bằng đúng câu hỏi gốc có thể bỏ
sót — sinh thêm biến thể tăng khả năng ít nhất 1 trong số các cách diễn đạt khớp với từ
ngữ tài liệu dùng.

### Kết quả thực nghiệm: không cải thiện — bài học quan trọng

Đã đo trên toàn bộ golden set: MRR giảm nhẹ so với dense-only, không cải thiện Recall.
**Bài học kỹ thuật đáng nhớ:** multi-query giá trị nhất khi retrieval phụ thuộc nhiều vào
khớp từ khoá chính xác (BM25/TF-IDF) — nơi cách diễn đạt câu hỏi ảnh hưởng trực tiếp đến
kết quả khớp. Với dense retrieval dùng embedding đa ngôn ngữ mạnh như bge-m3 (vốn đã nắm
bắt được ngữ nghĩa vượt qua khác biệt từ ngữ bề mặt), lợi ích của việc diễn đạt lại câu hỏi
gần như bị model embedding "hấp thụ" từ trước — không có gì nhiều để multi-query bổ sung
thêm. Đây là ví dụ cụ thể việc 1 kỹ thuật "nghe hợp lý về mặt lý thuyết" vẫn cần đo thật
trước khi đưa vào production, không nên chỉ dựa vào trực giác.

---

## Bước 10d — Answer Verification: Self-Check trước khi trả lời

**File:** [scripts/verify.py](scripts/verify.py)

### Khác biệt cốt lõi so với Evaluation (bước 9)

`evaluate.py` (bước 9) chạy **offline**, có `expected_answer` để so sánh — đo lường chất
lượng hệ thống nói chung. `verify.py` chạy **trong production** — không có đáp án chuẩn
nào để so sánh với câu hỏi thật của người dùng, nên chỉ kiểm tra được **Faithfulness** (câu
trả lời có bám vào context đã truy xuất hay không), không kiểm tra được **Correctness**
(context đó có thực sự đúng/đủ để trả lời câu hỏi hay không — đó là việc của tầng Retrieval
phía trước).

```python
def verify_faithfulness(llm, question, answer, chunks) -> bool:
    prompt = VERIFY_PROMPT.format(...)
    response = llm.invoke(prompt)
    return "FAITHFULNESS: CÓ" in extract_text(response).upper()
```

Đây là kỹ thuật **self-consistency check bằng 1 lệnh gọi LLM thứ hai** — dùng lại đúng
model đã sinh câu trả lời (hoặc có thể dùng model khác) để **tự chấm điểm lại chính nó**,
đóng vai trò như 1 lớp kiểm duyệt độc lập trước khi thông tin đến tay người dùng cuối.

### Cơ chế retry có giới hạn — không che giấu thất bại

```python
while not verified and attempts <= MAX_RETRIES:
    prompt = build_prompt(question, chunks) + RETRY_SUFFIX
    ...
return {..., "warning": None if verified else "⚠️ Câu trả lời chưa được xác nhận..."}
```

Nếu verify FAIL, thử sinh lại tối đa `MAX_RETRIES = 1` lần với prompt bổ sung nhắc nhở chặt
hơn (`RETRY_SUFFIX`). Nếu vẫn FAIL sau khi hết lượt thử, hệ thống **trả về câu trả lời kèm
cảnh báo rõ ràng** thay vì âm thầm trả lời như bình thường — nguyên tắc thiết kế quan trọng
ở đây: khi không chắc chắn, minh bạch hoá sự không chắc chắn đó cho người dùng tự quyết
định, thay vì cố che giấu để trải nghiệm "trông mượt" hơn. Đây chính là bằng chứng thực tế
đã quan sát được ở bước 11 (cùng câu hỏi, 2 lần chạy verified khác nhau do temperature>0) —
cơ chế này hoạt động thật, không phải luôn trả "OK" hình thức.

---

## Bước 11 — Đóng gói Demo: Streamlit

**File:** [app.py](app.py)

### `@st.cache_resource` — vì sao cần thiết

```python
@st.cache_resource(show_spinner="...")
def load_resources():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    embed_model = get_embed_model()
    llm = get_llm()
    return collection, embed_model, llm
```

Đặc điểm cốt lõi cần hiểu về Streamlit: **toàn bộ script chạy lại từ đầu mỗi khi người
dùng tương tác** (gõ câu hỏi, bấm nút...) — nếu không cache, model embedding (~2GB, load
mất vài giây) và kết nối Chroma sẽ bị khởi tạo lại mỗi lần, cực kỳ lãng phí. `@st.cache_resource`
là decorator của Streamlit dành riêng cho các đối tượng "tốn tài nguyên, dùng chung, không
đổi giữa các lần tương tác" (model, kết nối DB...) — phân biệt với `@st.cache_data` (dành
cho dữ liệu có thể serialize, negiate theo giá trị input).

### Ghép toàn bộ pipeline đã xây thành 1 lệnh gọi duy nhất

```python
def retrieve_fn(collection, embed_model, question, k):
    return hybrid_retrieve_and_rerank(collection, embed_model, question, k)

result = answer_with_verification(collection, embed_model, llm, question, k=TOP_K, retrieve_fn=retrieve_fn)
```

Đây là điểm thể hiện rõ nhất giá trị của việc thiết kế mỗi hàm ở các bước trước với tham số
`retrieve_fn` cắm được (dependency injection đơn giản): `app.py` không cần biết chi tiết
Hybrid Search/Reranker hoạt động thế nào, chỉ cần truyền đúng hàm vào
`answer_with_verification()` (bước 10d) — hàm này tự gọi xuống `answer_question()` (bước
8) → `build_prompt()` (bước 7), toàn bộ 8 bước kỹ thuật trước đó được ghép lại chỉ bằng
2 dòng code. Đây là lợi ích thực tế của việc tách hàm theo tầng rõ ràng ngay từ đầu, thay
vì viết 1 script lớn làm mọi việc.

---

## Bảng tổng hợp khái niệm nền tảng (tra cứu nhanh)

| Khái niệm | Xuất hiện ở bước | Ý nghĩa cốt lõi |
|---|---|---|
| Content-addressed caching (hash MD5/SHA256) | 2, 3, 4 | Cache theo nội dung, không theo tên/vị trí — đổi tên không mất cache |
| Cosine similarity | 3, 4, 6 | Đo góc giữa 2 vector; với vector đã chuẩn hoá, bằng tích vô hướng |
| Adaptive threshold (percentile) | 3 | Ngưỡng tính tương đối theo phân phối cục bộ, không phải số cố định |
| Bi-encoder vs Cross-encoder | 4/6 vs 10b | Encode riêng (nhanh, cache được) vs encode cặp cùng lúc (chậm, chính xác hơn) |
| HNSW / Approximate Nearest Neighbor | 4 | Tìm top-k gần nhất không cần so sánh tuần tự toàn bộ dữ liệu |
| Dense vs Sparse retrieval | 6 vs 10a | Vector đặc (ngữ nghĩa) vs vector thưa (khớp từ khoá chính xác) |
| Recall / Precision / MRR | 6, 9 | Tìm ra chưa? / có bị nhiễu không? / xếp hạng cao hay thấp? |
| Reciprocal Rank Fusion (RRF) | 10a, 10c | Hợp nhất nhiều danh sách xếp hạng khác thang điểm, dựa trên rank |
| LLM-as-judge | 9, 10d | Dùng LLM chấm điểm ngữ nghĩa thay vì so khớp từ thô |
| Grounding prompt / anti-hallucination | 7 | Ép model giới hạn phạm vi trả lời vào context, cho phép từ chối |
| Temperature | 8 | Độ "nhọn" của phân phối xác suất chọn token — thấp = bám sát, ít sáng tạo |
| Self-consistency verification | 10d | Gọi lại LLM để tự kiểm tra đầu ra của chính nó trước khi trả về |
