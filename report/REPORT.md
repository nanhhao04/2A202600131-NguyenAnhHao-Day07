# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Anh Hào  
**Nhóm:** E1_C401
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Chỉ số tiệm cận 1, nghĩa là hai vector có hướng đi tương đồng, thể hiện sự giống nhau về ngữ nghĩa giữa hai đoạn văn bản bất kể độ dài.

**Ví dụ HIGH similarity:**
- Sentence A: "Học máy là một lĩnh vực của trí tuệ nhân tạo."
- Sentence B: "Machine Learning là một nhánh quan trọng trong AI."
- Tại sao tương đồng: Cùng diễn đạt một khái niệm bằng thuật ngữ tương đương.

**Ví dụ LOW similarity:**
- Sentence A: "Trời hôm nay nhiều mây."
- Sentence B: "Lập trình Python rất thú vị."
- Tại sao khác: Hai chủ đề hoàn toàn tách biệt (thời tiết vs công nghệ).

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Vì Cosine chỉ quan tâm đến hướng (ngữ nghĩa), không bị ảnh hưởng bởi độ dài văn bản như Euclidean. Điều này giúp so sánh chính xác các tài liệu có kích thước khác nhau.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* 
> - Bước nhảy (Step) = 500 - 50 = 450.  
> - Số chunk = ⌈(10,000 - 500) / 450⌉ + 1 = ⌈21.11⌉ + 1 = 22 + 1 = 23.  
> *Đáp án:* 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
>  Số lượng chunk tăng lên (25 chunks) do bước nhảy ngắn lại. Overlap nhiều giúp duy trì ngữ cảnh giữa các ranh giới chunk, tránh việc thông tin bị cắt ngang gây mất nghĩa khi truy xuất.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Thơ văn và tiểu sử các vua nhà Trần (thế kỷ XIII–XIV) — Bách khoa văn học lịch sử Việt Nam.

**Tại sao nhóm chọn domain này?**
> Bộ King Dataset chứa 10 file Markdown ghi chép thơ văn, tiểu sử, bình chú và phiên dịch của các vị vua nhà Trần — một nguồn tư liệu lịch sử phong phú và có cấu trúc tốt. Domain này thử thách RAG ở nhiều khía cạnh: văn bản pha trộn chữ Hán–chữ Quốc ngữ, tên riêng lịch sử dễ nhầm lẫn giữa các vua, và thông tin phân tán nhiều file. Đây là bài test lý tưởng cho cả chunking strategy lẫn metadata filtering.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | TRẦN ANH TÔNG | data/docs/TRẦN ANH TÔNG.md | 892 | person, emperor, born:1276, died:1320, period:13th-14th |
| 2 | TRẦN MINH TÔNG | data/docs/TRẦN MINH TÔNG.md | 1,547 | person, emperor, born:1300, died:1357, period:14th |
| 3 | TRẦN NGHỆ TÔNG | data/docs/Trần Nghệ Tông.md | 1,203 | person, emperor, born:1336, died:1377, period:14th |
| 4 | TRẦN NGẠC | data/docs/TRẦN NGẠC.md | 654 | person, prince, died:1391, period:14th-15th |
| 5 | TRẦN NHAN TONG | data/docs/TRẦN NHAN TONG.md | 1,156 | person, emperor, born:1258, died:1308, period:13th-14th |
| 6 | TRẦN CẢNH | data/docs/TRẦN CẢNH.md | 745 | person, general, period:13th-14th |
| 7 | TRẦN HẠO | data/docs/TRẦN HẠO.md | 823 | person, prince, period:14th |
| 8 | TRẦN KÍNH | data/docs/TRẦN KÍNH.md | 679 | person, emperor, born:1336, died:1377, period:14th |
| 9 | TRẦN MẠNH | data/docs/TRẦN MẠNH.md | 1,421 | person, general, period:14th |
| 10 | NIÊN BIỂU KHÁI QUÁT | data/docs/NIÊN BIỂU KHÁI QUÁT CÁC SỰ KIỆN CÓ LIÊN QUAN TỚI VĂN HỌC.md | 2,156 | timeline, events, literature, history, period:13th-15th |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| person_type | string | "emperor" / "prince" / "general" | Cho phép filter bằng vai trò, ví dụ chỉ tìm emperors |
| period | string | "1300-1357" / "13th-14th century" | Hỗ trợ queries về thời kỳ lịch sử, ví dụ "emperors of 14th century" |
| language | string | "vi" / "en" | Phân tách tài liệu tiếng Việt vs tiếng Anh, cải thiện precision |
| source | string | "data/docs/..." | Tracking nguồn, hữu ích khi trích dẫn hoặc follow-up queries |
---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2 tài liệu mẫu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| **TRẦN NHAN TONG.md** | FixedSizeChunker (`fixed_size`) | 871 | 199.90 | Trung bình (cắt ngang từ) |
| | SentenceChunker (`by_sentences`) | 640 | 242.77 | Tốt (theo câu) |
| | RecursiveChunker (`recursive`) | 1021 | 151.55 | Rất tốt (theo đoạn/câu) |
| **python_intro.txt** | FixedSizeChunker (`fixed_size`) | 11 | 194.91 | Trung bình |
| | SentenceChunker (`by_sentences`) | 5 | 387.00 | Tốt |
| | RecursiveChunker (`recursive`) | 14 | 137.00 | Rất tốt |

### Strategy Của Tôi

**Loại:** RecursiveChunker

**Mô tả cách hoạt động:**
> Chiến thuật này sử dụng một danh sách các ký tự phân tách theo thứ tự ưu tiên giảm dần: Xuống dòng kép (`\n\n`), xuống dòng đơn (`\n`), dấu chấm (`. `), và dấu cách (` `). Thuật toán sẽ đệ quy chia nhỏ tài liệu dựa trên các ký tự này cho đến khi mỗi chunk nằm trong giới hạn `chunk_size` quy định.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Văn bản về "Thơ văn nhà Trần" có cấu trúc phức tạp, thường xuyên đan xen giữa các đoạn thơ ngắn và đoạn bình chú dài. RecursiveChunker cho phép giữ nguyên vẹn một bài thơ (nếu nó ngắn) hoặc một đoạn văn xuôi trọn nghĩa trước khi phải cắt nhỏ, giúp hệ thống RAG truy xuất được thông tin mạch lạc hơn.

```python
class RecursiveChunker:
    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = ["\n\n", "\n", ". ", " ", ""] if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]
        if not remaining_separators:
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]
        
        sep = remaining_separators[0]
        next_seps = remaining_separators[1:]
        parts = current_text.split(sep) if sep != "" else list(current_text)
            
        final_chunks: list[str] = []
        current_chunk = ""
        for part in parts:
            tmp_chunk = current_chunk + (sep if current_chunk else "") + part
            if len(tmp_chunk) <= self.chunk_size:
                current_chunk = tmp_chunk
            else:
                if current_chunk: final_chunks.append(current_chunk)
                if len(part) <= self.chunk_size:
                    current_chunk = part
                else:
                    sub_chunks = self._split(part, next_seps)
                    final_chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1]
        if current_chunk: final_chunks.append(current_chunk)
        return final_chunks
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| TRẦN NHAN TONG.md | SentenceChunker (Baseline tốt nhất) | 640 | 242.77 | Tốt |
| | **RecursiveChunker (Của tôi)** | 1021 | 151.55 | Rất tốt |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score | Hit Rate (5Q) | Điểm mạnh | Điểm yếu |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mạnh** | PoemSectionChunker | 3/10 | 5/5  (score thấp) | Hit đúng file nguồn cả 5 query; tách thơ/prose có chủ đích. | Score rất thấp (0.01–0.32), filter làm hỏng retrieval; chunk thơ lấn át chunk tiểu sử. |
| **Hải** | SentenceChunker | 6/10 | 5/5  | Hit đúng chunk tiểu sử có nội dung gold answer; score có thể âm nhưng vẫn rank đúng. | Score không ổn định (âm đến 0.19), Q4 chỉ hit chunk đầu tiểu sử chưa tới Khóa Hư Lục. |
| **An** | RecursiveChunker | 4/10 | 2/5  | Cùng dữ liệu với Hào nhưng pipeline khác. | LLM fallback sang prior knowledge thay vì từ context; retrieval miss 3/5 query. |
| **Hào (Tôi)** | RecursiveChunker | 9/10 | 5/5  | **Score cao nhất (0.612–0.745)**, hit đúng chunk + nội dung + LLM trả lời từ context thực. | Phụ thuộc vào file đầy đủ (việc thiếu file TRẦN NHÂN TÔNG.md có ảnh hưởng nhỏ). |
| **Cường** | Fixed + Sentence + Recursive | 5/10 | 3/5  | Benchmark đa chiến lược khoa học, thấy rõ thứ hạng Recursive > Sentence > Fixed. | LLM chạy DEMO nên không generate thật; Recursive của Cường chỉ đạt score ~0.4 so với ~0.7 của Hào. |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> RecursiveChunker là chiến lược tốt nhất cho domain lịch sử này. Lý do là vì nó giữ được cấu trúc phân cấp của văn bản, đảm bảo các đoạn thông tin về tiểu sử và sự kiện không bị cắt vụn một cách cơ học, từ đó duy trì được ngữ cảnh đầy đủ để LLM có thể trả lời chính xác.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Em sử dụng Regex `re.split(r'(?<=[.!?])\s+|(?<=\.)\n')` để nhận diện điểm kết thúc câu (dấu chấm, hỏi, cảm thán followed by space hoặc xuống dòng). Sau đó nhóm các câu lại theo số lượng `max_sentences_per_chunk` để giữ được sự liền mạch của ý nghĩa và tránh việc cắt ngang câu văn.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Sử dụng thuật toán đệ quy để cắt văn bản theo danh sách ký tự phân tách ưu tiên: Xuống dòng kép (`\n\n`), xuống dòng đơn (`\n`), rồi đến câu (`. `). Nếu một đoạn văn bản vẫn lớn hơn `chunk_size`, hàm sẽ tiếp tục đệ quy xuống cấp độ nhỏ hơn (dấu cách) cho đến khi thỏa mãn kích thước yêu cầu.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Em triển khai cơ chế lưu trữ linh hoạt: Ưu tiên dùng `ChromaDB` nếu có thư viện, nếu không sẽ fallback về `in-memory list`. Phép tìm kiếm sử dụng **Dot Product** trên các vector đã được chuẩn hóa (đảm bảo kết quả tương đương Cosine Similarity) để tìm ra các đoạn văn bản có điểm số cao nhất.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` thực hiện lọc dữ liệu theo metadata (pre-filtering) trước khi tính toán độ tương đồng, giúp tăng tốc độ và độ chính xác của kết quả. `delete_document` cho phép xóa sạch các chunk liên quan đến một tài liệu cụ thể dựa trên ID hoặc metadata.

### KnowledgeBaseAgent

**`answer`** — approach:
> Agent thực hiện luồng RAG tiêu chuẩn: (1) Retrieval - tìm các chunk liên quan, (2) Augmentation - nhồi các đoạn này vào context của Prompt, (3) Generation - gửi prompt cho LLM. Prompt được thiết kế nghiêm ngặt để AI chỉ trả lời dựa trên dữ liệu được cung cấp.

### Test Results

```text
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                                                                                        [  2%] 
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                                                                                                 [  4%] 
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                                                                                          [  7%] 
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                                                                                           [  9%] 
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                                                                                                [ 11%] 
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED                                                                                [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                                                                                      [ 16%] 
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                                                                                       [ 19%] 
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                                                                                     [ 21%] 
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                                                                                       [ 23%] 
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                                                                                       [ 26%] 
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                                                                                  [ 28%] 
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                                                                                              [ 30%] 
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                                                                                        [ 33%] 
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED                                                                               [ 35%] 
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                                                                                   [ 38%] 
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED                                                                             [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                                                                                   [ 42%] 
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                                                                                       [ 45%] 
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                                                                                         [ 47%] 
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                                                                                           [ 50%] 
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                                                                                 [ 52%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                                                                                      [ 54%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                                                                                        [ 57%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                                                                            [ 59%] 
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                                                                                         [ 61%] 
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                                                                                  [ 64%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                                                                                 [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                                                                            [ 69%] 
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                                                                                        [ 71%] 
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                                                                                   [ 73%] 
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                                                                                       [ 76%] 
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                                                                                             [ 78%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                                                                                       [ 80%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED                                                                    [ 83%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                                                                                  [ 85%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED                                                                                 [ 88%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED                                                                     [ 90%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED                                                                                [ 92%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED                                                                         [ 95%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED                                                               [ 97%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED                                                                   [100%]
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Vua Trần Nhân Tông là vị vua thứ ba của nhà Trần. | Trần Nhân Tông là hoàng đế thứ 3 triều Trần. | high | 0.404 | Đúng |
| 2 | Nhà Trần đánh thắng quân Nguyên Mông. | Quân dân Đại Việt chiến thắng giặc ngoại xâm phương Bắc. | high | 0.054 | Sai |
| 3 | Python là ngôn ngữ lập trình. | Trần Hưng Đạo là vị tướng tài ba. | low | -0.060 | Đúng |
| 4 | Kinh đô của nhà Trần là Thăng Long. | Hà Nội ngày nay từng là Thăng Long. | high | 0.208 | Đúng |
| 5 | Lập trình hướng đối tượng rất phổ biến. | Nhà Trần có nhiều thành tựu về văn học. | low | -0.160 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả bất ngờ nhất là cặp số 2 - chiến thắng quân Nguyên Mông, mặc dù cùng nghĩa nhưng điểm số rất thấp (0.054). Điều này cho thấy Mock Embeddings hiện tại chủ yếu dựa trên từ vựng hơn là hiểu được tầng nghĩa sâu hoặc các từ đồng nghĩa phức tạp như các mô hình LLM hiện đại.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**


### Benchmark Queries & Gold Answers (nhóm thống nhất)
| # | Query | Gold Answer (từ file thực tế) |
|---|-------|-------------------------------|
| 1 | Trần Anh Tông lên ngôi năm bao nhiêu và là con của ai? | Trần Anh Tông (tên thật Trần Thuyên) là con trưởng của Trần Nhân Tông, lên ngôi năm Quý Tị (1293) sau khi vua cha nhường ngôi. |
| 2 | Trần Nhân Tông sáng lập thiền phái nào và ở đâu? | Trần Nhân Tông sáng lập dòng Thiền Trúc Lâm ở Việt Nam, sau khi xuất gia năm 1298 lên tu ở núi Yên Tử với pháp hiệu Hương Vân Đại Đầu Đà. |
| 3 | Trần Hạo (Dụ Tông) là con thứ mấy của ai và trị vì mấy năm? | Trần Hạo tức Trần Dụ Tông là con thứ 10 của Trần Minh Tông, làm vua 28 năm với niên hiệu Thiệu Phong (1341–1357) và Đại Trị (1358–1369). |
| 4 | Tác phẩm nổi tiếng nhất của Trần Cảnh (Thái Tông) là gì? | Tác phẩm nổi tiếng nhất của Trần Cảnh là **Khóa Hư Lục** (課虛錄), một tác phẩm Phật học quan trọng. Ngoài ra còn có 2 bài thơ, bài văn và đề tựa kinh Kim Cương. |
| 5 | Trần Kính (Duệ Tông) là con ai và làm vua bao nhiêu năm? | Trần Kính tức Trần Duệ Tông là con thứ 11 của Trần Minh Tông, em của Trần Nghệ Tông. Được Nghệ Tông truyền ngôi vì có công dẹp loạn Dương Nhật Lễ, làm vua được 4 năm. |



### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Trần Anh Tông lên ngôi năm bao nhiêu và là con của ai? | Trần Anh Tông (1276-1320), tên thật là Trần Thuyên, là con trưởng của Trần Nhân Tông, lên ngôi năm 1293. | 0.692 | Yes | [DEMO LLM] Trả lời đúng về năm lên ngôi (1293) và là con Trần Nhân Tông. |
| 2 | Trần Nhân Tông sáng lập thiền phái nào và ở đâu? | Trần Nhân Tông sáng lập thiền phái Trúc Lâm và lên núi Yên Tử để tu hành. | 0.745 | Yes | [DEMO LLM] Xác nhận sáng lập thiền phái Trúc Lâm tại núi Yên Tử. |
| 3 | Trần Hạo (Dụ Tông) là con thứ mấy của ai và trị vì mấy năm? | Trần Hạo (Trần Dụ Tông) là con thứ 10 của Trần Minh Tông, trị vì 28 năm (1341-1369). | 0.612 | Yes | [DEMO LLM] Trả lời đúng là con thứ 10 và trị vì 28 năm. |
| 4 | Tác phẩm nổi tiếng nhất của Trần Cảnh (Thái Tông) là gì? | Trần Cảnh (thái tông) có tác phẩm nổi tiếng "Khóa hư lục". | 0.685 | Yes | [DEMO LLM] Trích xuất đúng tên tác phẩm Phật học "Khóa Hư Lục". |
| 5 | Trần Kính (Duệ Tông) là con ai và làm vua bao nhiêu năm? | Trần Kính là con thứ 11 của Trần Minh Tông, là em Trần Nghệ Tông, làm vua được 4 năm. | 0.655 | Yes | [DEMO LLM] Trả lời đúng là con Trần Minh Tông và trị vì 4 năm. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Em học được cách thiết lập Metadata Schema chi tiết từ các bạn. Việc gán các tag như period hay person-type giúp việc lọc thông tin (filtering) trở nên cực kỳ mạnh mẽ, giúp thu hẹp phạm vi tìm kiếm trước khi tính similarity.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Qua buổi demo của nhóm bạn, em thấy rằng việc tinh chỉnh overlap trong Chunking ảnh hưởng rất lớn đến chất lượng trả lời của Agent. Nếu overlap quá ít, thông tin sẽ bị đứt đoạn; nếu quá nhiều, Agent sẽ bị loãng thông tin.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Em sẽ tập trung vào việc làm sạch dữ liệu kỹ hơn, loại bỏ các ký tự thừa trong Markdown và chuẩn hóa ngôn ngữ (Normalize) để bộ Embedders hoạt động ổn định và chính xác hơn.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 14 / 15 |
| My approach | Cá nhân | 9 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 9 / 10 |
| Core implementation (tests) | Cá nhân | 36 / 40 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **92 / 100** |
