# Tóm tắt đoạn chat hiện tại (NCKH - TKG Pipeline)

## 1) Mục tiêu đã triển khai

Đã xây pipeline cho Temporal Knowledge Graph Forecasting (theo hướng AnRe) gồm:

- Tiền xử lý quadruple -> câu tự nhiên
- Clustering thực thể bằng sentence-transformers + KMeans + silhouette
- Truy xuất lịch sử theo entity/relation
- Chọn short-term history
- Lọc long-term history theo điểm LLM + dynamic threshold
- Sinh analogical reasoning bằng LLM
- Final inference để dự đoán object bị thiếu
- Unified LLM interface (OpenAI/Groq) + adapter Ollama local

---

## 2) Các module và hàm đã có

## `preprocessing/`

- `load_dataset(path, splits=None)`
  - Hỗ trợ file/directory dataset (đọc được `train/test/valid` không cần extension)
- `verbalize_event(s, r, o, t)`
  - Chuyển quadruple -> câu tự nhiên
- `build_corpus(data)`
  - Sinh danh sách câu từ danh sách quadruple

File chính: `preprocessing/verbalize.py`

---

## `clustering/`

- `embed_entities(entity_list)`
  - Encode entity names bằng sentence-transformers
- `find_optimal_k(embeddings)`
  - Chọn K tối ưu bằng silhouette score
- `run_kmeans(embeddings, k)`
  - Chạy KMeans, trả nhãn cluster
- Bổ sung:
  - `cluster_entities(...)` (pipeline đầy đủ)
  - `extract_entities(quads)`

File chính: `clustering/entity_cluster.py`

---

## `history/`

- `get_entity_history(entity, data)`
  - Lấy toàn bộ sự kiện liên quan entity (entity ở subject hoặc object)
  - Có sắp xếp theo timestamp
- `filter_by_relation(history, relation)`
  - Lọc sự kiện cùng relation

File chính: `history/history_retrieval.py`

---

## `short_term/`

- `get_short_term(history, l=20)`
  - Sort theo timestamp
  - Lấy `l` sự kiện gần nhất (mặc định 20)

File chính: `short_term/short_term.py`

---

## `long_term/`

- `compute_scores_with_llm(history)`
  - Đọc prompt từ `prompts/filter_prompt.txt`
  - Gọi hàm scorer qua env `LLM_SCORER="module:function"`
- `dynamic_threshold(F, delta_t, T, alpha)`
  - Công thức paper:
  - `c_j = 1/F + (1 - 1/F) * (delta_t / T)^alpha`
- `filter_long_term(history, scores)`
  - Group theo timestamp
  - Tính softmax theo từng group
  - Giữ event nếu `p(hl) >= c_j`

Files:
- `long_term/long_term_filter.py`
- `long_term/dummy_scorer.py` (test local)
- `long_term/test_long_term.py`

---

## `analogical/`

- `generate_analogical_reasoning(event, similar_events)`
  - Đọc prompt từ `prompts/reasoning_prompt.txt`
  - Gọi generator qua env `LLM_GENERATOR="module:function"`

Files:
- `analogical/analogical_reasoning.py`
- `analogical/dummy_generator.py`
- `analogical/test_analogical.py`
- `analogical/run_dummy_on_real_data.py`

---

## `inference/`

- `predict_next_object(query_event)`
  - Kết hợp:
    - short-term history
    - long-term filtered history
    - analogical reasoning
  - Xây final prompt
  - Gọi LLM predictor để trả object dự đoán

Files:
- `inference/final_prediction.py`
- `inference/dummy_predictor.py`
- `inference/test_prediction_dummy.py`

---

## `llm/`

- Unified API:
  - `call_llm(prompt)` trong `llm/unified.py`
  - Hỗ trợ `openai` / `groq` qua `LLM_PROVIDER`
- Ollama local:
  - `llm/ollama_adapter.py` có:
    - `generate_fn(prompt)`
    - `score_fn(prompt, events)`
    - `predict_fn(prompt)`
  - Default model đã chỉnh về:
    - `OLLAMA_MODEL = "llama3.2:1b"`

---

## 3) Prompt templates

- `prompts/filter_prompt.txt` (long-term scoring)
- `prompts/reasoning_prompt.txt` (analogical reasoning)
- `prompts/prediction_prompt.txt` (final prediction)

---

## 4) Cấu hình env quan trọng

### Dùng local Ollama (`llama3.2:1b`) cho toàn pipeline

```powershell
$env:LLM_GENERATOR = "llm.ollama_adapter:generate_fn"
$env:LLM_SCORER    = "llm.ollama_adapter:score_fn"
$env:LLM_PREDICTOR = "llm.ollama_adapter:predict_fn"
$env:OLLAMA_MODEL  = "llama3.2:1b"
$env:TKG_DATA_DIR  = "data/ICEWS05-15"
```

Chạy inference (từ project root, thư mục này chứa các folder `inference/`, `long_term/`, v.v.):

```powershell
python -m inference.final_prediction
```

---

## 5) Dependencies

`requirements.txt` đã có:

- `sentence-transformers`
- `scikit-learn`
- `numpy`

Lưu ý: nếu dùng OpenAI/Groq/unified interface thì cần thêm thư viện tương ứng nếu bạn muốn gọi bằng SDK. Bản hiện tại trong `unified.py` dùng HTTP chuẩn (`urllib`) nên không bắt buộc SDK.

---

## 6) Kết quả chạy gần nhất

- Đã chạy:
  - `python -m Code.inference.final_prediction`
- Output:
  - Dự đoán ra `China`
- Có warning `RuntimeWarning` từ `runpy` nhưng không làm hỏng kết quả.

---

## 7) Lưu ý kỹ thuật / TODO gợi ý cho chat mới

- Chuẩn hóa kiểu event:
  - Có thể dùng thống nhất `Code/common/events.py` nếu bạn đang refactor
- Cải thiện `predict_next_object`:
  - Thêm candidate selection tốt hơn (top-k theo ngữ cảnh)
  - Validate output thuộc candidate set chặt hơn
- Tối ưu long-term:
  - Thử alpha khác 2.75 theo dataset
  - Caching LLM scores để chạy nhanh hơn
- Thêm test tự động:
  - Unit tests cho từng module
  - Integration test end-to-end trên tập nhỏ

---

## 8) Entry points test nhanh

Chạy từ project root:

```powershell
python analogical/test_analogical.py
python long_term/test_long_term.py
python inference/test_prediction_dummy.py
python -m inference.final_prediction
```

