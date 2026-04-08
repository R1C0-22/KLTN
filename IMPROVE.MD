Gói cải tiến cụ thể (đủ làm luận văn)
Adaptive Oq/O2q (candidate set động)

Mặc định dùng 1-hop để tiết kiệm.
Nếu confidence thấp hoặc candidate quá ít thì tự mở rộng 2-hop.
So sánh với baseline luôn-1hop và luôn-2hop.
Caching cho LLM_SCORER và LLM_GENERATOR

Cache theo key (query, history_hash, model_name) để tránh gọi lại.
Báo cáo giảm thời gian suy luận bao nhiêu %.
Hyperparameter sweep nhỏ nhưng chuẩn

test L (history length), l (short-term), alpha (DTF).
Paper đã cho trend, bạn chứng minh lại trên setting của bạn.
Báo cáo ablation mini

w/o long-term, w/o short-term, w/o analogical (bạn đã có cấu trúc để làm).
Đây là phần hội đồng rất hay hỏi.

Colab / runtime (DTF cost)
- Mỗi **ngày** (timestep) trong long-term pool gọi một lần PDC (`compute_scores_with_llm`). Entity “đông” có thể cần hàng trăm ngày → một query vài chục phút trên T4. Biến môi trường ``MAX_DTF_TIMESTEP_ITERATIONS`` (mặc định trong ``colab_setup.setup()``: 40; ``0`` = không giới hạn, chạy đúng paper nhưng rất chậm).

Prediction §3.3 (đã chỉnh trong code)
- Parse số chỉ mục 1..|Oq| trước khi khớp substring; ưu tiên dòng cuối / “answer is …” / ``[k]``; tránh chọn nhầm khi LLM sinh đoạn dài nhắc nhiều entity.
- Giới hạn logprob path khi |Oq| rất lớn: env ``MAX_LOGPROB_CANDIDATES`` (mặc định 512), còn lại dùng generate + parse số.
- HF local: ``_logprobs_huggingface`` mặc định cộng logprob theo **toàn bộ** token của nhãn số (``"1"``…``"N"``), tránh lỗi trùng token đầu giữa ``1`` / ``10`` / ``100``. Đường cũ (chỉ token đầu): ``HF_LOGPROB_FAST=1``.