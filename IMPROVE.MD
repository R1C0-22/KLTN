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

Prediction §3.3 (đã chỉnh trong code)
- Parse số chỉ mục 1..|Oq| trước khi khớp substring; tránh chọn nhầm khi LLM sinh đoạn dài.
- Giới hạn logprob path khi |Oq| rất lớn: env ``MAX_LOGPROB_CANDIDATES`` (mặc định 512), còn lại dùng generate + parse số.