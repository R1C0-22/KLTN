# DOI CHIEU 3 NGUON (THEO UU TIEN)

Tai lieu nay duoc doi chieu theo dung thu tu uu tien:
1) `Document/De-Tai-2.pdf` (paper AnRe, ACL 2025) - NEN TANG COT LOI.
2) `Code/IMPROVE.MD` - GOI CAI TIEN PHUC VU KHOA LUAN.
3) `https://github.com/usc-isi-i2/isi-tkg-icl` - NGUON THAM KHAO DE FIX BUG/TOI UU MOT PHAN.

Ket luan dinh vi:
- Code hien tai CHU YEU duoc xay tren (1) + (2).
- (3) duoc dung o vai diem ky thuat de on dinh he thong, KHONG phai nguon cot loi cua de tai.

---

## 1) Cot loi tu (1) `De-Tai-2.pdf` da duoc ap dung

### A. Kien truc AnRe giu dung huong paper
- Semantic-driven historical clustering.
- Dual history extraction (short-term + long-term PDC/DTF).
- Analogical replay de tao reasoning examples.
- Final prediction theo candidate set va chon ket qua xep hang cao nhat.

### B. Hyperparameter va thiet ke thuc nghiem dung tinh than paper
- Co truc sweep cho `L`, `l`, `alpha` (phu hop phan 6.1 cua paper).
- Co ablation mini: w/o long-term, w/o short-term, w/o analogical.
- Co ghi nhan trade-off candidate set 1-hop/2-hop (phan tich giong muc 6.2 paper).

### C. Prompting theo 3 giai doan paper
- Prompt cho PDC (chon su kien huu ich nhat).
- Prompt cho analysis process (APC / analogical explanation).
- Prompt cho object prediction (OEP, tra ve chi so candidate).

=> Ve hoc thuat, code dang bam sat cot song cua AnRe (nguon 1).

## 2) Phan tu (2) `IMPROVE.MD` da duoc dua vao code

### A. Goi cai tien "de lam khoa luan"
- Adaptive `Oq/O2q` (mo rong candidate dong thay vi co dinh).
- LLM caching cho scorer/generator/predictor.
- Runtime benchmark de bao cao giam thoi gian do cache.
- Hyperparameter sweep + ablation de phuc vu bao cao/bao ve.

### B. Cac sua bug-trong-tam da phan anh trong code
- Parse prediction theo chi so `1..|Oq|` uu tien truoc substring.
- Gioi han duong logprob khi candidate qua lon (`MAX_LOGPROB_CANDIDATES`).
- HF logprob fix loi va cham token dau (`1` vs `10` vs `100`).

=> Day la lop cai tien de bai toan chay on dinh va co du so lieu bao cao.

## 3) Vai tro thuc te cua (3) `isi-tkg-icl`

### A. Da tham khao va ap dung mot phan de on dinh pipeline
- Filtered eval mode `none/static/time-aware`.
- Tu duy runner/eval theo output JSONL.
- Mot so ky thuat map-label + ranking khi infer voi LLM.

### B. Nhung gi CHUA (va KHONG bat buoc) phai dong bo theo repo tham chieu
- Baseline `recency/frequency` theo dung script goc.
- Bo runner CLI y het (`run_hf.py`, `run_openai.py`, `run_rule.py`).
- Full protocol head/tail nhu repo goc ICL 2023.

=> Nguon (3) la bo tham chieu huu ich de sua bug, khong doi vai tro cot loi cua de tai.

---

## 4) Ket luan doi chieu theo uu tien 1 -> 2 -> 3

### Tuyen bo hoc thuat (de dua vao mo ta khoa luan)
- Nen tang phuong phap cua code la AnRe (nguon 1).
- Huong cai tien va muc tieu bao cao den tu `IMPROVE.MD` (nguon 2).
- `isi-tkg-icl` (nguon 3) chi dong vai tro tham khao ky thuat de giam bug va tang do on dinh khi trien khai.

### Cac diem nen nhan manh khi bao ve
- Khung bai toan va module theo paper AnRe duoc giu nguyen tinh than.
- Co bo cai tien thuc dung, giai quyet han che runtime/oom/parse sai.
- Co thuc nghiem bo tro (ablation, sweep, cache benchmark) de chung minh gia tri ky thuat.

## 5) Backlog tiep theo (neu can bo sung ky thuat)

- [ ] Them baseline recency/frequency de doi chieu tham khao (khong lam thay doi cot loi de tai).
- [ ] Chuan hoa them 1 runner CLI de tai lap thuc nghiem nhanh.
- [ ] Chot bo bang ket qua theo truc: base AnRe / +improve.md / +tham khao ky thuat tu repo 3.
