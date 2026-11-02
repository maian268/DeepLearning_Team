"""Dựng báo cáo Word (.docx) + PDF từ log/hình trong logs/ và figures/.
  python build_report.py            # tạo report/BaoCao_BTL01.docx và .pdf
  python build_report.py --no-pdf   # chỉ tạo .docx
Chạy SAU khi đã chạy thực nghiệm (run_all.py). Nếu thiếu số liệu, mục tương ứng được đánh dấu nền VÀNG.
Cần điền tay (nền vàng): tên nhóm, thành viên, nhiệm vụ, tỷ lệ đóng góp; nên đọc lại và chỉnh phần nhận xét."""
import argparse
import shutil
import subprocess
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Cm
from src import config as cfg
from src import analysis as an

FONT = "Times New Roman"


class Report:
    def __init__(self):
        self.doc = Document()
        s = self.doc.sections[0]
        s.page_width, s.page_height = Cm(21), Cm(29.7)
        s.left_margin = s.right_margin = Cm(2.2); s.top_margin = s.bottom_margin = Cm(2.0)
        st = self.doc.styles["Normal"]
        st.font.name, st.font.size = FONT, Pt(12)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        st.paragraph_format.space_after = Pt(6); st.paragraph_format.line_spacing = 1.15
        for name, size in [("Heading 1", 16), ("Heading 2", 13.5), ("Heading 3", 12)]:
            h = self.doc.styles[name]; h.font.name, h.font.size, h.font.bold = FONT, Pt(size), True
            h.element.rPr.rFonts.set(qn("w:eastAsia"), FONT); h.font.color.rgb = None
        self.nf = self.nt = 0
        self._footer()

    def _footer(self):
        p = self.doc.sections[0].footer.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run()
        for t, txt in [("begin", None), (None, "PAGE"), ("end", None)]:
            if t:
                e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), t)
            else:
                e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt
            r._r.append(e)

    # ---- khối nội dung ----
    def h(self, text, lvl=1): self.doc.add_heading(text, level=lvl)

    def p(self, text="", bold=False, italic=False, align=None, size=None, hl=False, indent=True):
        para = self.doc.add_paragraph()
        if align == "center": para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif align == "justify" or align is None: para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if indent and align is None: para.paragraph_format.first_line_indent = Cm(0.8)
        r = para.add_run(text); r.bold, r.italic = bold, italic
        if size: r.font.size = Pt(size)
        if hl: r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        return para

    def todo(self, text): self.p("[CẦN BỔ SUNG] " + text, hl=True, indent=False)

    def bullets(self, items):
        for it in items:
            para = self.doc.add_paragraph(style="List Bullet"); para.add_run(it); para.paragraph_format.space_after = Pt(2)

    def formula(self, text):
        para = self.doc.add_paragraph(); para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = para.add_run(text); r.italic = True; r.font.name = "Cambria Math"

    def code(self, text):
        para = self.doc.add_paragraph(); para.paragraph_format.line_spacing = 1.0
        r = para.add_run(text); r.font.name = "Consolas"; r.font.size = Pt(8.5)
        r._r.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
        sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:fill"), "F2F2F2"); para._p.get_or_add_pPr().append(sh)

    def figure(self, name, caption, width=14.5):
        path = cfg.FIG_DIR / name
        self.nf += 1
        if path.exists():
            para = self.doc.add_paragraph(); para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            para.add_run().add_picture(str(path), width=Cm(width)); para.paragraph_format.keep_with_next = True
        else:
            self.todo(f"Thiếu hình {name} (chạy lại thực nghiệm).")
        self.p(f"Hình {self.nf}. {caption}", italic=True, align="center", size=10.5)

    def table(self, header, rows, caption, widths=None, size=9):
        self.nt += 1
        self.p(f"Bảng {self.nt}. {caption}", italic=True, align="center", size=10.5).paragraph_format.keep_with_next = True
        t = self.doc.add_table(rows=1, cols=len(header)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, h in enumerate(header):
            c = t.rows[0].cells[i]; c.text = ""; r = c.paragraphs[0].add_run(str(h)); r.bold = True; r.font.size = Pt(size)
            c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:fill"), "DCE6F2"); c._tc.get_or_add_tcPr().append(sh)
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""; para = cells[i].paragraphs[0]; para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                para.paragraph_format.space_after = Pt(0); para.paragraph_format.line_spacing = 1.0
                txt = str(v)
                if txt.startswith("[CẦN"):
                    r = para.add_run(txt); r.font.highlight_color = WD_COLOR_INDEX.YELLOW
                else:
                    r = para.add_run(txt)
                r.font.size = Pt(size)
        if widths:
            for row in t.rows:
                for i, w in enumerate(widths): row.cells[i].width = Cm(w)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)

    def pagebreak(self): self.doc.add_page_break()


def group_rows(s, digits=2):
    f = lambda c, i: f"{s[c + '_mean'].iloc[i]:.{digits}e}\n±{s[c + '_std'].iloc[i]:.1e}"
    return [[int(s["factor_value"].iloc[i]), int(round(s["n_params_mean"].iloc[i])), f("train_mse", i), f("val_mse", i), f("test_mse", i),
             f("test_rmse", i), f("test_mae", i), f"{s['train_time_s_mean'].iloc[i]:.1f}"] for i in range(len(s))]


def build(no_pdf=False):
    cfg.ensure_dirs()
    D = an.load_all()
    R = Report()
    seeds_txt = ", ".join(map(str, (D["config"] or {}).get("seeds", cfg.SEEDS)))
    quick = bool((D["config"] or {}).get("quick_mode", False))

    # ------------------------------------------------ bìa
    R.p("HỌC SÂU", bold=True, align="center", size=14)
    R.p("BÀI TẬP LỚN 01", bold=True, align="center", size=18)
    R.p("XẤP XỈ HÀM SỐ DÙNG MẠNG PERCEPTRON ĐA TẦNG (MLP) BẰNG PYTORCH", bold=True, align="center", size=16)
    R.p("Khảo sát ảnh hưởng của số mẫu huấn luyện (A), số nơ-ron tầng ẩn (B) và số tầng ẩn (C)", italic=True, align="center")
    R.p("")
    R.p("Nhóm: [CẦN BỔ SUNG: tên nhóm]   |   Lớp / Học kỳ: [CẦN BỔ SUNG]", align="center", hl=True)
    R.table(["Họ và tên", "MSSV", "Nhiệm vụ", "Tỷ lệ đóng góp (%)"],
            [["[CẦN BỔ SUNG]", "[CẦN BỔ SUNG]", "[CẦN BỔ SUNG]", "[CẦN BỔ SUNG]"] for _ in range(3)],
            "Thành viên, nhiệm vụ và tỷ lệ đóng góp (tổng 100%)", widths=[5, 3, 5.5, 3.3], size=10)
    if quick:
        R.p("LƯU Ý: số liệu trong bản này được tạo ở chế độ chạy thử nhanh (--quick), chỉ dùng để kiểm tra; hãy chạy đầy đủ rồi dựng lại báo cáo.", hl=True, indent=False)
    R.pagebreak()

    # ------------------------------------------------ 1. Giới thiệu
    R.h("1. Giới thiệu")
    R.h("1.1. Mục tiêu", 2)
    R.bullets(["Xây dựng và huấn luyện mạng perceptron đa tầng (MLP) cho bài toán hồi quy/xấp xỉ hàm hai biến.",
               "Thực hành toàn bộ quy trình bằng PyTorch: dữ liệu, mô hình, lan truyền ngược, tối ưu và đánh giá.",
               "Thực nghiệm có kiểm soát để phân tích tác động của số mẫu huấn luyện A, số nơ-ron tầng ẩn B và số tầng ẩn C.",
               "Trình bày kết quả bằng báo cáo và slide kèm phân tích."])
    R.h("1.2. Phát biểu bài toán", 2)
    R.p("Cho hàm mật độ Gaussian hai chiều f(x) với trung bình μ = (0, 0)ᵀ và ma trận hiệp phương sai C = [[0,3; −0,5], [−0,5; 2]]. "
        "Mẫu x = (x₁, x₂) được sinh ngẫu nhiên đều trong miền −2 ≤ x₁ ≤ 2, −4 ≤ x₂ ≤ 4 với nhãn y = f(x₁, x₂). "
        "Cần xây dựng mạng MLP có hai đầu vào, một đầu ra ŷ để xấp xỉ y; đánh giá trên tập test độc lập 100 mẫu, "
        "có thêm tập validation riêng dùng để chọn mô hình.")
    R.h("1.3. Nội dung thực hiện", 2)
    R.bullets(["Tự cài đặt hàm Gaussian (đối chiếu với SciPy), sinh và chia dữ liệu train/validation/test độc lập, cố định seed.",
               "Tự xây dựng MLP bằng nn.Module, nn.Linear, hàm kích hoạt tanh; vòng lặp huấn luyện/validation/checkpoint bằng PyTorch.",
               "Ba nhóm thực nghiệm A (200–2000 mẫu), B (1–10 nơ-ron và mở rộng đến 50), C (1–5 tầng ẩn), mỗi cấu hình chạy nhiều seed, báo cáo trung bình ± độ lệch chuẩn.",
               "Xuất log CSV, hình, bảng, mô hình tốt nhất (.pt); mã nguồn tổ chức theo chức năng và có thể chạy lại bằng một lệnh."])

    # ------------------------------------------------ 2. Dữ liệu
    R.h("2. Dữ liệu")
    R.h("2.1. Hàm Gaussian hai chiều", 2)
    R.formula("f(x) = 1 / (2π·√det C) · exp( −½ (x − μ)ᵀ C⁻¹ (x − μ) )")
    R.p("Với C = [[0,3; −0,5], [−0,5; 2]], det C = 0,3·2 − (−0,5)² = 0,35 > 0 và các giá trị riêng đều dương nên C xác định dương. "
        "Giá trị đỉnh của hàm là f(μ) = 1 / (2π·√0,35) ≈ 0,2693. Hệ số tương quan âm làm các đường đồng mức là ellipse nghiêng "
        "(x₁ tăng thì x₂ có xu hướng giảm). Hàm được cài đặt vector hóa bằng NumPy (einsum cho khoảng cách Mahalanobis) "
        "và được kiểm tra khớp với scipy.stats.multivariate_normal.")
    R.h("2.2. Lấy mẫu và chia dữ liệu", 2)
    R.p(f"Dữ liệu được sinh một lần với seed cố định {cfg.DATA_SEED}. Ba tập được sinh từ ba dòng số ngẫu nhiên độc lập "
        f"(numpy SeedSequence.spawn), nên không trùng mẫu và không rò rỉ giữa các tập. Tập train gốc có {cfg.N_POOL} mẫu; "
        f"với mức A, mô hình dùng A mẫu đầu tiên của tập gốc (các tập lồng nhau) để so sánh công bằng. "
        f"Validation ({cfg.N_VAL} mẫu) dùng cho early stopping và chọn checkpoint; test ({cfg.N_TEST} mẫu) chỉ dùng đánh giá cuối.")
    R.table(["Tập", "Số mẫu", "Mục đích", "Nguồn ngẫu nhiên"],
            [["Train gốc", cfg.N_POOL, "Huấn luyện (lấy A mẫu đầu)", "SeedSequence(2024) con #1"],
             ["Validation", cfg.N_VAL, "Early stopping, chọn mô hình", "SeedSequence(2024) con #2"],
             ["Test", cfg.N_TEST, "Đánh giá cuối", "SeedSequence(2024) con #3"]], "Chia dữ liệu", widths=[3, 2.5, 6, 5], size=10)
    R.h("2.3. Thống kê và trực quan hóa", 2)
    if D["stats"] is not None:
        st = D["stats"]
        R.table(["Tập", "Biến", "n", "Trung bình", "Độ lệch chuẩn", "Min", "Max"],
                [[r.tap, r.bien, int(r.n), f"{r['mean']:.4f}", f"{r['std']:.4f}", f"{r['min']:.4f}", f"{r['max']:.4f}"] for _, r in st.iterrows()],
                "Thống kê mô tả của các tập dữ liệu", widths=[2.6, 1.6, 1.5, 2.8, 3, 2.6, 2.6], size=9)
    else:
        R.todo("Chưa có logs/data_statistics.csv (chạy run_data.py).")
    R.figure("fig01_data_scatter3d.png", "Scatter 3D của dữ liệu train/validation/test.", 12)
    R.figure("fig02_true_surface.png", "Bề mặt hàm Gaussian thật và vị trí các mẫu train trên đường đồng mức.", 15)
    R.figure("fig03_distributions.png", "Phân bố x₁, x₂ và y của tập train gốc.", 15)
    nz = an.near_zero_pct()
    R.p("Do x₁ và x₂ được lấy mẫu đều nên histogram của chúng phẳng, còn y tập trung gần 0 với đuôi kéo dài về phía đỉnh"
        + (f": chỉ khoảng {100 - nz:.0f}% mẫu train có y lớn hơn 1% giá trị đỉnh. " if nz is not None else ". ")
        + "Vì vùng đỉnh nhọn chiếm tỉ lệ nhỏ, số mẫu rơi vào vùng này là yếu tố quyết định chất lượng xấp xỉ.")

    # ------------------------------------------------ 3. Mô hình
    R.h("3. Mô hình và cài đặt")
    R.h("3.1. Kiến trúc MLP", 2)
    R.p("Mô hình có hai đầu vào, C tầng ẩn (mỗi tầng nn.Linear theo sau bởi tanh) và một đầu ra tuyến tính. Số tầng ẩn và số nơ-ron mỗi tầng "
        "được truyền qua tham số hidden_sizes của lớp MLP (kế thừa nn.Module), nên cùng một lớp dùng được cho cả ba nhóm thực nghiệm.")
    R.figure("fig_architecture.png", "Sơ đồ MLP chuẩn (2 → 32 → 32 → 1) dùng trong nhóm A; mỗi tầng chỉ vẽ tối đa 6 nơ-ron.", 15)
    R.h("3.2. Công thức lan truyền xuôi và hàm mất mát", 2)
    R.formula("a⁽⁰⁾ = x̃ ;  z⁽ˡ⁾ = W⁽ˡ⁾ a⁽ˡ⁻¹⁾ + b⁽ˡ⁾ ;  a⁽ˡ⁾ = tanh(z⁽ˡ⁾),  l = 1…C ;  ŷ = W⁽ᶜ⁺¹⁾ a⁽ᶜ⁾ + b⁽ᶜ⁺¹⁾")
    R.formula("L = (1/N) Σₙ (ŷₙ − yₙ)²   (nn.MSELoss)")
    R.p("Chuẩn hóa: đầu vào chia cho nửa biên của miền, x̃ = (x₁/2, x₂/4) ∈ [−1, 1]²; nhãn chia cho giá trị đỉnh, ỹ = y / f(μ) ∈ (0, 1]. "
        "Đây là các hằng số suy ra từ miền và từ hàm, không ước lượng từ dữ liệu nên không gây rò rỉ. Mạng học ỹ; mọi chỉ số được tính lại trên thang đo gốc.")
    R.p("Các chỉ số đánh giá: MSE = mean((ŷ−y)²), RMSE = √MSE, MAE = mean|ŷ−y| và hệ số xác định R² = 1 − Σ(ŷ−y)² / Σ(y−ȳ)².")
    R.h("3.3. Quy trình huấn luyện bằng PyTorch", 2)
    R.p("Vòng lặp huấn luyện được tự viết (không dùng thư viện huấn luyện cấp cao). Đoạn mã cốt lõi (src/train.py):")
    R.code("for epoch in range(1, max_epochs + 1):\n"
           "    model.train()\n"
           "    for xb, yb in train_loader:          # DataLoader xáo trộn theo seed\n"
           "        y_hat = model(xb)                 # forward\n"
           "        loss = criterion(y_hat, yb)       # MSELoss\n"
           "        optimizer.zero_grad()             # xóa gradient cũ\n"
           "        loss.backward()                   # lan truyền ngược\n"
           "        optimizer.step()                  # cập nhật trọng số (Adam)\n"
           "    model.eval()\n"
           "    with torch.no_grad():                 # validation, không tính gradient\n"
           "        val_loss = evaluate_loss(model, val_loader, criterion, device)\n"
           "    if val_loss < best_val * (1 - rel_tol):   # checkpoint tốt nhất theo validation\n"
           "        best_state = copy.deepcopy(model.state_dict()); wait = 0\n"
           "    else:\n"
           "        wait += 1                         # early stopping nếu wait >= patience")
    R.bullets(["model.train() / model.eval() và torch.no_grad() được dùng đúng ngữ cảnh: train có cập nhật gradient, validation/test không.",
               "Checkpoint tốt nhất theo validation loss được lưu bằng sao chép state_dict và nạp lại khi kết thúc huấn luyện.",
               "Log loss theo epoch (train/validation) được ghi vào logs/history_*.csv; kết quả từng lần chạy ghi vào logs/results_*_runs.csv.",
               "Seed điều khiển khởi tạo trọng số, thứ tự batch (torch.Generator riêng); chạy CPU, có thể bật GPU bằng --gpu."])
    cfgj = D["config"] or {}
    R.p("Môi trường chạy: Python " + str(cfgj.get("python", "[CẦN BỔ SUNG]")) + ", PyTorch " + str(cfgj.get("torch", "[CẦN BỔ SUNG]"))
        + ", NumPy " + str(cfgj.get("numpy", "[CẦN BỔ SUNG]")) + ".", indent=False)

    # ------------------------------------------------ 4. Thực nghiệm
    R.h("4. Thực nghiệm")
    R.h("4.1. Thiết lập và quy trình đánh giá công bằng", 2)
    T = cfg.TRAIN
    R.table(["Tham số", "Giá trị"],
            [["Seed dữ liệu", cfg.DATA_SEED], ["Seed huấn luyện (khởi tạo + xáo batch)", seeds_txt],
             ["Optimizer / learning rate", f"Adam / {T['lr']}"], ["Batch size", T["batch_size"]],
             ["Số epoch tối đa", T["max_epochs"]], ["Điều kiện dừng", f"Early stopping theo validation, patience {T['patience']}, cải thiện tối thiểu {T['rel_tol'] * 100:.1f}%"],
             ["Hàm kích hoạt / mất mát", "tanh (tầng ẩn), tuyến tính (tầng ra) / MSELoss"],
             ["Nhóm A", f"A = {cfg.A_LIST[0]}…{cfg.A_LIST[-1]} (bước 200), kiến trúc {'-'.join(map(str, cfg.A_HIDDEN))}"],
             ["Nhóm B", f"B = {', '.join(map(str, cfg.B_LIST))} nơ-ron, 1 tầng ẩn, {cfg.N_FIXED} mẫu train"],
             ["Nhóm C", f"C = {', '.join(map(str, cfg.C_LIST))} tầng ẩn, mỗi tầng {cfg.C_WIDTH} nơ-ron, {cfg.N_FIXED} mẫu train"]],
            "Cấu hình chuẩn và thiết lập từng nhóm", widths=[6, 10.5], size=10)
    R.bullets(["Trong từng nhóm chỉ thay đổi yếu tố đang khảo sát; các điều kiện còn lại giữ nguyên.",
               "Cùng một bộ dữ liệu cố định cho mọi cấu hình; validation dùng để dừng sớm và chọn checkpoint; test không dùng để chọn siêu tham số.",
               "Mỗi cấu hình chạy nhiều seed; báo cáo trung bình ± độ lệch chuẩn (std tính theo mẫu, ddof = 1).",
               "Lưu ý: batch size và số epoch tối đa giữ nguyên nên khi A nhỏ số bước cập nhật mỗi epoch ít hơn; early stopping cho phép mỗi cấu hình tự dừng đúng lúc."])

    def result_section(g, title, analyze, extra=""):
        R.h(title, 2)
        s = D[g]
        spec = cfg.group_spec(g)
        if s is None:
            R.todo(f"Chưa có kết quả nhóm {g} (chạy: python run_experiment.py --group {g})."); return
        R.table([spec["short"], "Số tham số", "Train MSE", "Val MSE", "Test MSE", "Test RMSE", "Test MAE", "Thời gian (s)"],
                group_rows(s), f"Kết quả nhóm {g} (trung bình ± độ lệch chuẩn theo seed). Thang đo của MSE là bình phương đơn vị của y.",
                widths=[1.4, 1.7, 2.3, 2.3, 2.3, 2.1, 2.1, 1.7], size=8)
        R.figure(f"fig_{g}_mse.png", f"MSE train/validation/test theo {spec['name'].lower()} (thang log, thanh sai số là độ lệch chuẩn).", 13)
        R.figure(f"fig_{g}_rmse_mae.png", f"RMSE và MAE trên tập test theo {spec['short']}.", 12)
        R.figure(f"fig_{g}_curves.png", f"Đường cong loss theo epoch (seed nhỏ nhất) cho một số giá trị {spec['short']}; nét liền: train, nét đứt: validation.", 13)
        R.figure(f"fig_{g}_time.png", f"Thời gian huấn luyện và epoch đạt validation tốt nhất theo {spec['short']}.", 14)
        res = analyze(s)
        R.p("Nhận xét.", bold=True, indent=False)
        for t in res["paras"]: R.p(t)
        for t in res["notable"]: R.p("Trường hợp chưa như kỳ vọng: " + t, indent=False)

    result_section("A", "4.2. Thực nghiệm A – số mẫu huấn luyện", an.analyze_A)
    result_section("B", "4.3. Thực nghiệm B – số nơ-ron tầng ẩn (độ rộng)", an.analyze_B)
    result_section("C", "4.4. Thực nghiệm C – số tầng ẩn (độ sâu)", an.analyze_C)

    R.h("4.5. Mô hình được chọn và đánh giá cuối trên tập test", 2)
    F = D["final"]
    if F:
        R.p(f"Mô hình có validation MSE nhỏ nhất trong toàn bộ các lần chạy đến từ nhóm {F['group']} "
            f"(giá trị khảo sát = {F['factor_value']:.0f}, seed {F['seed']}, {F['n_train']} mẫu train, kiến trúc 2-{'-'.join(map(str, F['hidden_sizes']))}-1), "
            f"validation MSE = {an.fmt(F['val_mse'])}. Việc chọn chỉ dựa trên validation; test chỉ được dùng một lần để báo cáo.")
        R.table(["MSE", "RMSE", "MAE", "R²", "|f̂−f| lớn nhất (lưới)", "|f̂−f| trung bình (lưới)"],
                [[an.fmt(F["test_mse"]), f"{F['test_rmse']:.5f}", f"{F['test_mae']:.5f}", f"{F['test_r2']:.5f}",
                  f"{F['grid_abs_err_max']:.5f}", f"{F['grid_abs_err_mean']:.5f}"]], "Chỉ số trên tập test (100 mẫu) của mô hình được chọn", size=10)
        R.figure("fig_final_y_vs_yhat.png", "Giá trị thật y và dự đoán ŷ trên tập test của mô hình được chọn.", 9)
        R.figure("fig_final_surface.png", "Bề mặt hàm thật, bề mặt xấp xỉ của MLP và bản đồ sai số tuyệt đối trên lưới 120×120.", 16)
        br = an.best_rows(D)
        R.table(["Nhóm", "Giá trị tốt nhất (theo val)", "Val MSE", "Test MSE", "Test RMSE", "Test MAE", "Thời gian (s)"],
                [[r["label"], r["value"], an.fmt(r["val"]), an.fmt(r["test"]), f"{r['rmse']:.5f}", f"{r['mae']:.5f}", f"{r['time']:.1f}"] for r in br],
                "Cấu hình tốt nhất theo validation của từng nhóm", size=9)
    else:
        R.todo("Chưa có logs/final_test_metrics.json (chạy: python run_all.py hoặc experiments.final_evaluation).")

    # ------------------------------------------------ 5. Thảo luận
    R.h("5. Thảo luận")
    R.h("5.1. Diễn giải xu hướng", 2)
    R.p("Ba nhóm thực nghiệm cho thấy ba nguồn sai số khác nhau: (i) thiếu dữ liệu (nhóm A) làm tăng phương sai ước lượng, đặc biệt ở vùng đỉnh có ít mẫu; "
        "(ii) thiếu dung lượng biểu diễn (nhóm B với ít nơ-ron) tạo sai số hệ thống, thể hiện ở cả train và validation; "
        "(iii) khó khăn tối ưu và chi phí tính toán khi tăng độ sâu (nhóm C). Chi tiết số liệu xem các mục 4.2–4.4.")
    if D["A"] is not None and D["B"] is not None and D["C"] is not None:
        for t in an.analyze_B(D["B"])["short"] + an.analyze_C(D["C"])["short"]: R.bullets([t])
    R.h("5.2. Trường hợp không như kỳ vọng", 2)
    notes = []
    for g, fn in [("A", an.analyze_A), ("B", an.analyze_B), ("C", an.analyze_C)]:
        if D[g] is not None: notes += fn(D[g])["notable"]
    if notes:
        R.bullets(notes)
        R.p("Các điểm này thường do (a) số seed ít nên độ lệch chuẩn lớn so với khác biệt giữa hai cấu hình kề nhau, (b) early stopping dừng ở các epoch khác nhau, "
            "(c) tập test chỉ có 100 mẫu nên sai số test nhiễu hơn validation. Nên đối chiếu với thanh sai số trong các hình MSE và tăng số seed để kiểm chứng.")
    else:
        R.p("Với các giá trị khảo sát, validation MSE trung bình giảm đơn điệu ở cả ba nhóm; không ghi nhận điểm bất thường rõ rệt." if all(D[g] is not None for g in "ABC")
            else "Chưa đủ số liệu để đánh giá.")
    R.h("5.3. Hạn chế", 2)
    R.bullets(["Chỉ dùng số seed hạn chế (xem mục 4.1) nên độ lệch chuẩn ước lượng chưa chặt.", "Tập test chỉ 100 mẫu nên sai số test dao động lớn.",
               "Siêu tham số (learning rate, batch size, patience) giữ cố định cho mọi cấu hình, chưa tinh chỉnh riêng từng cấu hình.",
               "Nhãn không có nhiễu và hàm chỉ có một đỉnh trơn, nên kết luận về quá khớp có thể khác khi dữ liệu có nhiễu.",
               "Số bước cập nhật khác nhau giữa các A do batch size và số epoch cố định."])
    R.h("5.4. Đề xuất cải tiến", 2)
    R.bullets(["Tăng số seed (≥ 5) và dùng k-fold cross-validation để so sánh cấu hình chặt chẽ hơn.",
               "Tinh chỉnh learning rate / lịch giảm learning rate, thử hàm kích hoạt khác (ReLU, SiLU) và regularization (weight decay).",
               "Lấy mẫu dày hơn ở vùng đỉnh (importance sampling) để giảm sai số mà không tăng tổng số mẫu.",
               "Mở rộng khảo sát đồng thời độ rộng × độ sâu, hoặc thêm nhiễu vào nhãn để quan sát quá khớp rõ hơn."])

    # ------------------------------------------------ 6. Kết luận
    R.h("6. Kết luận")
    if all(D[g] is not None for g in "ABC"):
        a, b, c = an.analyze_A(D["A"]), an.analyze_B(D["B"]), an.analyze_C(D["C"])
        R.bullets([a["short"][0], a["short"][1], b["short"][0], b["short"][1], c["short"][0], c["short"][1]])
    else:
        R.todo("Điền kết luận sau khi có đủ kết quả ba nhóm.")
    R.p("Nhóm đã triển khai đầy đủ quy trình xấp xỉ hàm bằng MLP trên PyTorch với dữ liệu, chia tập và seed được ghi rõ, "
        "kết quả có thể tái lập bằng các lệnh ở phần phụ lục.")
    R.h("Tài liệu tham khảo")
    R.bullets(["I. Goodfellow, Y. Bengio, A. Courville, Deep Learning, MIT Press, 2016.",
               "A. Paszke et al., PyTorch: An Imperative Style, High-Performance Deep Learning Library, NeurIPS, 2019.",
               "D. P. Kingma, J. Ba, Adam: A Method for Stochastic Optimization, ICLR, 2015.",
               "K. Hornik, M. Stinchcombe, H. White, Multilayer feedforward networks are universal approximators, Neural Networks, 1989.",
               "Tài liệu bài giảng và đề bài tập lớn môn Học sâu (BTL01)."])
    R.h("Phụ lục. Lệnh tái lập thực nghiệm")
    R.code("pip install -r requirements.txt\n"
           "python run_data.py                      # sinh dữ liệu, thống kê, hình dữ liệu\n"
           "python run_experiment.py --group A      # thực nghiệm A\n"
           "python run_experiment.py --group B      # thực nghiệm B\n"
           "python run_experiment.py --group C      # thực nghiệm C\n"
           "python run_all.py                       # chạy toàn bộ + đánh giá cuối trên test\n"
           "python build_report.py                  # dựng báo cáo .docx/.pdf\n"
           "python build_slides.py                  # dựng slide .pptx")

    out = cfg.REPORT_DIR / "BaoCao_BTL01.docx"
    R.doc.save(out)
    print("Đã tạo", out)
    if not no_pdf:
        to_pdf(out)
    return out


def to_pdf(docx_path):
    pdf = docx_path.with_suffix(".pdf")
    try:
        from docx2pdf import convert
        convert(str(docx_path), str(pdf))
        if pdf.exists():
            print("Đã tạo", pdf); return
    except Exception:
        pass
    for exe in ("soffice", "libreoffice"):
        if shutil.which(exe):
            try:
                subprocess.run([exe, "--headless", "--convert-to", "pdf", "--outdir", str(docx_path.parent), str(docx_path)],
                               check=True, timeout=240, capture_output=True)
                if pdf.exists():
                    print("Đã tạo", pdf); return
            except Exception:
                pass
    print("Không tự chuyển được sang PDF: mở file .docx bằng Word rồi chọn File > Save as > PDF.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    build(ap.parse_args().no_pdf)
