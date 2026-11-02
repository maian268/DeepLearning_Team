"""Dựng slide thuyết trình (.pptx) từ log/hình trong logs/ và figures/.
  python build_slides.py
Chạy SAU khi đã chạy thực nghiệm. Nội dung thiếu số liệu được đánh dấu nền VÀNG; cần điền tên nhóm/thành viên.
Có thể xuất PDF từ PowerPoint (File > Export) nếu muốn nộp kèm bản PDF."""
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt
from src import config as cfg
from src import analysis as an

NAVY, TEAL, INK, GREY, YEL = RGBColor(0x1F, 0x3A, 0x5F), RGBColor(0x2A, 0x9D, 0x8F), RGBColor(0x22, 0x22, 0x22), RGBColor(0x66, 0x66, 0x66), RGBColor(0xFF, 0xF2, 0x7A)
W, H = 13.333, 7.5
FONT = "Calibri"


class Deck:
    def __init__(self):
        self.prs = Presentation(); self.prs.slide_width, self.prs.slide_height = Inches(W), Inches(H)
        self.n = 0

    def slide(self, title, notes=None):
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6]); self.n += 1
        band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(W), Inches(1.05)); band.fill.solid(); band.fill.fore_color.rgb = NAVY; band.line.fill.background()
        self.text(s, title, 0.6, 0.12, W - 1.2, 0.8, 28, bold=True, color=RGBColor(255, 255, 255), anchor=MSO_ANCHOR.MIDDLE)
        self.text(s, f"BTL01 – Học sâu  |  {self.n}", 0.6, 7.05, W - 1.2, 0.3, 11, color=GREY, align=PP_ALIGN.RIGHT)
        if notes: s.notes_slide.notes_text_frame.text = notes
        return s

    def text(self, s, txt, x, y, w, h, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font=FONT, fill=None):
        tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
        tf.margin_left = tf.margin_right = Inches(0.05)
        if fill is not None: tb.fill.solid(); tb.fill.fore_color.rgb = fill
        p = tf.paragraphs[0]; p.alignment = align; r = p.add_run(); r.text = txt
        r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), bold, color, font
        return tb

    def bullets(self, s, items, x, y, w, h, size=18, gap=8):
        tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
        for i, it in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.space_after = Pt(gap)
            r = p.add_run(); r.text = "•  " + it; r.font.size, r.font.color.rgb, r.font.name = Pt(size), INK, FONT
        return tb

    def todo(self, s, txt, x, y, w, h, size=16):
        return self.text(s, "[CẦN BỔ SUNG] " + txt, x, y, w, h, size, color=INK, fill=YEL)

    def picture(self, s, name, x, y, w, h, caption=None):
        path = cfg.FIG_DIR / name
        if not path.exists():
            self.todo(s, f"Thiếu hình {name} – chạy thực nghiệm rồi dựng lại slide.", x, y, w, min(h, 1.2)); return
        iw, ih = Image.open(path).size; sc = min(w / iw, h / ih); pw, ph = iw * sc, ih * sc
        s.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))
        if caption: self.text(s, caption, x, y + h, w, 0.3, 11, color=GREY, align=PP_ALIGN.CENTER)

    def table(self, s, header, rows, x, y, w, col_w=None, size=13, row_h=0.42):
        shp = s.shapes.add_table(len(rows) + 1, len(header), Inches(x), Inches(y), Inches(w), Inches(row_h * (len(rows) + 1)))
        t = shp.table
        if col_w:
            for i, cw in enumerate(col_w): t.columns[i].width = Inches(cw)
        for j, hd in enumerate(header):
            c = t.cell(0, j); c.text = str(hd); c.fill.solid(); c.fill.fore_color.rgb = NAVY
            for p in c.text_frame.paragraphs:
                for r in p.runs: r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), True, RGBColor(255, 255, 255), FONT
        for i, row in enumerate(rows, start=1):
            for j, v in enumerate(row):
                c = t.cell(i, j); c.text = str(v); c.fill.solid(); c.fill.fore_color.rgb = RGBColor(0xF4, 0xF6, 0xFA) if i % 2 else RGBColor(255, 255, 255)
                for p in c.text_frame.paragraphs:
                    for r in p.runs: r.font.size, r.font.color.rgb, r.font.name = Pt(size), INK, FONT


def build():
    cfg.ensure_dirs()
    D = an.load_all(); K = Deck(); T = cfg.TRAIN
    quick = bool((D["config"] or {}).get("quick_mode", False))

    # 1. Tiêu đề
    s = K.prs.slides.add_slide(K.prs.slide_layouts[6]); K.n += 1
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(W), Inches(H)); bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
    K.text(s, "BÀI TẬP LỚN 01 – HỌC SÂU", 0.9, 1.3, 11.5, 0.6, 20, color=RGBColor(0xBF, 0xD7, 0xEA))
    K.text(s, "Xấp xỉ hàm Gaussian 2D bằng mạng MLP (PyTorch)", 0.9, 2.0, 11.5, 1.5, 40, bold=True, color=RGBColor(255, 255, 255))
    K.text(s, "Ảnh hưởng của số mẫu huấn luyện (A), số nơ-ron tầng ẩn (B) và số tầng ẩn (C)", 0.9, 3.7, 11.5, 0.8, 22, color=RGBColor(0xBF, 0xD7, 0xEA))
    K.todo(s, "Tên nhóm – họ tên và MSSV các thành viên – giảng viên", 0.9, 5.3, 9.5, 0.9, 18)
    if quick: K.todo(s, "Số liệu đang ở chế độ chạy thử (--quick); hãy chạy đầy đủ rồi dựng lại.", 0.9, 6.4, 11, 0.6, 14)

    # 2. Mục tiêu & bài toán
    s = K.slide("Mục tiêu và bài toán", "Nêu bài toán hồi quy: MLP 2 đầu vào, 1 đầu ra xấp xỉ hàm Gaussian; ba câu hỏi nghiên cứu A, B, C.")
    K.bullets(s, ["Xây dựng và huấn luyện MLP (PyTorch) để xấp xỉ hàm mật độ Gaussian hai chiều f(x₁, x₂).",
                  "Quy trình đầy đủ: dữ liệu → mô hình → lan truyền ngược → tối ưu → đánh giá.",
                  "A: số mẫu huấn luyện ảnh hưởng thế nào đến sai số?",
                  "B: bao nhiêu nơ-ron tầng ẩn thì đủ? Khi nào thiếu khớp / quá khớp?",
                  "C: tăng số tầng ẩn có cải thiện không, và tốn thêm bao nhiêu thời gian?",
                  "Đánh giá công bằng: dữ liệu cố định, validation để chọn mô hình, test chỉ để báo cáo cuối."], 0.7, 1.4, 12, 5, 22, 14)

    # 3. Dữ liệu Gaussian
    s = K.slide("Hàm Gaussian 2D và dữ liệu", "Công thức, tham số μ, C; det C = 0.35; đỉnh f(μ) ≈ 0.2693. Lấy mẫu đều trong miền, nhãn y = f(x).")
    K.text(s, "f(x) = exp(−½ (x−μ)ᵀ C⁻¹ (x−μ)) / (2π √det C)", 0.6, 1.3, 12, 0.7, 20, bold=True, color=NAVY)
    K.bullets(s, ["μ = (0, 0)ᵀ;  C = [[0,3; −0,5], [−0,5; 2]]", "det C = 0,35 > 0, đỉnh f(μ) ≈ 0,2693",
                  "x₁ ∈ [−2, 2], x₂ ∈ [−4, 4], lấy mẫu đều", f"Seed dữ liệu {cfg.DATA_SEED}; 3 dòng ngẫu nhiên độc lập",
                  f"Train gốc {cfg.N_POOL} · Validation {cfg.N_VAL} · Test {cfg.N_TEST}", "Mức A dùng A mẫu đầu của train gốc"], 0.6, 2.3, 5.2, 4.5, 16, 8)
    K.picture(s, "fig02_true_surface.png", 6.0, 1.6, 7.0, 4.6, "Hàm thật và đường đồng mức")

    # 4. Tập dữ liệu
    s = K.slide("Minh họa tập dữ liệu", "Scatter 3D train/val/test; y tập trung gần 0, đỉnh nhọn chiếm ít mẫu.")
    K.picture(s, "fig01_data_scatter3d.png", 0.5, 1.3, 6.2, 5.4, "Scatter 3D train / validation / test")
    K.picture(s, "fig03_distributions.png", 6.8, 1.5, 6.1, 2.6, "Phân bố x₁, x₂, y")
    nz = an.near_zero_pct()
    K.bullets(s, [(f"Chỉ ≈ {100 - nz:.0f}% mẫu train có y > 1% đỉnh" if nz is not None else "Phần lớn mẫu có y gần 0"),
                  "x₁, x₂ phân bố đều; y lệch phải", "Vùng đỉnh ít mẫu → thách thức khi A nhỏ"], 6.9, 4.6, 6.0, 2.2, 17, 8)

    # 5. Sơ đồ MLP
    s = K.slide("Sơ đồ MLP và kiến trúc", "Kiến trúc 2-32-32-1 với tanh cho nhóm A; B đổi số nơ-ron, C đổi số tầng.")
    K.picture(s, "fig_architecture.png", 0.5, 1.3, 12.3, 3.9, "MLP chuẩn: 2 → 32 → 32 → 1 (tanh, tầng ra tuyến tính)")
    K.bullets(s, ["Forward: z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾,  a⁽ˡ⁾ = tanh(z⁽ˡ⁾),  ŷ = tầng ra tuyến tính",
                  "Mất mát MSE;  chuẩn hóa x/(2, 4), y/f(μ) (hằng số theo miền, không rò rỉ)"], 0.7, 5.6, 12, 1.3, 18, 8)

    # 6. PyTorch
    s = K.slide("Thành phần chính trong chương trình PyTorch", "Giải thích vòng lặp: forward → loss → zero_grad → backward → step; eval + no_grad cho validation; checkpoint.")
    code = ("for xb, yb in train_loader:\n    y_hat = model(xb)          # forward\n    loss = criterion(y_hat, yb) # MSELoss\n"
            "    optimizer.zero_grad()\n    loss.backward()            # backprop\n    optimizer.step()           # Adam\n\n"
            "model.eval()\nwith torch.no_grad():\n    val_loss = evaluate_loss(...)\n\nif val_loss < best_val*(1-tol):\n    best_state = deepcopy(model.state_dict())")
    K.text(s, code, 0.6, 1.35, 6.6, 5.4, 14, font="Consolas", fill=RGBColor(0xF2, 0xF2, 0xF2))
    K.bullets(s, ["MLP(nn.Module): nn.Linear + Tanh, hidden_sizes tùy biến", "TensorDataset + DataLoader (xáo batch theo seed)", "nn.MSELoss, optimizer Adam (lr 3e-3)",
                  "model.train() / model.eval() / torch.no_grad()", "Early stopping + checkpoint theo validation", "Log CSV, lưu mô hình .pt"], 7.5, 1.4, 5.4, 5.3, 18, 10)

    # 7. Thiết lập A/B/C
    s = K.slide("Thiết lập thực nghiệm A / B / C", f"Cố định: seed {', '.join(map(str, cfg.SEEDS))}, batch {T['batch_size']}, tối đa {T['max_epochs']} epoch, patience {T['patience']}.")
    K.table(s, ["Nhóm", "Biến thay đổi", "Giá trị", "Giữ cố định"],
            [["A", "Số mẫu train", f"{cfg.A_LIST[0]}…{cfg.A_LIST[-1]} (10 mức)", f"Kiến trúc {'-'.join(map(str, cfg.A_HIDDEN))}"],
             ["B", "Số nơ-ron (1 tầng ẩn)", ", ".join(map(str, cfg.B_LIST)), f"{cfg.N_FIXED} mẫu train"],
             ["C", "Số tầng ẩn", ", ".join(map(str, cfg.C_LIST)), f"{cfg.C_WIDTH} nơ-ron/tầng, {cfg.N_FIXED} mẫu"]],
            0.6, 1.5, 12.1, [1.0, 3.2, 4.6, 3.3], 15, 0.6)
    K.bullets(s, [f"Adam, lr {T['lr']}, batch {T['batch_size']}, tối đa {T['max_epochs']} epoch, early stopping patience {T['patience']}",
                  "Mỗi cấu hình chạy nhiều seed → báo cáo trung bình ± độ lệch chuẩn", "Validation: dừng sớm + chọn checkpoint; Test: chỉ đánh giá cuối",
                  "Metric: MSE (chính), RMSE, MAE, R² trên thang đo gốc của y"], 0.7, 4.0, 12, 3, 19, 10)

    # 8-10. Kết quả A, B, C
    def result_slide(g, title, fn, note):
        s = K.slide(title, note)
        K.picture(s, f"fig_{g}_mse.png", 0.4, 1.25, 7.5, 5.4, f"MSE theo {cfg.group_spec(g)['short']} (mean ± std)")
        if D[g] is None:
            K.todo(s, f"Chưa có kết quả nhóm {g}: chạy python run_experiment.py --group {g}", 8.2, 1.4, 4.7, 1.5); return
        r = fn(D[g]); K.bullets(s, r["short"], 8.2, 1.4, 4.8, 5.4, 16, 12)
    result_slide("A", "Kết quả A – số mẫu huấn luyện", an.analyze_A, "Đọc xu hướng sai số theo A, điểm bão hòa, khoảng cách train–validation.")
    result_slide("B", "Kết quả B – số nơ-ron tầng ẩn", an.analyze_B, "Vùng thiếu khớp / quá khớp theo B.")
    result_slide("C", "Kết quả C – số tầng ẩn", an.analyze_C, "Sai số, hội tụ và thời gian theo C.")

    # 11. Hội tụ và thời gian
    s = K.slide("Đường cong hội tụ và thời gian huấn luyện", "So sánh tốc độ hội tụ và chi phí giữa các độ sâu (C) và độ rộng (B).")
    K.picture(s, "fig_C_curves.png", 0.4, 1.3, 6.3, 5.0, "C: loss theo epoch (nét đứt: validation)")
    K.picture(s, "fig_C_time.png", 6.8, 1.3, 6.2, 3.0, "C: thời gian và epoch tốt nhất")
    K.picture(s, "fig_B_curves.png", 6.8, 4.3, 6.2, 2.5, "B: loss theo epoch")

    # 12. Bảng tổng hợp
    s = K.slide("Bảng tổng hợp cấu hình tốt nhất theo validation", "Chọn theo validation; test chỉ báo cáo.")
    br = an.best_rows(D)
    if br:
        K.table(s, ["Nhóm", "Giá trị tốt nhất", "Val MSE", "Test MSE", "Test RMSE", "Test MAE", "Thời gian (s)"],
                [[r["label"], r["value"], an.fmt(r["val"]), an.fmt(r["test"]), f"{r['rmse']:.4f}", f"{r['mae']:.4f}", f"{r['time']:.1f}"] for r in br],
                0.5, 1.6, 12.3, [3.0, 1.7, 1.6, 1.6, 1.5, 1.5, 1.4], 14, 0.6)
        if D["final"]:
            F = D["final"]
            K.bullets(s, [f"Mô hình cuối: nhóm {F['group']}, kiến trúc 2-{'-'.join(map(str, F['hidden_sizes']))}-1, {F['n_train']} mẫu train, seed {F['seed']}",
                          f"Test: MSE {an.fmt(F['test_mse'])}, RMSE {F['test_rmse']:.4f}, MAE {F['test_mae']:.4f}, R² {F['test_r2']:.4f}"], 0.7, 4.2, 12, 2.5, 19, 10)
    else:
        K.todo(s, "Chưa có kết quả thực nghiệm.", 0.6, 1.6, 12, 1.0)

    # 13. Minh họa xấp xỉ
    s = K.slide("Minh họa mô hình xấp xỉ và dữ liệu thật", "Bề mặt thật vs bề mặt MLP, sai số tập trung ở đâu.")
    K.picture(s, "fig_final_surface.png", 0.4, 1.2, 12.5, 3.7, "Hàm thật · bề mặt MLP · sai số tuyệt đối")
    K.picture(s, "fig_final_y_vs_yhat.png", 0.6, 5.2, 2.4, 1.85)
    if D["final"]:
        F = D["final"]
        K.bullets(s, [f"Sai số tuyệt đối lớn nhất trên lưới: {F['grid_abs_err_max']:.4f} (đỉnh hàm ≈ {F['y_peak']:.3f})",
                      f"Sai số trung bình trên lưới: {F['grid_abs_err_mean']:.5f}"], 3.0, 5.3, 9.8, 1.6, 17, 8)

    # 14. Kết luận
    s = K.slide("Kết luận, hạn chế và hướng cải tiến", "Tóm tắt 3 kết luận chính, 2–3 hạn chế, 2–3 hướng phát triển.")
    concl = []
    if all(D[g] is not None for g in "ABC"):
        concl = [an.analyze_A(D["A"])["short"][0], an.analyze_B(D["B"])["short"][0], an.analyze_C(D["C"])["short"][0]]
        K.bullets(s, concl, 0.6, 1.3, 12.2, 2.6, 17, 8)
    else:
        K.todo(s, "Điền kết luận sau khi có đủ kết quả A, B, C.", 0.6, 1.3, 12, 0.9)
    K.text(s, "Hạn chế", 0.6, 4.0, 5.8, 0.4, 20, bold=True, color=TEAL)
    K.bullets(s, ["Ít seed, test chỉ 100 mẫu", "Siêu tham số cố định, không tinh chỉnh riêng", "Nhãn không nhiễu, hàm một đỉnh"], 0.6, 4.5, 5.9, 2.4, 16, 6)
    K.text(s, "Hướng cải tiến", 6.9, 4.0, 5.8, 0.4, 20, bold=True, color=TEAL)
    K.bullets(s, ["Tăng seed, cross-validation", "Lấy mẫu dày ở vùng đỉnh, thử ReLU/SiLU, weight decay", "Khảo sát đồng thời độ rộng × độ sâu"], 6.9, 4.5, 5.9, 2.4, 16, 6)

    # 15. Demo
    s = K.slide("Demo mã nguồn và hỏi đáp", "Sẵn sàng chạy: python run_all.py --quick để demo nhanh; giải thích train/validation/test.")
    K.text(s, "pip install -r requirements.txt\npython run_data.py\npython run_experiment.py --group A\npython run_experiment.py --group B\npython run_experiment.py --group C\npython run_all.py --quick     # demo nhanh\npython build_report.py && python build_slides.py",
           0.7, 1.5, 11.9, 3.2, 17, font="Consolas", fill=RGBColor(0xF2, 0xF2, 0xF2))
    K.bullets(s, ["Cấu trúc: src/ (data, model, train, experiments, plots) · run_*.py · logs/ · figures/ · models/",
                  "Có thể giải thích: quy trình train/validation/test, early stopping, vì sao không dùng test để chọn cấu hình"], 0.7, 5.0, 12, 1.8, 17, 8)

    out = cfg.SLIDE_DIR / "Slide_BTL01.pptx"
    K.prs.save(out); print("Đã tạo", out, f"({K.n} slide)")


if __name__ == "__main__":
    build()
