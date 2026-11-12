"""
app/app.py - Giao diện Web App Streamlit cho Mô hình ESC-50

Hạng mục công việc: H-04 (Thanh - SV4)
Ghi chú: Các vị trí cần nhóm hoàn thiện/nối code được đánh dấu bằng tag `# TODO:`
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
import base64
import streamlit.components.v1 as components

from datetime import datetime
# -----------------------------------------------------------------------------
# TODO [THIẾU 1]: CẤU HÌNH THƯ MỤC NGUỒN (PROJECT ROOT)
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# -----------------------------------------------------------------------------
# TODO [THIẾU 2]: IMPORT CÁC THƯ VIỆN/MODULE NỘI BỘ CỦA NHÓM
# -----------------------------------------------------------------------------
try:
    from src.models import build_model
except ImportError:
    build_model = None

try:
    from src.data_lib import wav_to_logmel
except ImportError:
    wav_to_logmel = None

# Danh sách 50 lớp âm thanh ESC-50
ESC50_CLASSES = [
    "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow",
    "rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds", "water_drops",
    "wind", "pouring_water", "toilet_flush", "thunderstorm", "crying_baby", "sneezing",
    "clapping", "coughing", "footsteps", "laughing", "tooting_synth", "snoring", "drinking_sipping",
    "door_wood_knock", "mouse_click", "keyboard_typing", "door_wood_creak", "can_opening",
    "washing_machine", "vacuum_cleaner", "clock_ticking", "glass_breaking", "helicopter",
    "chainsaw", "siren", "car_horn", "engine", "train", "church_bells", "airplane", "fireworks", "hand_saw"
]

DEFAULT_CONFIDENCE_THRESHOLD = 0.30  # Ngưỡng cảnh báo độ tin cậy thấp (30%)
MAX_FILE_SIZE_MB = 10               # Dung lượng tối đa 10 MB
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Đường dẫn tới 2 tấm ảnh cảnh báo
IMG_INVALID_FORMAT = "app/Assets/Thông báo sai định dạng.png"
IMG_FILE_TOO_LARGE = "app/Assets/Thông báo vượt giới hạn dung lượng tệp.png"

st.set_page_config(
    page_title="ESC-50 Audio Classifier",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 ESC-50 Audio Classification Demo")
st.markdown("Ứng dụng phân loại 50 lớp âm thanh môi trường sử dụng **CNN** và **CRNN**.")



# =========================================================
# SIDEBAR: CẤU HÌNH MÔ HÌNH VÀ TẢI CHECKPOINT
# =========================================================
st.sidebar.header("⚙️ Cấu hình Mô hình")

arch_choice = st.sidebar.selectbox(
    "1. Chọn kiến trúc mô hình (arch):",
    options=["crnn", "cnn"],
    format_func=lambda x: "CRNN (Convolutional Recurrent NN)" if x == "crnn" else "CNN Baseline"
)

n_mels_choice = st.sidebar.selectbox(
    "2. Chọn số lượng dải Mel (n_mels):",
    options=[128, 64]
)

threshold_input = st.sidebar.slider(
    "3. Ngưỡng cảnh báo tin cậy (Low confidence threshold):",
    min_value=0.10,
    max_value=0.80,
    value=DEFAULT_CONFIDENCE_THRESHOLD,
    step=0.05,
    help="Nếu xác suất của nhãn dự đoán cao nhất nhỏ hơn ngưỡng này, hệ thống sẽ đưa ra cảnh báo."
)

# -----------------------------------------------------------------------------
# TODO [THIẾU 3]: ĐƯỜNG DẪN CHECKPOINT THẬT TỪ THỰC NGHIỆM (EXP1 - EXP5)
# -----------------------------------------------------------------------------
checkpoint_path_input = st.sidebar.text_input(
    "4. Đường dẫn file checkpoint (.pth):",
    value=f"checkpoints/{arch_choice}_{n_mels_choice}bins_best.pth",
    help="Cần đường dẫn tới file trọng số .pth thật sau khi huấn luyện xong."
)

@st.cache_resource
def load_trained_model(arch: str, n_mels: int, ckpt_path: str):
    """Hàm nạp trọng số mô hình PyTorch đã huấn luyện"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # TODO [THIẾU 4]: KIỂM TRA HÀM BUILD_MODEL REGISTRY
    if build_model is None:
        return None, device, "❌ Chưa kết nối được hàm `build_model` từ src/models/__init__.py."

    path = Path(ckpt_path)
    # TODO [THIẾU 5]: ĐOẠN XỬ LÝ KHI THIẾU FILE CHECKPOINT (.pth)
    if not path.exists():
        return None, device, f"⚠️ Chưa tìm thấy file checkpoint tại: `{ckpt_path}`. Đang chạy chế độ mô phỏng (Dummy mode)."

    try:
        model = build_model(arch=arch, n_mels=n_mels, n_classes=50)
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, device, None
    except Exception as e:
        return None, device, f"❌ Lỗi nạp checkpoint: {e}"

model, device, load_status_msg = load_trained_model(arch_choice, n_mels_choice, checkpoint_path_input)

if load_status_msg:
    st.sidebar.info(load_status_msg)
else:
    st.sidebar.success(f"✅ Đã nạp thành công mô hình `{arch_choice.upper()}` ({n_mels_choice} bins).")

# =========================================================
# PHẦN CHÍNH: UPLOAD VÀ XỬ LÝ ÂM THANH
# =========================================================
# Hàm phụ trợ đọc ảnh sang chuỗi Base64
def get_image_base64(img_path: str) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Khởi tạo trạng thái lưu vết lỗi đã tắt hay chưa
if "dismissed_error_for" not in st.session_state:
    st.session_state.dismissed_error_for = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

def reset_uploader():
    st.session_state.dismissed_error_for = None
    st.session_state.uploader_key += 1

# Khởi tạo danh sách lưu lịch sử phân loại trong phiên làm việc
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ---------------------------------------------------------
# BƯỚC 1: TẢI LÊN TỆP ÂM THANH
# ---------------------------------------------------------
st.subheader("Tải lên tệp âm thanh (.wav)")

uploaded_file = st.file_uploader(
    "Vui lòng chọn 1 file WAV âm thanh 5 giây (Chỉ hỗ trợ định dạng .wav, tối đa 10MB):",
    type=None,
    key=f"audio_uploader_{st.session_state.uploader_key}"
)

# JS ép hộp thoại Windows mặc định chỉ hiện file .wav
# JS theo dõi liên tục và giữ cố định bộ lọc .wav trên hộp thoại chọn file
components.html(
    """
    <script>
    const doc = window.parent.document;
    function enforceWavAccept() {
        const inputs = doc.querySelectorAll('input[type="file"]');
        inputs.forEach(input => {
            if (input.getAttribute('accept') !== '.wav') {
                input.setAttribute('accept', '.wav');
            }
        });
    }
    enforceWavAccept();
    const observer = new MutationObserver(enforceWavAccept);
    observer.observe(doc.body, { childList: true, subtree: true });
    </script>
    """,
    height=0
)
col_left, col_right = st.columns([1, 1])

if uploaded_file is not None:
    file_name = uploaded_file.name
    file_size = uploaded_file.size
    file_id = f"{file_name}_{file_size}"

    # ---------------------------------------------------------
    # BƯỚC 2: KIỂM TRA LỖI (QUÁ DUNG LƯỢNG / SAI ĐỊNH DẠNG)
    # ---------------------------------------------------------
    is_valid_file = True
    error_img_path = None
    error_reason = ""

    if file_size > MAX_FILE_SIZE_BYTES:
        is_valid_file = False
        error_img_path = IMG_FILE_TOO_LARGE
        error_reason = f"dung lượng tệp ({file_size / (1024*1024):.2f} MB) vượt quá giới hạn {MAX_FILE_SIZE_MB} MB"
    elif not file_name.lower().endswith(".wav"):
        is_valid_file = False
        error_img_path = IMG_INVALID_FORMAT
        error_reason = "định dạng tệp không phải là .wav"

    if not is_valid_file:
        with col_left:
            # TRƯỜNG HỢP A: CHƯA BẤM DẤU X -> HIỆN ẢNH TƯƠNG TÁC TRỰC TIẾP
            if st.session_state.dismissed_error_for != file_id:
                try:
                    img_b64 = get_image_base64(error_img_path)

                    # 1. Tạo nút Streamlit ẩn để nhận sự kiện click vào dấu X trên ảnh
                    st.markdown(
                        """
                        <style>
                        div[data-testid="stButton"] button[kind="secondary"] {
                            display: none !important;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
                    if st.button("HIDDEN_CLOSE_TRIGGER", key="hidden_close_btn"):
                        st.session_state.dismissed_error_for = file_id
                        st.rerun()

                    # 2. Render ảnh kèm 2 vùng bấm trong suốt khớp tọa độ nút X và thanh hướng dẫn
                    components.html(
                        f"""
                        <div style="position: relative; width: 100%; max-width: 560px; font-family: sans-serif;">
                            <img src="data:image/png;base64,{img_b64}" 
                                 style="width: 100%; display: block;" />

                            <!-- VÙNG BẤM 1: Khớp nút X tròn màu xanh góc trên bên phải -->
                            <div onclick="closeAlert()" 
                                 title="Đóng thông báo"
                                 style="position: absolute; top: 14.5%; right: 4.2%; width: 6.5%; height: 10.5%; 
                                        cursor: pointer; border-radius: 50%; z-index: 10;">
                            </div>

                            <!-- VÙNG BẤM 2: Khớp khung 'Vui lòng chọn lại / tải lên...' ở đáy bên phải -->
                            <div onclick="triggerReupload()" 
                                 title="Bấm để chọn tệp âm thanh mới"
                                 style="position: absolute; bottom: 22.5%; left: 32%; width: 64%; height: 17.5%; 
                                        cursor: pointer; border-radius: 16px; z-index: 10;">
                            </div>
                        </div>

                        <script>
                        function closeAlert() {{
                            const buttons = window.parent.document.querySelectorAll('button');
                            for (const btn of buttons) {{
                                if (btn.innerText.includes('HIDDEN_CLOSE_TRIGGER')) {{
                                    btn.click();
                                    break;
                                }}
                            }}
                        }}

                        function triggerReupload() {{
                            const fileInputs = window.parent.document.querySelectorAll('input[type="file"]');
                            if (fileInputs.length > 0) {{
                                fileInputs[0].click();
                            }}
                        }}
                        </script>
                        """,
                        height=380
                    )

                except Exception:
                    st.error(f"❌ Tệp không hợp lệ ({error_reason}). Vui lòng chọn lại file .wav!")

            # TRƯỜNG HỢP B: ĐÃ BẤM VÀO DẤU X TRÊN ẢNH -> TẮT ẢNH & HIỆN CÂU NHẮC NHẬP LẠI + NÚT UPLOAD
           # TRƯỜNG HỢP B: ĐÃ BẤM VÀO DẤU X TRÊN ẢNH -> HIỆN NHẮC NHỞ + NÚT MỞ THẲNG HỘP THOẠI CHỌN FILE
            else:
                st.warning(f"⚠️ Tệp vừa tải lên bị lỗi do **{error_reason}**. **Vui lòng nhập lại tệp âm thanh định dạng `.wav` hợp lệ!**")
                
                # Nút bấm kích hoạt trực tiếp hộp thoại chọn file của hệ điều hành (không reset về Bước 1)
                components.html(
                    """
                    <style>
                    body { margin: 0; padding: 0; font-family: "Source Sans Pro", sans-serif; }
                    .reupload-btn {
                        background-color: #ff4b4b;
                        color: white;
                        border: none;
                        border-radius: 8px;
                        padding: 10px 20px;
                        font-size: 15px;
                        font-weight: 600;
                        cursor: pointer;
                        display: inline-flex;
                        align-items: center;
                        gap: 8px;
                        transition: background-color 0.2s;
                    }
                    .reupload-btn:hover {
                        background-color: #ff3333;
                    }
                    </style>

                    <button class="reupload-btn" onclick="openFileDialog()">
                        📂 Tải lên tệp âm thanh mới
                    </button>

                    <script>
                    function openFileDialog() {
                        const doc = window.parent.document;
                        const fileInputs = doc.querySelectorAll('input[type="file"]');
                        if (fileInputs.length > 0) {
                            fileInputs[0].setAttribute('accept', '.wav');
                            fileInputs[0].click();
                        }
                    }
                    </script>
                    """,
                    height=50
                )

    # BƯỚC 3: TRÌNH PHÁT ÂM THANH VÀ VẼ LOG-MEL SPECTROGRAM
    # (Chỉ thực hiện nếu file HỢP LỆ)
    if is_valid_file:
        with col_left:
            st.markdown("#### 🔊 Trình phát âm thanh")
            audio_bytes = uploaded_file.read()
            st.audio(audio_bytes, format="audio/wav")

        # TODO [THIẾU 6]: TRÍCH XUẤT PHỔ LOG-MEL THẬT TỪ FILE AUDIO (.wav)
        mel_spec = None
        if wav_to_logmel is not None:
            try:
                uploaded_file.seek(0)
                mel_spec = wav_to_logmel(uploaded_file, n_mels=n_mels_choice)
            except Exception as e:
                st.warning(f"Không thể xử lý file audio bằng data_lib: {e}. Đang dùng phổ Mel giả lập.")
                mel_spec = None

        if mel_spec is None:
            # TODO [TẠM THỜI]: Dữ liệu giả lập phổ Log-Mel [n_mels, 216]
            np.random.seed(42)
            mel_spec = np.random.randn(n_mels_choice, 216)

        # Vẽ biểu đồ Log-Mel Spectrogram
        with col_left:
            st.markdown("#### 📊 Log-Mel Spectrogram")
            fig_mel, ax_mel = plt.subplots(figsize=(6, 3.5))
            cax = ax_mel.imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis')
            fig_mel.colorbar(cax, ax=ax_mel, format="%+2.0f dB")
            ax_mel.set_title(f"Log-Mel Spectrogram ({n_mels_choice} Mel bins)")
            ax_mel.set_xlabel("Time Frames")
            ax_mel.set_ylabel("Mel Frequency Bins")
            fig_mel.tight_layout()
            st.pyplot(fig_mel)

        # ---------------------------------------------------------
        # BƯỚC 4: SUY LUẬN VÀ TÍNH XÁC SUẤT TOP-1 (VỚI CẢNH BÁO TÍN CẬY THẤP)
        # ---------------------------------------------------------
        with col_right:
            st.markdown("#### 🤖 Kết quả Phân loại & Dự đoán")
            
            # TODO [THIẾU 7]: CHẠY DỰ ĐOÁN VỚI MÔ HÌNH THẬT NẾU ĐÃ LOAD ĐƯỢC MODEL
            if model is not None:
                mel_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(mel_tensor)
                    probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            else:
                # TODO [TẠM THỜI]: Mảng xác suất giả ngẫu nhiên
                np.random.seed(len(audio_bytes) % 100)
                probs = np.random.dirichlet(np.ones(50))

            top5_indices = np.argsort(probs)[::-1][:5]
            top1_index = top5_indices[0]
            top1_class = ESC50_CLASSES[top1_index]
            top1_prob = probs[top1_index]
        # Tạo mã định danh duy nhất cho lần chạy này (tên file + cấu hình mô hình)
        run_signature = f"{file_name}_{arch_choice}_{n_mels_choice}_{top1_class}_{top1_prob:.4f}"

        # Chỉ lưu nếu bản ghi này chưa nằm ở dòng mới nhất (tránh trùng lặp khi rerun)
        if not st.session_state.prediction_history or st.session_state.prediction_history[0]["_sig"] != run_signature:
            st.session_state.prediction_history.insert(0, {
                "_sig": run_signature,
                "Thời gian": datetime.now().strftime("%H:%M:%S"),
                "Tên tệp": file_name,
                "Mô hình": f"{arch_choice.upper()} ({n_mels_choice} bins)",
                "Dự đoán Top-1": top1_class,
                "Độ tin cậy (%)": round(float(top1_prob * 100), 2),
                "Trạng thái": "⚠️ Tin cậy thấp" if top1_prob < threshold_input else "✅ Đạt ngưỡng"
            })
            st.metric(
                label="Nhãn dự đoán cao nhất (Top-1):",
                value=top1_class.upper(),
                delta=f"Độ tin cậy: {top1_prob * 100:.2f}%"
            )

            # CẢNH BÁO ĐỘ TIN CẬY THẤP (LOW CONFIDENCE ALERT)
            if top1_prob < threshold_input:
                st.warning(
                    f"⚠️ **Cảnh báo độ tin cậy thấp!**\n\n"
                    f"Độ tin cậy của nhãn dự đoán top-1 ({top1_prob * 100:.2f}%) nhỏ hơn ngưỡng quy định ({threshold_input * 100:.0f}%). "
                    f"Mẫu âm thanh này có thể bị nhiễu hoặc nằm ngoài các dạng âm thanh ESC-50 điển hình."
                )

            # ---------------------------------------------------------
            # BƯỚC 5: BIỂU ĐỒ CỘT TOP-5 XÁC SUẤT VÀ BẢNG SỐ LIỆU
            # ---------------------------------------------------------
            st.markdown("##### 🏆 Biểu đồ Top-5 Xác suất cao nhất")
            top5_classes = [ESC50_CLASSES[i] for i in top5_indices]
            top5_scores = [probs[i] * 100 for i in top5_indices]

            df_top5 = pd.DataFrame({
                "Lớp âm thanh": top5_classes,
                "Xác suất (%)": top5_scores
            })

            fig_bar, ax_bar = plt.subplots(figsize=(6, 3.5))
            bars = ax_bar.barh(df_top5["Lớp âm thanh"][::-1], df_top5["Xác suất (%)"][::-1], color='#1f77b4')
            ax_bar.set_xlabel("Xác suất (%)")
            ax_bar.set_xlim(0, 100)
            ax_bar.set_title("Top-5 Predicted Classes")

            for bar in bars:
                width = bar.get_width()
                ax_bar.text(width + 1, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", ha='left', va='center', fontsize=9)

            fig_bar.tight_layout()
            st.pyplot(fig_bar)

            st.dataframe(df_top5.style.format({"Xác suất (%)": "{:.2f}%"}), use_container_width=True)

else:
    st.info("👋 Hãy chọn và tải lên một file WAV âm thanh 5 giây ở cột bên trái để bắt đầu phân loại.")
# =========================================================
# BƯỚC 6: LỊCH SỬ PHÂN LOẠI TRONG PHIÊN LÀM VIỆC
# =========================================================
st.markdown("---")
st.subheader("📜 Lịch sử Phân loại Âm thanh")

if len(st.session_state.prediction_history) > 0:
    # Chuyển dữ liệu sang DataFrame và ẩn cột kỹ thuật '_sig'
    df_history = pd.DataFrame(st.session_state.prediction_history).drop(columns=["_sig"])

    # Hiển thị bảng lịch sử
    st.dataframe(
        df_history.style.format({"Độ tin cậy (%)": "{:.2f}%"}),
        use_container_width=True
    )

    # Hai nút chức năng: Tải báo cáo CSV & Xóa lịch sử
    hist_col1, hist_col2, _ = st.columns([1.5, 1.2, 4])
    
    with hist_col1:
        csv_data = df_history.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 Tải lịch sử (.CSV)",
            data=csv_data,
            file_name=f"esc50_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with hist_col2:
        if st.button("🗑️ Xóa lịch sử", use_container_width=True):
            st.session_state.prediction_history = []
            st.rerun()
else:
    st.caption("Chưa có lượt phân loại nào được ghi lại trong phiên này.")





# Mã chú ý           Vị trí / Tệp	    Nội dung đang thiếu

# TODO [THIẾU 2]     app/app.py         Hàm wav_to_logmel trong src/data_lib.py
# TODO [THIẾU 3]     checkpoints/       Các tệp trọng số mô hình thật
# TODO [THIẾU 4]     src/models/        Đăng ký đầy đủ CNN và CRNN vào build_model trong __init__.py
# TODO [THIẾU 5]     app/app.py         Xử lý logic cảnh báo khi file checkpoint chưa tồn tại trên đĩa.
# TODO [THIẾU 6]     app/app.py         Thay thế dữ liệu phổ Mel giả np.random.randn() bằng kết quả từ wav_to_logmel
# TODO [THIẾU 7]     app/app.py         Thay mảng xác suất giả np.random.dirichlet() bằng kết quả dự đoán từ mô hình thật
