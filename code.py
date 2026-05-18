import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="Stress Level Predictor", page_icon="🧠", layout="centered")

# --- HÀM TRAIN MÔ HÌNH (CHỈ CHẠY 1 LẦN DUY NHẤT NHỜ @st.cache_resource) ---
@st.cache_resource
def train_perceptron_model():
    try:
        # 1. Đọc dữ liệu từ file csv nằm cùng thư mục
        data = pd.read_csv('stresslevel.csv')
        
        # 2. Phân tách Feature (X) và Target (y)
        x = data.loc[:, ['Hours of work/day', 'Hours of sleep', 'Cups of coffee/day', 'Age', 'Screen time', 'Anxiety score']]
        y = data['Stress level']
        
        # 3. Khởi tạo và Fit bộ chuẩn hóa dữ liệu
        scaler = StandardScaler()
        x_std = scaler.fit_transform(x)
        
        # 4. Khởi tạo và Train mô hình Perceptron
        model = Perceptron(max_iter=1000, eta0=0.05, random_state=42)
        model.fit(x_std, y)
        
        return model, scaler
    except FileNotFoundError:
        return None, None

# --- KHỞI CHẠY TRAIN / TẢI MÔ HÌNH VÀO APP ---
model, scaler = train_perceptron_model()

if model is None or scaler is None:
    st.error("❌ Không tìm thấy file `stresslevel.csv`. Bạn hãy đảm bảo đã upload file dữ liệu này lên cùng thư mục với file `app.py` trên GitHub nhé!")
    st.stop()


# --- GIAO DIỆN ỨNG DỤNG STREAMLIT ---
st.title("🧠 Ứng Dụng Dự Đoán Mức Độ Căng Thẳng")
st.write("Nhập các chỉ số sinh hoạt hàng ngày của bạn để mô hình Perceptron dự đoán mức độ stress.")
st.markdown("---")

# Chia giao diện làm 2 cột cho gọn gàng
col1, col2 = st.columns(2)

with col1:
    hours_of_work = st.slider("Số giờ làm việc / ngày", min_value=0.0, max_value=16.0, value=8.0, step=0.5)
    hours_of_sleep = st.slider("Số giờ ngủ / đêm", min_value=0.0, max_value=12.0, value=7.0, step=0.5)
    cups_of_coffee = st.number_input("Số ly cà phê / ngày", min_value=0, max_value=10, value=2, step=1)

with col2:
    age = st.number_input("Tuổi của bạn", min_value=15, max_value=80, value=25, step=1)
    screen_time = st.slider("Thời gian xem màn hình (giờ/ngày)", min_value=0.0, max_value=18.0, value=6.0, step=0.5)
    anxiety_score = st.slider("Điểm số lo âu (Anxiety Score)", min_value=0, max_value=100, value=45, step=1)

st.markdown("---")


# --- XỬ LÝ DỰ ĐOÁN KHI BẤM NÚT ---
if st.button("📊 Dự Đoán Ngay", type="primary"):
    # 1. Thu thập dữ liệu từ giao diện thành mảng 2D
    input_data = np.array([[hours_of_work, hours_of_sleep, cups_of_coffee, age, screen_time, anxiety_score]])
    
    # 2. Chuẩn hóa dữ liệu đầu vào bằng Scaler đã học lúc khởi động app
    input_data_scaled = scaler.transform(input_data)
    
    # 3. Dự đoán nhãn số (0, 1 hoặc 2)
    prediction = model.predict(input_data_scaled)[0]
    
    # 4. Ánh xạ kết quả từ Float/Int sang String để hiển thị
    if prediction == 0 or prediction == 0.0:
        st.success("### Kết quả: **Relaxed (Thư giãn)** 🎉")
        st.balloons()
    elif prediction == 1 or prediction == 1.0:
        st.info("### Kết quả: **Normal (Bình thường)** ☕")
    elif prediction == 2 or prediction == 2.0:
        st.error("### Kết quả: **Stressful (Căng thẳng)** ⚠️")
        st.warning("💡 Lời khuyên: Hãy dành thời gian nghỉ ngơi, giảm bớt caffeine và ngủ đủ giấc bạn nhé!")