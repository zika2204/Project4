import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Stress Sense - Predictor", 
    page_icon="🌿", 
    layout="centered"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #F4F7F5;
    }
    .main-title {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #2C4A3E; 
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
        letter-spacing: -0.5px;
    }
    .sub-title {
        color: #607266;
        font-size: 16px;
        text-align: center;
        margin-bottom: 30px;
    }
    .stAlert {
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    }
    div.stButton > button:first-child {
        background-color: #4A7C59 !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 30px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        width: 100% !important;
        box-shadow: 0 4px 10px rgba(74, 124, 89, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #3B6347 !important;
        box-shadow: 0 6px 15px rgba(74, 124, 89, 0.3) !important;
        transform: translateY(-1px);
    }
    div[data-baseweb="slider"] {
        padding-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def train_perceptron_model():
    try:
        data = pd.read_csv('stresslevel.csv')
        x = data.loc[:, ['Hours of work/day', 'Hours of sleep', 'Cups of coffee/day', 'Age', 'Screen time', 'Anxiety score']]
        y = data['Stress level']
        scaler = StandardScaler()
        x_std = scaler.fit_transform(x)
        model = Perceptron(max_iter=1000, eta0=0.05, random_state=42)
        model.fit(x_std, y)
        return model, scaler
    except FileNotFoundError:
        return None, None
model, scaler = train_perceptron_model()

if model is None or scaler is None:
    st.error("❌ Không tìm thấy file `stresslevel.csv`. Bạn hãy đảm bảo đã upload file dữ liệu này lên cùng thư mục với file `app.py` trên GitHub nhé!")
    st.stop()

st.markdown('<div class="main-title">🌿 Stress Sense</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Lắng nghe cơ thể — Dự đoán và cân bằng mức độ căng thẳng bằng AI</div>', unsafe_allow_html=True)

with st.container():
    # Chia giao diện làm 2 cột
    col1, col2 = st.columns(2)

    with col1:
        hours_of_work = st.slider("Số giờ làm việc / ngày", min_value=0.0, max_value=16.0, value=8.0, step=0.5)
        hours_of_sleep = st.slider("Số giờ ngủ / đêm", min_value=0.0, max_value=12.0, value=7.0, step=0.5)
        cups_of_coffee = st.number_input("Số ly cà phê / ngày", min_value=0, max_value=10, value=2, step=1)

    with col2:
        age = st.number_input("Tuổi của bạn", min_value=15, max_value=80, value=25, step=1)
        screen_time = st.slider("Thời gian xem màn hình (giờ/ngày)", min_value=0.0, max_value=18.0, value=6.0, step=0.5)
        anxiety_score = st.slider("Điểm số lo âu (Anxiety Score)", min_value=0, max_value=100, value=45, step=1)

st.markdown("<br>", unsafe_allow_html=True) # Tạo một khoảng cách nhỏ tạo độ thoáng

if st.button("📊 Dự Đoán Ngay", type="primary"):
    input_data = np.array([[hours_of_work, hours_of_sleep, cups_of_coffee, age, screen_time, anxiety_score]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)[0]
    
    st.markdown("### Kết quả phân tích từ Stress Sense:")
    if prediction == 0 or prediction == 0.0:
        st.success("### Mức độ: **Relaxed (Thư giãn)** 🎉")
        st.balloons()
    elif prediction == 1 or prediction == 1.0:
        st.info("### Mức độ: **Normal (Bình thường)** ☕")
    elif prediction == 2 or prediction == 2.0:
        st.error("### Mức độ: **Stressful (Căng thẳng)** ⚠️")
        st.warning("💡 **Lời khuyên từ Stress Sense:** Chỉ số của bạn đang ở mức cảnh báo. Hãy tạm gác lại công việc, dành thời gian nghỉ ngơi, giảm bớt lượng caffeine nạp vào và cố gắng ngủ đủ giấc hơn bạn nhé!")
