import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Cấu hình giao diện Web
st.set_page_config(page_title="PCB Defect Detection", layout="wide")
st.title("🛠️ Hệ Thống Nhận Diện Lỗi Bo Mạch (PCB)")
st.write("Sử dụng YOLOv8 để phát hiện vết xước, đoản mạch tự động trên dây chuyền.")

# Nạp mô hình (Sử dụng bản .pt gốc cho ổn định trên Web)
@st.cache_resource
def load_model():
    # Đường dẫn trỏ tới file mô hình tốt nhất của YOLOv8n
   return YOLO('best.pt')

model = load_model()

# Tạo nút tải ảnh
uploaded_file = st.file_uploader("Tải ảnh bo mạch cần kiểm tra lên đây (JPG, PNG)", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Đọc ảnh từ Web
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ảnh Gốc")
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("Kết Quả Phân Tích")
        with st.spinner('Đang dùng AI quét lỗi...'):
            # Đưa ảnh qua mô hình YOLO
            results = model(img_array)
            
            # Lấy ảnh đã được vẽ khung lỗi
            res_plotted = results[0].plot()
            # Đổi hệ màu OpenCV BGR sang RGB cho Web
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            st.image(res_rgb, use_column_width=True)
            
            # Thống kê lỗi
            num_defects = len(results[0].boxes)
            if num_defects > 0:
                st.error(f"⚠️ CẢNH BÁO: Phát hiện {num_defects} điểm lỗi trên bo mạch!")
            else:
                st.success("✅ Bo mạch hoàn hảo, không phát hiện lỗi.")
