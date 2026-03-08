import streamlit as st
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av

# 1. Page Configuration (Wide Layout)
st.set_page_config(page_title="Age Estimation AI", page_icon="", layout="wide")

# 2. Injecting "Live Objects" (Floating Glowing Orbs in the background)
st.markdown("""
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>
    <div class="orb orb-3"></div>
""", unsafe_allow_html=True)

# 3. Advanced Custom CSS Injection
st.markdown("""
    <style>
    /* Global Background Gradient */
    .stApp {
        background: linear-gradient(-45deg, #0b132b, #11294b, #0a1128, #0b132b);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #E0E6ED;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Live Background Objects (Floating Orbs) */
    .orb {
        position: fixed;
        border-radius: 50%;
        filter: blur(100px);
        z-index: -1;
        opacity: 0.5;
        animation: float 20s infinite ease-in-out alternate;
    }
    .orb-1 { width: 400px; height: 400px; background: #00D2FF; top: -5%; left: -5%; }
    .orb-2 { width: 500px; height: 500px; background: #3A7BD5; bottom: -10%; right: -5%; animation-delay: -5s; }
    .orb-3 { width: 300px; height: 300px; background: #1a2a6c; top: 40%; left: 40%; animation-delay: -10s; }
    
    @keyframes float {
        0% { transform: translate(0, 0) scale(1); }
        100% { transform: translate(50px, -50px) scale(1.2); }
    }

    /* Make the top Streamlit header transparent */
    [data-testid="stHeader"] { background-color: transparent !important; }

    /* Main Header Typography */
    .main-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: -webkit-linear-gradient(45deg, #00D2FF, #3A7BD5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        color: #94A3B8;
        font-size: 1.2rem;
        margin-bottom: 40px;
    }

    /* Frosted Glass Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(11, 19, 43, 0.3) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(0, 210, 255, 0.1);
    }
    
    /* Styled Info Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        border: 1px solid rgba(0, 210, 255, 0.2);
        padding: 20px;
        margin-bottom: 20px;
        border-left: 4px solid #00D2FF;
        color: #E0E6ED;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }

    /* --- FIXING THE WEBRTC CAMERA COMPONENT --- */
    [data-testid="stWebRtc"] video, 
    [data-testid="stWebRtc"] div {
        background-color: transparent !important; 
    }
    
    div[data-testid="stWebRtc"] {
        background: rgba(11, 19, 43, 0.4) !important;
        backdrop-filter: blur(10px);
        border-radius: 15px !important;
        border: 1px solid rgba(0, 210, 255, 0.3) !important;
        padding: 15px;
    }
    
    div[data-testid="stWebRtc"] button {
        background-color: rgba(0, 210, 255, 0.1) !important;
        color: #00D2FF !important;
        border: 1px solid #00D2FF !important;
        border-radius: 8px !important;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
    }
    div[data-testid="stWebRtc"] button:hover {
        background-color: #00D2FF !important;
        color: #0b132b !important;
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.5);
    }
    
    video { border-radius: 10px !important; }
    </style>
""", unsafe_allow_html=True)

# 4. Cleaned-Up Sidebar
with st.sidebar:
    st.markdown("### Age Categories")
    st.markdown("""
    * **Infant:** 0 - 7
    * **Young:** 8 - 17
    * **Adult:** 18 - 34
    * **Middle-aged:** 35 - 59
    * **Senior:** 60+
    """)

# 5. Main Area UI
st.markdown('<div class="main-title">Real-Time Age Estimation AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Powered by Computer Vision and Deep Learning</div>', unsafe_allow_html=True)

# 6. Model Loading (MobileNetV2)
@st.cache_resource
def load_mobile_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None) 
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 5)
    
    # --- UPDATED FILENAME HERE ---
    model_path = 'best_mobilenetv2.pth'
    
    if not os.path.exists(model_path):
        return None, None

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_mobile_model()

# 7. Preprocessing
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

age_classes = ['Infant (0-7)', 'Young (8-17)', 'Adult (18-34)', 'Middle-aged (35-59)', 'Senior (60+)']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Layout Columns
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if model is None:
        # --- UPDATED ERROR MESSAGE HERE ---
        st.error("⚠️ Model file 'best_mobilenetv2.pth' not found in your folder.")
    else:
        st.markdown('<div class="glass-card" style="text-align:center;">Click <b>START</b> to activate the webcam. Ensure your face is clearly lit.</div>', unsafe_allow_html=True)
        
        # 8. WebRTC Live Video Engine
        def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 210, 0), 3) 
                
                face_roi = img[y:y+h, x:x+w]
                face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                img_tensor = data_transforms(face_pil).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    confidence, predicted_idx = torch.max(probabilities, 0)
                    label = f"{age_classes[predicted_idx.item()]} ({confidence.item()*100:.1f}%)"
                    
                cv2.putText(img, label, (x, y-15), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 210, 0), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # The Streamlit WebRTC Component
        webrtc_streamer(
            key="mobilenet-live",
            video_frame_callback=video_frame_callback,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"video": True, "audio": False}
        )