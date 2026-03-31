# ============================================================
# app.py — Classroom Engagement Analyzer
# streamlit run app.py
# ============================================================
import streamlit as st
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, cv2, pickle, tempfile, os, time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import pandas as pd

st.set_page_config(
    page_title="Engagement Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #ffffff;
    color: #0f172a !important;
}
.stApp, .stText, p, span, div.stMarkdown p {
    color: #0f172a !important;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 2.5rem 3rem 3.5rem 3rem !important;
    max-width: 1000px !important;
}

/* ── Sidebar ───────────────────────────────────────────────── */
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stSidebar"] * {
    color: #475569 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Nav buttons in sidebar */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    width: 100% !important;
    text-align: left !important;
    padding: 0.75rem 1rem !important;
    font-size: 0.9rem !important;
    color: #475569 !important;
    font-weight: 500 !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    display: flex !important;
    justify-content: flex-start !important;
    margin-bottom: 0.25rem !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    color: #0f172a !important;
    background: #f1f5f9 !important;
    transform: translateX(4px) !important;
}
[data-testid="stSidebar"] .stButton > button[data-active="true"] {
    color: #2563eb !important;
    background: #eff6ff !important;
    font-weight: 600 !important;
}

/* ── Typography ───────────────────────────────────────────── */
.page-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2.25rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    color: #0f172a;
    margin-bottom: 0.5rem;
    line-height: 1.1;
}
.page-caption {
    font-size: 0.95rem;
    color: #64748b;
    font-weight: 400;
    margin-bottom: 0;
}
.section-label {
    font-family: 'Outfit', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94a3b8;
    margin-bottom: 1rem;
    margin-top: 0;
}

/* ── Divider ──────────────────────────────────────────────── */
.rule { height:1px; background:#e2e8f0; margin: 2rem 0; border-radius: 2px;}

/* ── Result card ──────────────────────────────────────────── */
.result-card {
    padding: 2rem;
    background: #ffffff;
    border-radius: 16px;
    border: 1px solid rgba(226, 232, 240, 0.8);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.025);
    margin-top: 0.5rem;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.result-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.04);
}
.result-level {
    font-family: 'Outfit', sans-serif;
    font-size: 3.5rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.result-pct {
    font-size: 0.85rem;
    font-weight: 600;
    color: #94a3b8;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}
.bar {
    height: 6px;
    background: #f1f5f9;
    border-radius: 6px;
    overflow: hidden;
    margin-bottom: 1.5rem;
}
.bar-inner { 
    height:100%; 
    border-radius:6px;
    transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
}
.result-advice {
    font-size: 0.9rem;
    color: #475569;
    line-height: 1.6;
    border-top: 1px solid #f1f5f9;
    padding-top: 1.25rem;
}

/* ── Upload empty state ───────────────────────────────────── */
.upload-empty {
    border: 2px dashed #cbd5e1;
    border-radius: 12px;
    padding: 3.5rem 2rem;
    text-align: center;
    background: #f8fafc;
    color: #94a3b8;
    font-size: 0.95rem;
    font-weight: 500;
    transition: all 0.3s ease;
}
.upload-empty:hover {
    border-color: #94a3b8;
    background: #f1f5f9;
    color: #64748b;
}

/* ── How it works ─────────────────────────────────────────── */
.step { display:flex; gap:1.2rem; margin-bottom:1.25rem; align-items:flex-start; }
.step-num {
    font-family: 'Outfit', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: #2563eb;
    background: #eff6ff;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
}
.step-text { font-size:0.95rem; color:#475569; line-height:1.5; padding-top: 0.1rem;}

/* ── Scale Container ──────────────────────────────────────────── */
.scale-container {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    margin-bottom: 2rem;
}
.scale-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #ffffff;
    padding: 0.85rem 1.25rem;
    border-radius: 12px;
    border: 1px solid #f1f5f9;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    transition: all 0.2s ease;
}
.scale-row:hover {
    transform: translateX(4px);
    border-color: #e2e8f0;
    box-shadow: 0 4px 8px -1px rgba(0,0,0,0.05);
}
.scale-left {
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.scale-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.scale-lbl {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    color: #0f172a;
    font-size: 0.95rem;
}
.scale-desc {
    font-size: 0.85rem;
    color: #64748b;
    text-align: right;
}

/* ── Notice box ───────────────────────────────────────────── */
.notice {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    font-size: 0.9rem;
    color: #92400e;
    line-height: 1.6;
    display: flex;
    align-items: center;
}

/* ── Live student table ───────────────────────────────────── */
.student-tbl { width:100%; border-collapse:collapse; background: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; }
.student-tbl tr { border-bottom:1px solid #e2e8f0; }
.student-tbl tr:last-child { border-bottom:none; }
.student-tbl td { padding:0.8rem 1rem; font-size:0.9rem; color: #0f172a; }
.student-tbl td:first-child { color:#0f172a; font-weight: 500; }
.student-tbl td:last-child {
    text-align:right;
    font-family:'Outfit',sans-serif;
    color: #0f172a;
    font-weight:600;
}
.live-stat {
    font-family: 'Outfit', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94a3b8;
    margin-top: 1rem;
    text-align: right;
}

/* ── Summary card ─────────────────────────────────────────── */
.sum-card {
    padding:1.5rem;
    background:#ffffff;
    border:1px solid #e2e8f0;
    border-radius:12px;
    box-shadow: 0 2px 4px -1px rgba(0,0,0,0.03);
    text-align: center;
    transition: transform 0.2s ease;
}
.sum-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
}
.sum-level {
    font-family:'Outfit',sans-serif;
    font-size:1.75rem;
    font-weight:700;
    letter-spacing:-0.03em;
    margin-bottom: 0.25rem;
}

/* ── Streamlit overrides ──────────────────────────────────── */
.stButton > button {
    font-family:'Inter',sans-serif !important;
    font-size:0.9rem !important;
    font-weight:600 !important;
    border-radius:8px !important;
    padding: 0.5rem 1rem !important;
    background: #0f172a !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    transition: all 0.2s ease !important;
}
.stButton > button p, .stButton > button span {
    color: white !important;
}
.stButton > button:hover {
    background: #2563eb !important;
    color: white !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 10px -1px rgba(0, 0, 0, 0.15) !important;
}
.stButton > button:hover p, .stButton > button:hover span {
    color: white !important;
}
/* Primary button styling override */
button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%) !important;
    border: none !important;
    box-shadow: 0 4px 10px rgba(37, 99, 235, 0.2) !important;
    color: white !important;
}
button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 14px rgba(37, 99, 235, 0.3) !important;
    color: white !important;
}
[data-testid="stFileUploader"] section {
    border:2px dashed #cbd5e1 !important;
    border-radius:12px !important;
    background:#f8fafc !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #94a3b8 !important;
    background: #f1f5f9 !important;
}

@media (max-width:720px) {
    .block-container { padding:1.5rem !important; }
    .page-title { font-size:1.75rem; }
    .result-level { font-size:2.5rem; }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL
# ============================================================
class TemporalSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        o, _ = self.attn(x, x, x)
        return self.norm(x + o)

class AdaptiveFeatureGate(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d, d//2), nn.ReLU(),
            nn.Linear(d//2, d), nn.Sigmoid())
    def forward(self, x): return x * self.gate(x)

class RecurrentBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, cell_type="lstm"):
        super().__init__()
        cls       = nn.LSTM if cell_type == "lstm" else nn.GRU
        self.rnn  = cls(input_dim, hidden_dim, num_layers=2,
                        batch_first=True, bidirectional=True, dropout=0.3)
        self.attn = TemporalSelfAttention(hidden_dim * 2, num_heads=4)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.fc   = nn.Linear(hidden_dim * 2, 64)
        self.drop = nn.Dropout(0.3)
    def forward(self, x):
        o, _ = self.rnn(x)
        o    = self.norm(self.attn(o)).mean(dim=1)
        return F.relu(self.fc(self.drop(o)))

class HybridImLNBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, input_dim)
        self.gate        = AdaptiveFeatureGate(input_dim)
        self.branch1     = RecurrentBranch(input_dim, hidden_dim, "lstm")
        self.branch2     = RecurrentBranch(input_dim, hidden_dim, "lstm")
        self.branch3     = RecurrentBranch(input_dim, hidden_dim, "gru")
        self.branch4     = RecurrentBranch(input_dim, hidden_dim, "gru")
        self.fusion_attn = nn.Sequential(
            nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))
        self.regressor   = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
    def forward(self, x):
        x = self.gate(F.relu(self.input_proj(x)))
        s = torch.stack([self.branch1(x), self.branch2(x),
                         self.branch3(x), self.branch4(x)], dim=1)
        w = F.softmax(self.fusion_attn(s), dim=1)
        return self.regressor((s * w).sum(dim=1)).squeeze(-1)


# ============================================================
# RESOURCES
# ============================================================
@st.cache_resource
def load_model():
    scaler = pickle.load(open("scaler.pkl", "rb"))
    cal    = pickle.load(open("calibrator.pkl", "rb")) \
             if os.path.exists("calibrator.pkl") else None
    ckpt   = torch.load("best_model.pt", map_location="cpu")
    m      = HybridImLNBiLSTM(input_dim=ckpt["input_dim"])
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m, scaler, cal

@st.cache_resource
def load_detector():
    path = "face_landmarker.task"
    if not os.path.exists(path):
        import urllib.request
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            path)
    opts = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=path),
        num_faces=4,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.4)
    return vision.FaceLandmarker.create_from_options(opts)


# ============================================================
# HELPERS
# ============================================================
def lm_to_feat(lm):
    d = lambda a,b: np.sqrt((lm[a].x-lm[b].x)**2+(lm[a].y-lm[b].y)**2)
    y = lambda a,b: abs(lm[a].y-lm[b].y)
    return np.array([
        y(70,105),y(300,334),y(107,9), y(336,9),
        d(70,107), d(300,336),y(159,145),y(386,374),
        d(33,133), d(362,263),y(61,291), d(61,291),
        y(50,280), y(330,280),y(13,14),  d(13,14),
        d(78,308), y(0,17),   d(61,185), d(291,409),
        abs(lm[1].x-0.5), abs(lm[1].y-0.5)
    ], dtype=np.float32)

def seq_to_feat(seq):
    seq = np.array(seq)
    if len(seq) > 30:
        seq = seq[np.linspace(0, len(seq)-1, 30, dtype=int)]
    d = np.diff(seq, axis=0)
    d = np.vstack([d[:1], d])
    c = np.concatenate([seq, d], axis=1)
    return np.concatenate([c.mean(0), c.std(0)])

def infer(feat, scaler, model, cal):
    x   = torch.tensor(
        scaler.transform(feat.reshape(1,-1)),
        dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        raw = float(model(x).item())
    if cal is not None:
        raw = float(cal.predict([raw])[0])
    return float(np.clip(raw, 0, 1))

LEVELS = [
    (0.80, "High",  "#10b981", "#047857",
     "Highly focused. Students are actively following the session."),
    (0.60, "Good",  "#34d399", "#059669",
     "Solid attention. The class is keeping up well."),
    (0.38, "Fair",  "#fbbf24", "#d97706",
     "Partial attention. A brief pause or question may help."),
    (0.18, "Low",   "#fb923c", "#ea580c",
     "Attention is drifting. Consider switching pace or activity."),
    (0.00, "Poor",  "#ef4444", "#b91c1c",
     "Very low engagement. An intervention is recommended now."),
]

def get_level(score):
    if score is None:
        return ("—", "#e2e8f0", "#94a3b8", "Waiting for data.")
    for min_s, label, bar_c, txt_c, advice in LEVELS:
        if score >= min_s:
            return (label, bar_c, txt_c, advice)
    return ("Poor", "#ef4444", "#b91c1c",
            "Very low engagement detected.")

def hex_to_bgr(h):
    h = h.lstrip("#")
    if len(h) == 3:
        h = h[0]*2 + h[1]*2 + h[2]*2
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return (b, g, r)

def extract_video_feat(path, detector):
    cap, idx, seq = cv2.VideoCapture(path), 0, []
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % 5 == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = detector.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            seq.append(lm_to_feat(res.face_landmarks[0])
                       if res.face_landmarks
                       else np.zeros(22, dtype=np.float32))
        idx += 1
    cap.release()
    return seq_to_feat(seq) if len(seq) >= 2 else None


# ============================================================
# SESSION STATE
# ============================================================
for k, v in [("page","upload"),("running",False),("slog",[])]:
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# TOP NAVIGATION
# ============================================================
col_header, col_nav1, col_nav2 = st.columns([5, 1, 1], gap="small", vertical_alignment="center")

with col_header:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.5rem">
        <div style="font-family:'Outfit',sans-serif;font-size:1.6rem;font-weight:700;color:#0f172a;letter-spacing:-0.03em;line-height:1.1">
            Engagement Analyzer
        </div>
    </div>
    <div style="font-size:0.75rem;color:#64748b;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;margin-top:0.2rem">
        Classroom · CV
    </div>
    """, unsafe_allow_html=True)

with col_nav1:
    if st.button("Upload video", use_container_width=True, key="nav_upload"):
        st.session_state.page = "upload"
        st.rerun()

with col_nav2:
    if st.button("Live webcam", use_container_width=True, key="nav_live"):
        st.session_state.page = "live"
        st.rerun()

st.markdown('<div class="rule" style="margin-top:1.5rem; margin-bottom:2rem"></div>', unsafe_allow_html=True)


# ============================================================
# PAGE — UPLOAD VIDEO
# ============================================================
if st.session_state.page == "upload":

    st.markdown("""
    <div class="page-title">Video analysis</div>
    <div class="page-caption">
        Upload a classroom recording to get an engagement assessment.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>",
                unsafe_allow_html=True)

    # 1. Full-width Uploader / Video Area
    uploaded = st.file_uploader(
        "video", type=["mp4","avi","mov"],
        label_visibility="collapsed"
    )
    
    if uploaded:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
            
        st.video(tmp_path)
        run = st.button("Run analysis", use_container_width=True, type="primary")
    else:
        st.markdown("""
        <div class="upload-empty">
            Drop a video file here<br>
            <span style="font-size:0.72rem">MP4 · AVI · MOV</span>
        </div>
        """, unsafe_allow_html=True)
        run = False

    # 2. Analysis Results Area
    if uploaded and run:
        st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
        try:
            model, scaler, cal = load_model()
            det                = load_detector()
        except FileNotFoundError as e:
            st.error(str(e)); st.stop()

        with st.spinner("Processing..."):
            feat = extract_video_feat(tmp_path, det)

        if feat is None:
            st.markdown("""
            <div class="notice">
                No face detected. Try a clip where student
                faces are clearly visible.
            </div>
            """, unsafe_allow_html=True)
        else:
            score               = infer(feat, scaler, model, cal)
            lbl, bar_c, txt_c, advice = get_level(score)
            pct                 = int(score * 100)

            st.markdown(f"""
            <div class="result-card">
                <div class="section-label">Engagement level</div>
                <div class="result-level" style="color:{txt_c}">
                    {lbl}
                </div>
                <div class="result-pct">{pct}%</div>
                <div class="bar">
                    <div class="bar-inner"
                         style="width:{pct}%;background:{bar_c}">
                    </div>
                </div>
                <div class="result-advice">{advice}</div>
            </div>
            """, unsafe_allow_html=True)

        os.unlink(tmp_path)

    # 3. Information Columns (Underneath Uploader)
    if not uploaded:
        st.markdown("<div style='height:3rem'></div>", unsafe_allow_html=True)
        info_left, info_right = st.columns([1, 1], gap="large")
        
        with info_left:
            st.markdown("""
            <div class="section-label">How it works</div>
            """, unsafe_allow_html=True)
            for num, txt in [
                ("01","Upload a classroom video clip"),
                ("02","Facial movement is analyzed frame by frame"),
                ("03","An engagement level is returned for the clip"),
            ]:
                st.markdown(f"""
                <div class="step">
                    <span class="step-num">{num}</span>
                    <span class="step-text">{txt}</span>
                </div>
                """, unsafe_allow_html=True)

        with info_right:
            st.markdown('<div class="section-label">Scale Reference</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="scale-container">',
                        unsafe_allow_html=True)
            for _, lvl_lbl, color, _, desc in LEVELS:
                short = desc.split(".")[0]
                st.markdown(f"""
                <div class="scale-row">
                  <div class="scale-left">
                    <div class="scale-dot" style="background:{color}; box-shadow: 0 0 10px {color}80;"></div>
                    <span class="scale-lbl">{lvl_lbl}</span>
                  </div>
                  <div class="scale-desc">{short}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# PAGE — LIVE WEBCAM
# ============================================================
elif st.session_state.page == "live":

    st.markdown("""
    <div class="page-title">Live webcam</div>
    <div class="page-caption">
        Real-time engagement monitoring from your webcam.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>",
                unsafe_allow_html=True)

    is_cloud = os.environ.get("STREAMLIT_SHARING_MODE") is not None
    if is_cloud:
        st.markdown("""
        <div class="notice">
            Live webcam is not available on the hosted version —
            the server has no access to a camera.<br><br>
            Run the app locally to use this mode.
            Upload video works fully here.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    try:
        model, scaler, cal = load_model()
        det                = load_detector()
    except FileNotFoundError as e:
        st.error(str(e)); st.stop()

    c1, c2, c3, _ = st.columns([1,1,2,4])
    with c1:
        start = st.button("Start", type="primary",
                          use_container_width=True)
    with c2:
        stop  = st.button("Stop", use_container_width=True)
    with c3:
        every = st.select_slider(
            "freq", [10,15,20,30,45,60], value=20,
            label_visibility="collapsed")

    st.markdown(
        f'<div class="live-stat">'
        f'Updating every {every} frames</div>',
        unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>",
                unsafe_allow_html=True)

    feed_col, data_col = st.columns([1.4,1], gap="large")
    with feed_col:
        frame_win  = st.empty()
    with data_col:
        scores_win = st.empty()
        chart_win  = st.empty()
        export_win = st.empty()

    if start:
        st.session_state.running = True
        st.session_state.slog    = []
    if stop:
        st.session_state.running = False

    if st.session_state.running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.markdown("""
            <div class="notice">
                Could not open webcam. Check your camera is connected
                and Terminal has camera permission in System Settings.
            </div>
            """, unsafe_allow_html=True)
            st.session_state.running = False
            st.stop()

        face_buf    = {i: deque(maxlen=30) for i in range(4)}
        face_scores = {i: None             for i in range(4)}
        slog, fc    = [], 0

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret: break
            fc += 1

            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res  = det.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            nf   = len(res.face_landmarks) if res.face_landmarks else 0
            out  = frame.copy()
            h, w = frame.shape[:2]

            for i in range(nf):
                lm = res.face_landmarks[i]
                face_buf[i].append(lm_to_feat(lm))

                if fc % every == 0 and len(face_buf[i]) >= 2:
                    face_scores[i] = infer(
                        seq_to_feat(list(face_buf[i])),
                        scaler, model, cal)
                    row = {"time": time.strftime("%H:%M:%S"),
                           "frame": fc}
                    for j in range(nf):
                        row[f"s{j+1}"] = (
                            round(face_scores[j],3)
                            if face_scores[j] is not None else None)
                    slog.append(row)

                xs  = [int(p.x*w) for p in lm]
                ys  = [int(p.y*h) for p in lm]
                x1  = max(0, min(xs)-10)
                y1  = max(0, min(ys)-10)
                x2  = min(w, max(xs)+10)
                y2  = min(h, max(ys)+10)
                s   = face_scores[i]
                lbl, bc, _, _ = get_level(s)
                bgr = hex_to_bgr(bc)
                cv2.rectangle(out,(x1,y1),(x2,y2),bgr,2)
                cv2.putText(out, f"S{i+1} {lbl}",
                            (x1,max(y1-7,14)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.46, bgr, 2)

            frame_win.image(
                cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
                channels="RGB", use_container_width=True)

            with scores_win.container():
                if nf == 0:
                    st.markdown(
                        '<p style="font-size:0.8rem;color:#bbb">'
                        'No face in frame</p>',
                        unsafe_allow_html=True)
                else:
                    rows = ""
                    for i in range(nf):
                        s = face_scores[i]
                        lbl, bc, tc, _ = get_level(s)
                        pct = f"{int(s*100)}%" if s else "—"
                        rows += f"""
                        <tr>
                          <td>Student {i+1}</td>
                          <td style="color:{tc}">{lbl}
                            <span style="font-size:0.7rem;color:#ccc;
                                         font-family:'Inter',sans-serif;
                                         margin-left:4px">{pct}</span>
                          </td>
                        </tr>"""
                    st.markdown(
                        f'<table class="student-tbl">{rows}</table>'
                        f'<div class="live-stat">Frame {fc} '
                        f'· {nf} detected</div>',
                        unsafe_allow_html=True)

            if len(slog) >= 2:
                df = pd.DataFrame(slog)
                cs = [c for c in df.columns if c.startswith("s")]
                if cs:
                    dp = df[cs].tail(80).ffill().astype(float)
                    dp.columns = [f"Student {c[1:]}" for c in cs]
                    chart_win.line_chart(
                        dp, use_container_width=True, height=150)

            if slog:
                export_win.download_button(
                    "Download log",
                    pd.DataFrame(slog).to_csv(index=False),
                    "engagement_log.csv", "text/csv",
                    use_container_width=True,
                    key=f"dl_{fc}")

            time.sleep(0.03)

        cap.release()

        if slog:
            st.markdown('<div class="rule"></div>',
                        unsafe_allow_html=True)
            st.markdown(
                '<div class="section-label">Session summary</div>',
                unsafe_allow_html=True)
            df_end = pd.DataFrame(slog)
            scols  = [c for c in df_end.columns if c.startswith("s")]
            cols   = st.columns(max(len(scols),1))
            for i, col in enumerate(scols):
                vals = pd.to_numeric(
                    df_end[col], errors="coerce").dropna()
                if len(vals):
                    avg            = float(vals.mean())
                    lbl, bc, tc, _ = get_level(avg)
                    with cols[i]:
                        st.markdown(f"""
                        <div class="sum-card">
                            <div class="section-label">
                                Student {i+1}
                            </div>
                            <div class="sum-level"
                                 style="color:{tc}">{lbl}</div>
                            <div style="font-size:0.75rem;color:#bbb;
                                        margin-top:0.2rem">
                                {int(avg*100)}% average
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            st.download_button(
                "Download full session report",
                df_end.to_csv(index=False),
                "session_report.csv", "text/csv",
                key="dl_final")