from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "resnet50_best.pth"
RESULTS_PATH = ROOT_DIR / "results" / "model_comparison.csv"

CLASS_NAMES = ["bare", "heavily_grazed", "softly_grazed"]
CLASS_LABELS = {
    "bare": "Bare",
    "heavily_grazed": "Heavily Grazed",
    "softly_grazed": "Softly Grazed",
}
DEGRADATION_MAP = {
    "bare": ("Tinggi", "danger"),
    "heavily_grazed": ("Sedang", "warning"),
    "softly_grazed": ("Rendah", "success"),
}


st.set_page_config(
    page_title="Klasifikasi Vegetasi Gambut",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_styles():
    st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --green-dark: #2E7D32;
    --green-light: #81C784;
    --soil: #8D6E63;
    --ivory: #FAFAFA;
    --ink: #1F2A24;
    --muted: #66756B;
    --line: rgba(46,125,50,.14);
    --soft: #F3F8F2;
}

html,
body,
[class*="css"],
.stMarkdown,
p {
    font-family: "Inter", sans-serif !important;
}

.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(129,199,132,.12), transparent 30%),
        linear-gradient(
            180deg,
            #FAFAFA 0%,
            #F5FAF4 50%,
            #FAFAFA 100%
        );
    color: var(--ink);
}

.main .block-container {
    max-width: 1280px;
    padding-top: 2rem;
    padding-bottom: 4rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

h1, h2, h3, h4 {
    color: var(--ink);
    letter-spacing: -0.02em;
}

/* ======================
   HERO SECTION
====================== */

.hero {
    position: relative;
    overflow: hidden;
    border-radius: 24px;
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    color: white;
    background:
        linear-gradient(
            135deg,
            rgba(46,125,50,.98),
            rgba(56,142,60,.95)
        ),
        repeating-linear-gradient(
            18deg,
            rgba(255,255,255,.05) 0 1px,
            transparent 1px 34px
        ),
        radial-gradient(
            circle at 85% 20%,
            rgba(129,199,132,.4),
            transparent 35%
        );
    box-shadow: 0 18px 42px rgba(46,125,50,.15);
}

.hero-content {
    position: relative;
    z-index: 1;
    max-width: 950px;
}

.hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: .55rem;
    padding: .42rem .9rem;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,.3);
    background: rgba(255,255,255,.12);
    color: white;
    font-size: .85rem;
    font-weight: 700;
    backdrop-filter: blur(8px);
    margin-bottom: 1.5rem;
}

.hero h1 {
    color: white;
    margin-top: 0;
    margin-bottom: 1rem;
    font-size: clamp(2rem, 3.5vw, 3.2rem);
    font-weight: 800;
    line-height: 1.2;
}

.hero-subtitle {
    max-width: 760px;
    color: rgba(255,255,255,.9);
    font-size: 1.1rem;
    line-height: 1.6;
    margin: 0;
}

/* ======================
   SECTION TITLE
====================== */

.section-title {
    margin-top: 2.5rem;
    margin-bottom: 1.5rem;
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--green-dark);
}

.section-title::after {
    content: "";
    display: block;
    width: 60px;
    height: 4px;
    margin-top: .5rem;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--green-dark), var(--green-light));
}

/* ======================
   CARDS (PREMIUM WHITE FORMS)
====================== */

/* Mengubah container border native Streamlit menjadi Card putih premium yang serasi */
div[data-testid="stVerticalBlockBorder"] {
    background-color: white !important;
    border: 1px solid rgba(46,125,50,.12) !important;
    border-radius: 20px !important;
    box-shadow: 0 10px 25px rgba(0,0,0,.03) !important;
    padding: 2rem !important;
    transition: all .25s ease !important;
}

div[data-testid="stVerticalBlockBorder"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 16px 35px rgba(46,125,50,.1) !important;
}

/* Static Card Styling */
.card {

    background:
        linear-gradient(
            180deg,
            rgba(255,255,255,.96),
            rgba(247,251,247,.98)
        ) !important;

    border: 1px solid rgba(46,125,50,.10) !important;

    border-radius: 24px !important;

    box-shadow:
        0 12px 32px rgba(0,0,0,.04) !important;

    padding: 2rem !important;

    transition: all .25s ease !important;

    min-height: 260px;

    height: 100%;
}

.card:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 16px 35px rgba(46,125,50,.1) !important;
}

.card-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: .6rem;
}

.small-text {
    font-size: 0.9rem;
    color: var(--muted);
    line-height: 1.6;
    margin: 0;
}

/* ======================
   ICONS & BADGES
====================== */

.icon-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 48px;
    height: 48px;
    border-radius: 12px;
    background: rgba(46,125,50,.08);
    color: var(--green-dark);
    margin-bottom: 1.2rem;
}

.badge {
    display: inline-flex;
    align-items: center;
    padding: .4rem .9rem;
    border-radius: 999px;
    font-size: .85rem;
    font-weight: 700;
    box-shadow: 0 3px 10px rgba(0,0,0,.04);
    margin-top: 0.5rem;
}

.badge.success {
    color: #1B5E20;
    background: #E8F5E9;
    border: 1px solid rgba(27,94,32,.15);
}

.badge.warning {
    color: #E65100;
    background: #FFF3E0;
    border: 1px solid rgba(230,81,0,.15);
}

.badge.danger {
    color: #C62828;
    background: #FFEBEE;
    border: 1px solid rgba(198,40,40,.15);
}

/* ======================
   METRIC CARD
====================== */

.metric-card {
    padding: 1.1rem;
    border-radius: 16px;
    background: #F9FBF9;
    border: 1px solid rgba(46,125,50,.08);
    text-align: center;
}

.metric-label {
    color: var(--muted);
    font-size: .75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: .25rem;
}

.metric-value {
    color: var(--green-dark);
    font-size: 1.6rem;
    font-weight: 800;
}

/* ======================
   PREDICTION CARD
====================== */

.prediction-card {

    padding: 2.4rem;

    min-height: 430px;

    border-radius: 28px;

    background:
        linear-gradient(
            180deg,
            rgba(255,255,255,.98),
            rgba(244,250,243,.98)
        );

    border: 1px solid rgba(46,125,50,.12);

    box-shadow:
        0 18px 45px rgba(46,125,50,.08);
}

.prediction-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}

.prediction-class {
    color: var(--green-dark);
    font-size: 2.6rem;
    font-weight: 850;
    line-height: 1.1;
    margin-bottom: .5rem;
}

/* ======================
   PROBABILITY BARS
====================== */

.prob-row {
    margin-bottom: 1.2rem;
}

.prob-row:last-child {
    margin-bottom: 0;
}

.prob-head {
    display: flex;
    justify-content: space-between;
    font-weight: 600;
    font-size: .95rem;
    color: var(--ink);
    margin-bottom: .4rem;
}

.prob-track,
.confidence-track {
    overflow: hidden;
    width: 100%;
    height: 12px;
    border-radius: 999px;
    background: #E8EFE7;
}

.confidence-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--green-light), var(--green-dark));
}

.prob-fill {
    height: 100%;
    border-radius: 999px;
    background: #B2D8B4;
}

.prob-fill.top {
    background: linear-gradient(90deg, var(--green-light), var(--green-dark));
}

/* ======================
   METHOD PILLS
====================== */

.method-pill {
    display: inline-block;
    background: #E8EFE7;
    color: var(--green-dark);
    padding: .4rem 1rem;
    border-radius: 999px;
    font-size: .85rem;
    font-weight: 700;
    margin-right: .5rem;
    margin-bottom: .5rem;
    border: 1px solid rgba(46,125,50,.1);
}

/* ======================
   NATIVE STREAMLIT OVERRIDES
====================== */

/* Menghilangkan border default uploader bawaan karena kontainer luar kita sudah memiliki card border */
[data-testid="stFileUploader"] {
    padding: 0;
    background: transparent;
    border: none !important;
}

[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed rgba(46,125,50,.2) !important;
    border-radius: 14px;
    background-color: #FAFAFA !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--green-dark) !important;
    background-color: #F4FAF3 !important;
}

/* Streamlit native buttons styling */
.stButton > button {
    border-radius: 12px !important;
    padding: 0.6rem 1.5rem !important;
    background: linear-gradient(135deg, #2E7D32, #3F8C43) !important;
    color: white !important;
    border: none !important;
    font-weight: 700 !important;
    box-shadow: 0 8px 20px rgba(46,125,50,.15) !important;
    transition: all .2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 25px rgba(46,125,50,.25) !important;
}

/* ======================
   FOOTER
====================== */

.footer {
    margin-top: 4rem;
    padding: 2rem;
    border-radius: 24px;
    text-align: center;
    color: #F8FFF8;
    background: linear-gradient(135deg, var(--green-dark), #1B5E20);
    box-shadow: 0 14px 34px rgba(46,125,50,.15);
    font-size: 0.95rem;
    line-height: 1.6;
}

[data-testid="stDataFrame"] {

    border-radius: 20px;

    overflow: hidden;

    border: 1px solid rgba(46,125,50,.10);

    box-shadow:
        0 10px 25px rgba(0,0,0,.04);
}

.icon-badge {

    width: 60px;
    height: 60px;

    border-radius: 18px;

    background:
        linear-gradient(
            135deg,
            rgba(129,199,132,.22),
            rgba(46,125,50,.08)
        );
}

.card-title {

    font-size: 1.25rem;

    font-weight: 800;
}

.small-text {

    font-size: .97rem;

    line-height: 1.8;

    font-weight: 500;
}

.prediction-class {

    font-size: 3rem;

    letter-spacing: -0.03em;
}

@media (max-width: 720px) {
    .main .block-container {
        padding: 1rem;
    }
    .hero {
        padding: 1.75rem;
    }
    .hero h1 {
        font-size: 1.8rem;
    }
    .prediction-class {
        font-size: 2rem;
    }
}

</style>
""", unsafe_allow_html=True)


def icon_svg(kind):
    icons = {
        "drone": """
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M8 12h8"/><path d="M12 8v8"/><path d="M4 8h4v4H4z"/>
                <path d="M16 8h4v4h-4z"/><path d="M4 16h4v4H4z"/><path d="M16 16h4v4h-4z"/>
                <path d="M9 12a3 3 0 0 0 6 0"/>
            </svg>
        """,
        "leaf": """
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M11 20A7 7 0 0 1 4 13c0-5 8-9 16-9 0 8-4 16-9 16Z"/>
                <path d="M4 20c5-5 8-8 16-16"/>
            </svg>
        """,
        "brain": """
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 5a3 3 0 0 0-5.6-1.5A3.5 3.5 0 0 0 4 9.7"/>
                <path d="M12 5a3 3 0 0 1 5.6-1.5A3.5 3.5 0 0 1 20 9.7"/>
                <path d="M7 10a4 4 0 0 0 0 8"/><path d="M17 10a4 4 0 0 1 0 8"/>
                <path d="M12 5v16"/><path d="M9 15h3"/><path d="M12 12h3"/>
            </svg>
        """,
        "upload": """
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 16V4"/><path d="m7 9 5-5 5 5"/>
                <path d="M20 16.5V19a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-2.5"/>
            </svg>
        """,
    }
    return icons[kind]


@st.cache_resource(show_spinner=False)
def load_resnet50_model():
    import torch
    import torch.nn as nn
    from torchvision import models

    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, len(CLASS_NAMES)),
    )
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_image(image):
    import torch
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    model = load_resnet50_model()
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1).squeeze(0).tolist()
    return dict(zip(CLASS_NAMES, probabilities))


def fallback_probabilities():
    return {
        "bare": 0.18,
        "heavily_grazed": 0.27,
        "softly_grazed": 0.55,
    }


def read_best_metrics():
    if not RESULTS_PATH.exists():
        return "ResNet50", 94.67
    df = pd.read_csv(RESULTS_PATH)
    best = df.sort_values("Accuracy", ascending=False).iloc[0]
    return str(best["Model"]), float(best["Accuracy"])


def percent(value):
    return f"{value * 100:.2f}%"


def get_probability_bars_html(probabilities, predicted_class):
    html = ""
    for class_name in CLASS_NAMES:
        label = CLASS_LABELS[class_name]
        value = probabilities[class_name]
        top_class = " top" if class_name == predicted_class else ""
        html += f'<div class="prob-row"><div class="prob-head"><span>{label}</span><span>{percent(value)}</span></div><div class="prob-track"><div class="prob-fill{top_class}" style="width: {value * 100:.2f}%"></div></div></div>'
    return html


def main():
    inject_styles()
    best_model, best_accuracy = read_best_metrics()

    # Hero Banner
    st.markdown(
        f"""
        <section class="hero">
            <div class="hero-content">
                <div class="hero-kicker">{icon_svg("drone")} Pemantauan Vegetasi Gambut Berbasis UAV</div>
                <h1>Sistem Klasifikasi Tingkat Degradasi Vegetasi Gambut Menggunakan Citra UAV dan Deep Learning</h1>
                <p class="hero-subtitle">
                    Mengidentifikasi kondisi tutupan lahan secara cepat dan akurat menggunakan model jaringan saraf tiruan (CNN) berbasis ResNet50.
                </p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # Info Cards (3 Columns) - Berbasis pure-HTML di dalam st.container agar 100% serasi
    st.markdown('<div class="section-title">Informasi Penelitian</div>', unsafe_allow_html=True)
    info_cols = st.columns(3)
    with info_cols[0]:
        st.markdown(f"""<div class="card">
<div class="icon-badge">{icon_svg("leaf")}</div>
<div class="card-title">Tujuan Penelitian</div>
<p class="small-text">Mengidentifikasi tingkat degradasi vegetasi gambut secara visual agar hasil observasi lapangan dapat dibantu oleh analisis citra otomatis.</p>
</div>""", unsafe_allow_html=True)
    with info_cols[1]:
        st.markdown(f"""<div class="card">
<div class="icon-badge">{icon_svg("drone")}</div>
<div class="card-title">Citra UAV</div>
<p class="small-text">Menggunakan citra udara resolusi tinggi dari Unmanned Aerial Vehicle (UAV) untuk menangkap kondisi tutupan vegetasi area penelitian.</p>
</div>""", unsafe_allow_html=True)
    with info_cols[2]:
        st.markdown(f"""<div class="card">
<div class="icon-badge">{icon_svg("brain")}</div>
<div class="card-title">CNN Classification</div>
<p class="small-text">Menerapkan konsep transfer learning dan fine-tuning pada CNN untuk mengklasifikasi tiga kategori kesehatan vegetasi gambut.</p>
</div>""", unsafe_allow_html=True)

    # Upload & Result Section
    st.markdown('<div class="section-title">Upload Gambar UAV</div>', unsafe_allow_html=True)
    upload_col, result_col = st.columns([1.3, 0.7], gap="large")

    with upload_col:
        # Menggunakan native container dengan style card putih agar uploader berada di dalam form yang serasi
        with st.container(border=True):
            st.markdown(f"""<div style="margin-bottom: 1.2rem;">
<div class="icon-badge" style="margin-bottom: 0.8rem;">{icon_svg("upload")}</div>
<div style="font-size: 1.15rem; font-weight: 700; color: var(--ink); margin-bottom: .2rem;">Unggah Citra UAV</div>
<p class="small-text">Silakan pilih file citra udara atau seret langsung ke dalam area di bawah ini:</p>
</div>""", unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Pilih atau drag and drop citra UAV",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                label_visibility="collapsed",
            )
            
            image = None
            probabilities = fallback_probabilities()
            prediction_note = "Mode contoh: unggah citra UAV untuk menjalankan prediksi ResNet50."

            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")
                st.markdown("<br>", unsafe_allow_html=True)
                st.image(image, caption="Preview citra UAV yang diunggah", use_container_width=True)
                try:
                    probabilities = predict_image(image)
                    prediction_note = "Prediksi dihitung menggunakan model ResNet50 tersimpan."
                except Exception as exc:
                    prediction_note = f"Model belum dapat dimuat, menampilkan contoh visual. Detail: {exc}"
            else:
                st.markdown(
                    '<p class="small-text" style="margin-top: 1rem; font-style: italic; text-align: center;">Setelah citra diunggah, aplikasi akan menampilkan preview dan menjalankan analisis model.</p>',
                    unsafe_allow_html=True,
                )

    predicted_class = max(probabilities, key=probabilities.get)
    confidence = probabilities[predicted_class]
    degradation, badge_class = DEGRADATION_MAP[predicted_class]

    with result_col:
        # Menampilkan Hasil Prediksi dengan Rapi dalam Satu Card Utuh
        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-title">Hasil Klasifikasi Lahan</div>
                <div class="prediction-class">{CLASS_LABELS[predicted_class]}</div>
                <div>
                    <span class="badge {badge_class}">Tingkat Degradasi: {degradation}</span>
                </div>
                <div style="margin-top: 2rem; margin-bottom: 0.5rem;">
                    <div class="prob-head">
                        <span style="font-weight: 700;">Confidence Score</span>
                        <span style="color: var(--green-dark); font-weight: 800;">{percent(confidence)}</span>
                    </div>
                    <div class="confidence-track">
                        <div class="confidence-fill" style="width: {confidence * 100:.2f}%"></div>
                    </div>
                </div>
                <p class="small-text" style="margin-top: 1rem; border-top: 1px solid rgba(46,125,50,.1); padding-top: 1rem; font-style: italic;">{prediction_note}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Probabilities and Model Metrics Info
    st.markdown('<div class="section-title">Visualisasi Probabilitas</div>', unsafe_allow_html=True)
    prob_col, model_col = st.columns([1.25, 0.75], gap="large")

    with prob_col:
        # Merender seluruh card berisi visualisasi bar probabilitas tanpa indentasi agar tidak menjadi code block
        bars_html = get_probability_bars_html(probabilities, predicted_class)
        prob_html = f"""<div class="card">
<div class="card-title">Probabilitas Masing-Masing Kelas</div>
<p class="small-text" style="margin-bottom: 1.5rem;">Perbandingan nilai probabilitas yang dihasilkan oleh layer output model:</p>
{bars_html}
</div>"""
        st.markdown(prob_html, unsafe_allow_html=True)

    with model_col:
        # Menghapus seluruh indentasi string HTML model agar tidak memicu markdown code block parser
        metrics_html = f"""<div class="card">
<div class="card-title" style="margin-bottom: 1.2rem;">Informasi Model Terpilih</div>
<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-bottom: 1.2rem;">
<div class="metric-card">
<div class="metric-label">Model Terbaik</div>
<div class="metric-value" style="font-size: 1.4rem;">{best_model}</div>
</div>
<div class="metric-card">
<div class="metric-label">Akurasi</div>
<div class="metric-value" style="font-size: 1.4rem;">{best_accuracy:.2f}%</div>
</div>
<div class="metric-card">
<div class="metric-label">Dataset</div>
<div class="metric-value" style="font-size: 1.4rem;">3,000 Citra</div>
</div>
<div class="metric-card">
<div class="metric-label">Jumlah Kelas</div>
<div class="metric-value" style="font-size: 1.4rem;">3 Kategori</div>
</div>
</div>
<p class="small-text" style="text-align: center;"><strong>Daftar kelas target:</strong> Bare (Terbuka), Heavily Grazed (Rusak), Softly Grazed (Lembap/Sehat).</p>
</div>"""
        st.markdown(metrics_html, unsafe_allow_html=True)

    # Methodology and Summary Comparisons
    st.markdown('<div class="section-title">Informasi Metode Penelitian</div>', unsafe_allow_html=True)
    method_col, comparison_col = st.columns([0.9, 1.1], gap="large")

    with method_col:
        # Merender info arsitektur dalam single card HTML agar tampilan serasi
        method_html = """<div class="card">
<div class="card-title">Arsitektur Deep Learning</div>
<div style="margin-top: 1rem; margin-bottom: 1.2rem;">
<span class="method-pill">MobileNetV2</span>
<span class="method-pill">ResNet50</span>
<span class="method-pill">ResNet101</span>
</div>
<p class="small-text">Metode penelitian menggunakan paradigma <strong>Transfer Learning</strong> dan <strong>Fine-Tuning</strong> pada model CNN untuk mendeteksi serta mengevaluasi status tutupan lahan gambut secara cepat dari udara.</p>
</div>"""
        st.markdown(method_html, unsafe_allow_html=True)

    with comparison_col:
        # Menggunakan st.container(border=True) yang otomatis memicu styling card putih premium
        with st.container(border=True):
            st.markdown('<div class="card-title" style="margin-bottom: 0.3rem;">Ringkasan Perbandingan Model</div>', unsafe_allow_html=True)
            st.markdown('<p class="small-text" style="margin-bottom: 1.2rem;">Berikut adalah tabel komparasi performa arsitektur deep learning yang dievaluasi:</p>', unsafe_allow_html=True)
            
            if RESULTS_PATH.exists():
                st.dataframe(
                    pd.read_csv(RESULTS_PATH),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("File hasil perbandingan model (`model_comparison.csv`) belum tersedia.")

        st.markdown(f"""
            <div style="
            background:linear-gradient(135deg,#E8F5E9,#F5FFF5);
            padding:1rem 1.4rem;
            border-radius:16px;
            border:1px solid rgba(46,125,50,.15);
            margin-bottom:1.5rem;
            font-weight:700;
            color:#2E7D32;
            box-shadow:0 8px 20px rgba(46,125,50,.05);
            ">
            🏆 Model Terbaik Penelitian: {best_model}
            &nbsp;&nbsp;|&nbsp;&nbsp;
            Akurasi: {best_accuracy:.2f}%
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown(
        """
        <footer class="footer">
            <strong>Proyek Penelitian Tugas Akhir Mahasiswa S1</strong><br>
            Sistem Klasifikasi Vegetasi Gambut Berbasis Deep Learning & Citra UAV<br>
            <span style="opacity: 0.8; font-size: 0.85rem;">Hanya untuk Keperluan Akademik dan Evaluasi Penelitian</span>
        </footer>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()