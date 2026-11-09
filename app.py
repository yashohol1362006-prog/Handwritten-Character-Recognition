
import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image
from datetime import datetime

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="GlyphAI | Character Recognition",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "models/character_model.keras"

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, #202044 0%, transparent 40%),
        radial-gradient(ellipse at 90% 20%, #182b46 0%, transparent 35%),
        #090d18;
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background: #101727;
    border-right: 1px solid #27344a;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

.hero {
    padding: 30px;
    border: 1px solid #33415c;
    border-radius: 22px;
    background: linear-gradient(120deg, #192746, #17172e, #172b40);
    margin-bottom: 25px;
}

.hero-tag {
    color: #a78bfa;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    color: white;
    margin: 10px 0;
}

.hero-subtitle {
    color: #a8b5ca;
    font-size: 15px;
}

.section-title {
    color: white;
    font-size: 21px;
    font-weight: 750;
    margin: 15px 0;
}

.panel {
    background: rgba(22, 32, 52, 0.88);
    border: 1px solid #34415a;
    border-radius: 17px;
    padding: 22px;
    margin-bottom: 15px;
}

.prediction-letter {
    font-size: 100px;
    font-weight: 800;
    text-align: center;
    line-height: 1.2;
    background: linear-gradient(135deg, #a78bfa, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.prediction-caption {
    color: #94a3b8;
    text-align: center;
    font-size: 13px;
    letter-spacing: 2px;
}

.confidence {
    font-size: 30px;
    color: #38bdf8;
    font-weight: 800;
    text-align: center;
}

.small-label {
    color: #94a3b8;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.info-value {
    color: white;
    font-size: 23px;
    font-weight: 750;
    margin-top: 8px;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 25px;
    font-size: 12px;
}

div[data-testid="stFileUploader"] {
    background: rgba(30, 41, 59, 0.6);
    padding: 15px;
    border-radius: 15px;
    border: 1px dashed #64748b;
}

.stButton > button {
    border-radius: 10px;
    background: linear-gradient(90deg, #7c3aed, #2563eb);
    color: white;
    border: none;
    font-weight: 700;
    padding: 12px;
}

.stButton > button:hover {
    border: 1px solid #a78bfa;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD CNN MODEL
# ==========================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.error("CNN model not found. Please run python src/train.py first.")
    st.stop()

model = load_model()


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

def preprocess_image(uploaded_image):

    image = np.array(uploaded_image.convert("L"))

    # Otsu thresholding
    _, threshold = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Determine whether background is light or dark
    border_pixels = np.concatenate([
        threshold[0, :],
        threshold[-1, :],
        threshold[:, 0],
        threshold[:, -1]
    ])

    # Model expects black background and white character
    if np.mean(border_pixels) > 127:
        threshold = cv2.bitwise_not(threshold)

    # Find character
    coords = cv2.findNonZero(threshold)

    if coords is None:
        raise ValueError("No character detected. Upload a clearer image.")

    x, y, w, h = cv2.boundingRect(coords)

    character = threshold[y:y+h, x:x+w]

    # Add padding
    size = max(w, h) + 20

    square = np.zeros((size, size), dtype=np.uint8)

    x_offset = (size - w) // 2
    y_offset = (size - h) // 2

    square[
        y_offset:y_offset+h,
        x_offset:x_offset+w
    ] = character

    # Resize
    resized = cv2.resize(
        square,
        (28, 28),
        interpolation=cv2.INTER_AREA
    )

    # Normalize
    normalized = resized.astype("float32") / 255.0

    # CNN input shape
    normalized = normalized.reshape(1, 28, 28, 1)

    return normalized, resized


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("## ✍️ GlyphAI")

    st.caption("INTELLIGENT CHARACTER RECOGNITION")

    st.divider()

    st.markdown("### Navigation")

    page = st.radio(
        "Select page",
        ["Recognition", "About Project", "Model Information"],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### Technology Stack")

    st.markdown("""
    - 🐍 Python
    - 🧠 TensorFlow / Keras
    - 🔍 OpenCV
    - 📊 NumPy
    - 🌐 Streamlit
    """)

    st.divider()

    st.caption("DEEP LEARNING PROJECT")
    st.caption("EMNIST Letters Dataset")


# ==========================================
# SESSION HISTORY
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []


# ==========================================
# RECOGNITION PAGE
# ==========================================

if page == "Recognition":

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">ARTIFICIAL INTELLIGENCE • COMPUTER VISION</div>
        <div class="hero-title">Handwritten Character Recognition</div>
        <div class="hero-subtitle">
            Transform handwritten alphabets into digital predictions
            using a Convolutional Neural Network.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Statistics
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="panel">
            <div class="small-label">Character Classes</div>
            <div class="info-value">26</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="panel">
            <div class="small-label">Image Dimensions</div>
            <div class="info-value">28 × 28</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="panel">
            <div class="small-label">Model Architecture</div>
            <div class="info-value">CNN</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Upload Your Character</div>',
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drag and drop a handwritten alphabet image",
        type=["png", "jpg", "jpeg"],
        help="Upload a clear image containing one handwritten alphabet."
    )

    if uploaded_file:

        original_image = Image.open(uploaded_file)

        try:
            processed_input, processed_display = preprocess_image(original_image)

            # Predict
            probabilities = model.predict(
                processed_input,
                verbose=0
            )[0]

            predicted_index = int(np.argmax(probabilities))
            predicted_character = chr(predicted_index + 65)

            confidence = float(probabilities[predicted_index]) * 100

            # Save prediction history
            file_key = uploaded_file.name + str(uploaded_file.size)

            if st.session_state.get("last_file") != file_key:

                st.session_state.history.insert(0, {
                    "Character": predicted_character,
                    "Confidence": round(confidence, 2),
                    "File": uploaded_file.name,
                    "Time": datetime.now().strftime("%H:%M:%S")
                })

                st.session_state.last_file = file_key

            st.divider()

            # Images and result
            left, right = st.columns([1, 1])

            with left:

                st.markdown("### Image Analysis")

                img1, img2 = st.columns(2)

                with img1:
                    st.caption("Original")
                    st.image(original_image, width="stretch")

                with img2:
                    st.caption("Processed 28 × 28")
                    st.image(
                        processed_display,
                        width="stretch",
                        clamp=True
                    )

            with right:

                st.markdown("### Prediction Result")

                st.markdown(f"""
                <div class="panel">
                    <div class="prediction-caption">PREDICTED CHARACTER</div>
                    <div class="prediction-letter">{predicted_character}</div>
                    <div class="prediction-caption">CNN CLASSIFICATION</div>
                    <br>
                    <div class="small-label" style="text-align:center">
                        CONFIDENCE SCORE
                    </div>
                    <div class="confidence">{confidence:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

                st.progress(min(confidence / 100, 1.0))

                if confidence < 50:
                    st.warning("Low confidence. Try uploading a clearer image.")
                elif confidence < 80:
                    st.info("Moderate confidence. Check the alternative predictions.")
                else:
                    st.success("Strong model prediction.")

            # Top 5 predictions
            st.markdown('<div class="section-title">Top 5 Predictions</div>',
                        unsafe_allow_html=True)

            top_indices = np.argsort(probabilities)[-5:][::-1]

            top_data = pd.DataFrame({
                "Character": [
                    chr(int(i) + 65) for i in top_indices
                ],
                "Probability (%)": [
                    round(float(probabilities[i]) * 100, 2)
                    for i in top_indices
                ]
            })

            st.dataframe(
                top_data,
                use_container_width=True,
                hide_index=True
            )

            # All class probabilities
            with st.expander("View all 26 character probabilities"):

                probability_data = pd.DataFrame({
                    "Character": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
                    "Probability": probabilities
                })

                st.bar_chart(
                    probability_data,
                    x="Character",
                    y="Probability"
                )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

    else:

        st.info("👆 Upload an image above to start character recognition.")

        st.markdown("### How it works")

        step1, step2, step3 = st.columns(3)

        with step1:
            st.markdown("""
            <div class="panel">
                <h2>📤</h2>
                <b>01. Upload</b>
                <p>Upload a handwritten alphabet image.</p>
            </div>
            """, unsafe_allow_html=True)

        with step2:
            st.markdown("""
            <div class="panel">
                <h2>🧠</h2>
                <b>02. Analyze</b>
                <p>CNN processes the image and extracts features.</p>
            </div>
            """, unsafe_allow_html=True)

        with step3:
            st.markdown("""
            <div class="panel">
                <h2>🎯</h2>
                <b>03. Predict</b>
                <p>Receive the predicted character and confidence.</p>
            </div>
            """, unsafe_allow_html=True)


    # Prediction history
    st.divider()

    st.markdown('<div class="section-title">Recent Predictions</div>',
                unsafe_allow_html=True)

    if st.session_state.history:

        st.dataframe(
            pd.DataFrame(st.session_state.history),
            use_container_width=True,
            hide_index=True
        )

        if st.button("Clear Prediction History"):
            st.session_state.history = []
            st.session_state.last_file = None
            st.rerun()

    else:
        st.caption("Your prediction history will appear here.")


# ==========================================
# ABOUT PAGE
# ==========================================

elif page == "About Project":

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">PROJECT OVERVIEW</div>
        <div class="hero-title">About GlyphAI</div>
        <div class="hero-subtitle">
            An intelligent handwritten alphabet recognition system
            powered by deep learning.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Project Objective")

    st.write("""
    The objective of this project is to identify handwritten
    English characters using image processing and Convolutional
    Neural Networks.
    """)

    st.markdown("### Project Workflow")

    st.markdown("""
    1. Dataset collection using EMNIST Letters.
    2. Image preprocessing and normalization.
    3. Feature extraction using CNN layers.
    4. Model training and validation.
    5. Character classification into 26 classes.
    6. Prediction through an interactive web application.
    """)

    st.markdown("### Applications")

    st.markdown("""
    - Digitization of handwritten documents.
    - Automatic form processing.
    - Educational handwriting tools.
    - Character recognition systems.
    - Document analysis.
    """)


# ==========================================
# MODEL INFORMATION PAGE
# ==========================================

elif page == "Model Information":

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">DEEP LEARNING ARCHITECTURE</div>
        <div class="hero-title">CNN Model Information</div>
        <div class="hero-subtitle">
            Understanding the neural network used for character classification.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Model Architecture")

    st.markdown("""
    | Layer | Configuration |
    |---|---|
    | Input | 28 × 28 × 1 |
    | Conv2D | 32 filters, 3 × 3 |
    | MaxPooling2D | 2 × 2 |
    | Conv2D | 64 filters, 3 × 3 |
    | MaxPooling2D | 2 × 2 |
    | Flatten | Feature vector |
    | Dense | 128 neurons |
    | Dropout | 0.3 |
    | Output | 26 neurons, Softmax |
    """)

    st.markdown("### Training Configuration")

    st.markdown("""
    - Optimizer: Adam
    - Loss: Sparse Categorical Crossentropy
    - Metric: Accuracy
    - Dataset: EMNIST Letters
    - Classes: 26
    """)

    with st.expander("View full model summary"):

        import io

        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + "\n"))

        st.code(stream.getvalue())


# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    GLYPHAI • HANDWRITTEN CHARACTER RECOGNITION<br>
    Powered by Python, TensorFlow, CNN & OpenCV
</div>
""", unsafe_allow_html=True)
