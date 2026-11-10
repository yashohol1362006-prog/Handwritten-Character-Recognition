# ✍️ Handwritten Character Recognition using CNN

An AI-powered handwritten English alphabet recognition system that identifies handwritten characters (A–Z) using **Convolutional Neural Networks (CNN)**, TensorFlow, OpenCV and the EMNIST Letters dataset.

The project provides an interactive web application built with Streamlit, allowing users to upload handwritten character images and receive real-time predictions with confidence scores.
 
---

## 🚀 Project Overview

Handwritten Character Recognition is a Deep Learning and Computer Vision project designed to automatically identify handwritten English alphabets.

The system uses image preprocessing techniques and a CNN model trained on the EMNIST Letters dataset to classify handwritten characters into 26 categories (A–Z).

### 🎯 Objective

- Recognize handwritten English alphabets from images.
- Apply image processing techniques to prepare input images.
- Develop and train a CNN-based classification model.
- Evaluate model performance using accuracy and classification metrics.
- Build an interactive web application for real-time character prediction.

---

## ✨ Key Features

- **26 Character Classification:** Recognizes English alphabets from A to Z.
- **CNN Architecture:** Uses a Convolutional Neural Network for image classification.
- **Image Preprocessing:** Grayscale conversion, thresholding, resizing and normalization.
- **Image Upload:** Upload handwritten character images in PNG, JPG and JPEG formats.
- **Confidence Score:** Displays the model's prediction confidence.
- **Top 5 Predictions:** Shows the five most likely character predictions.
- **Probability Visualization:** Visualizes prediction probabilities for all 26 classes.
- **Prediction History:** Maintains recent predictions during the application session.
- **Interactive Dashboard:** Modern Streamlit-based web interface.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow | Deep Learning framework |
| Keras | CNN model development |
| OpenCV | Image processing |
| NumPy | Numerical computations |
| Pandas | Data handling |
| Matplotlib | Data visualization |
| Scikit-learn | Model evaluation |
| Streamlit | Web application |
| EMNIST | Handwritten character dataset |

---

## 📚 Dataset

**Dataset: EMNIST Letters**

The project uses the EMNIST (Extended MNIST) Letters dataset.

Dataset characteristics:

- 26 English alphabet classes (A–Z).
- Grayscale handwritten character images.
- Image dimensions: 28 × 28 pixels.
- 88,800 training images.
- 14,800 testing images.

Dataset labels are converted from 1–26 into 0–25 for model classification.

Dataset source: [EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)

---

## 🧠 CNN Model Architecture

The project uses a Convolutional Neural Network to extract important visual features from handwritten character images.

### Model Layers

| Layer | Configuration |
|---|---|
| Input Layer | 28 × 28 × 1 |
| Convolutional Layer 1 | 32 filters, 3 × 3 |
| Max Pooling 1 | 2 × 2 |
| Convolutional Layer 2 | 64 filters, 3 × 3 |
| Max Pooling 2 | 2 × 2 |
| Flatten Layer | Feature extraction |
| Dense Layer | 128 neurons |
| Dropout | 0.3 |
| Output Layer | 26 neurons, Softmax |

### Training Configuration

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Evaluation Metric: Accuracy
- Activation Function: ReLU
- Output Activation: Softmax
- Batch Size: 128
- Epochs: 10

---

## ⚙️ Project Workflow

1. Dataset Collection – Load the EMNIST Letters dataset.
2. Image Preprocessing – Resize, normalize and reshape images.
3. Model Development – Build the CNN architecture.
4. Model Training – Train the CNN using handwritten character images.
5. Model Evaluation – Evaluate performance using testing data.
6. Model Saving – Save the trained model in Keras format.
7. Prediction – Process uploaded images and predict characters.
8. Deployment – Run the interactive Streamlit application.

---

## 📂 Project Structure

```text
Handwritten-Character-Recognition/
│
├── dataset/
│   └── dataset.py
│
├── models/
│   ├── character_model.keras
│   └── training_history.npy
│
├── notebooks/
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 💻 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/yashohol1362006-prog/Handwritten-Character-Recognition.git
```

### 2. Navigate to the project directory

```bash
cd Handwritten-Character-Recognition
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

Windows:

```powershell
venv\Scripts\activate
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

---

## ▶️ How to Run the Project

### Step 1: Train the CNN model

```bash
python src/train.py
```

The script loads the EMNIST dataset, preprocesses the images, trains the CNN model, evaluates its performance and saves the trained model inside the `models/` directory.

### Step 2: Run character prediction

```bash
python src/predict.py test.png
```

Replace `test.png` with the path of your handwritten character image.

### Step 3: Launch the web application

```bash
python -m streamlit run app.py
```

The application will open in your default browser.

Upload a handwritten alphabet image to receive the predicted character and confidence score.

---

## 📊 Model Evaluation

The trained model is evaluated using:

- Training Accuracy
- Validation Accuracy
- Testing Accuracy
- Classification Report
- Precision
- Recall
- F1-Score

Actual evaluation results will be added after model training.

---

## 🌐 Future Enhancements

- Handwritten word recognition.
- Full sentence recognition using CRNN and sequence modeling.
- Real-time handwriting recognition using a drawing canvas.
- Improved image preprocessing and character segmentation.
- Support for additional languages and character sets.
- Cloud deployment and API integration.

---

## 🔮 Applications

- Digitization of handwritten documents.
- Automatic form processing.
- Educational handwriting recognition tools.
- Document analysis systems.
- Automated character identification.

---

## 👨‍💻 Developer

**Yash Ohol**

Mechanical Engineering Student | AI & Deep Learning Project

GitHub: [@yashohol1362006-prog](https://github.com/yashohol1362006-prog)

---

## 📜 License

This project is developed for educational and academic purposes.

---

**Built with Python, TensorFlow, CNN, OpenCV and Streamlit.**
