# Gender_Classification-ai-ml-model
# Gender Classification Using MobileNetV2

## Project Overview

This project is a Deep Learning-based image classification application that predicts whether an uploaded facial image belongs to the Male or Female class.

The model is developed using **MobileNetV2 Transfer Learning**, a pretrained Convolutional Neural Network (CNN), and integrated into an interactive **Streamlit web application**.

The trained model is included in this repository, so users can run the application without downloading the original dataset or training the model again.

## Features

- **MobileNetV2 Architecture:** Uses a pretrained CNN model with ImageNet weights.
- **Transfer Learning:** Reuses pretrained features with custom classification layers.
- **Image Preprocessing:** Resizes images to 224 × 224 pixels.
- **Data Augmentation:** Uses rotation, zoom, and horizontal flipping during training.
- **Binary Classification:** Predicts Male or Female dataset classes.
- **Interactive Web Interface:** Upload images through a Streamlit application.
- **Confidence Score:** Displays the prediction confidence.

## Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- Deep Learning
- Convolutional Neural Networks (CNN)
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit

## Dataset Information

The model was trained using the Gender Classification Dataset from Kaggle.

**Dataset Link:** https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset

| Dataset | Female Images | Male Images |
|---|---:|---:|
| Training | 23,243 | 23,766 |
| Validation | 5,841 | 5,808 |

**Note:** The dataset is not included in this repository because the model has already been trained. The saved model is sufficient to run the application. Downloading the dataset is only necessary if you want to retrain or fine-tune the model.

## Model Architecture

MobileNetV2 is used as the pretrained base model with ImageNet weights.

The custom classification head consists of:

- Global Average Pooling
- Dense layer with 128 neurons and ReLU activation
- Dropout layer (0.3)
- Output layer with one neuron and Sigmoid activation

The model uses Binary Crossentropy as its loss function and Adam as its optimizer.

## Model Performance

The model was trained for 5 epochs.

| Metric | Result |
|---|---:|
| Training Accuracy | 89.54% |
| Validation Accuracy | 89.13% |
| Validation Loss | 0.2731 |

The model was also evaluated using a confusion matrix, precision, recall, and F1-score.

## Project Structure

```text
Gender_Classification/
│
├── notebooks/
│   └── Gender_Classification.ipynb
│
├── models/
│   └── gender_classification_mobilenetv2.keras
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run This Project on Your PC

Follow these steps to run the application on your computer.

### Step 1: Download the Project

Clone the GitHub repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Alternatively, download the repository as a ZIP file from GitHub and extract it.

### Step 2: Open the Project Folder

```bash
cd Gender_Classification
```

### Step 3: Install Python

Install a compatible Python version (Python 3.12 is recommended).

Download Python from:

https://www.python.org/downloads/

### Step 4: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### Step 5: Install Required Libraries

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
python -m streamlit run app.py
```

Streamlit will start the application and provide a local URL, usually:

```text
http://localhost:8501
```

Open the URL in your browser to use the application.

### Step 7: Make Predictions

1. Upload a JPG, JPEG, or PNG image.
2. Preview the uploaded image.
3. Click the **Predict** button.
4. View the predicted class and confidence score.

**Important:** Make sure the trained model file exists inside the `models` folder before running the application.

No dataset download or model training is required to use the application.

## Learning Outcomes

Through this project, I gained practical experience in:

- Transfer Learning using MobileNetV2.
- Building and training Deep Learning models.
- Image preprocessing and data augmentation.
- Evaluating classification performance.
- Building interactive applications using Streamlit.
- Integrating trained models into an application.

## Author

**Shayan Ali**

GitHub: https://github.com/ShayanAli10

---

*This project was developed for educational and learning purposes. Predictions represent the model's learned binary dataset labels and should not be treated as definitive judgments about a person's gender.*
