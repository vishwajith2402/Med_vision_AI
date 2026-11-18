# MedAI Vision - Medical Image Diagnosis using Machine Learning

## Project Overview

MedAI Vision is a machine learning based medical diagnosis system designed to analyze medical images such as X-ray and MRI scans and identify whether a disease is present or not. Medical imaging plays an important role in early detection of serious health conditions such as pneumonia, tumors, infections, and neurological disorders.
Manual analysis of medical images requires skilled radiologists and significant time. In hospitals with a large number of patients, delays in diagnosis may affect treatment decisions. This project provides an automated diagnostic assistant that helps medical professionals quickly analyze medical images and predict possible abnormalities.
The system uses image processing techniques and machine learning algorithms to classify medical images into two categories:
- Normal, or healthy
- Disease, or abnormal
The system also provides confidence scores to indicate prediction reliability.

## Key Features
- Secure login page for controlled access
- Medical-themed Streamlit interface
- Animated ECG design and medical cross branding
- Image upload support for JPG, JPEG, and PNG files
- Image preprocessing using OpenCV
- Binary classification into Normal or Disease
- Confidence score and class probability display
- Synthetic dataset generator for demo and testing
- Trained model saving and loading using Joblib

## Demo Login Details
Use any one of the following demo accounts to access the application:
| Role   | Username | Password   |
|   ---  |   ---    |    ---     |
| Admin  | `admin`  | `admin123` |
| Doctor | `doctor` | `medai123` |

These credentials are defined in `login.py`. They are suitable for project demonstration only. For real deployment, replace them with a database-backed authentication system and encrypted password storage.

## Problem Statement
Hospitals and diagnostic centers process thousands of X-rays and MRI scans every day. Identifying abnormalities manually can be time-consuming and may lead to delayed treatment.
Common challenges include:
- Large number of medical images
- Time-consuming manual diagnosis
- Risk of human error
- Need for faster clinical decision-making
- Shortage of specialists in remote areas
The goal of this project is to build an AI-powered diagnostic assistant that can analyze medical scan images, detect abnormalities, provide quick prediction results, and assist doctors in decision-making.

## Objectives
The main objectives of this project are:

1. Develop an AI system that can classify medical images.
2. Automate the disease detection process.
3. Reduce the time required for diagnosis.
4. Improve healthcare efficiency.
5. Demonstrate the practical application of machine learning in the medical field.
6. Build a user-friendly interface for image upload and prediction.
7. Implement a secure login system.

## Scope

The project focuses on binary classification of medical images into Normal or Disease categories. It can be extended to detect multiple diseases or analyze different types of scans such as CT scans, ultrasound images, or pathology images.

This system can be used in:

- Hospitals
- Diagnostic centers
- Medical research labs
- Telemedicine platforms
- Medical training institutions

## Technologies Used

### Programming Language

- Python

### Libraries

- NumPy for numerical operations
- Pandas for dataset handling
- OpenCV for image processing
- Scikit-learn for machine learning
- Streamlit for the web interface
- Joblib for model saving and loading

### Machine Learning Algorithm

- Random Forest Classifier

### Tools

- VS Code
- Python
- Streamlit

## Project Files

| File or Folder | Purpose |
| --- | --- |
| `ui.py` | Main Streamlit user interface for login, image upload, prediction, and result display |
| `login.py` | Handles demo authentication and logout |
| `model_utils.py` | Contains image reading, preprocessing, feature extraction, model loading, and prediction functions |
| `train_model.py` | Trains the Random Forest model using images and labels |
| `generate_synthetic_data.py` | Creates synthetic demo images and updates `Labels.csv` |
| `Labels.csv` | Stores image filenames and their labels |
| `dataset/` | Stores training and testing images |
| `models/` | Stores the trained model file |
| `requirement.txt` | Lists required Python libraries |
| `run_app.bat` | Windows shortcut script to start the Streamlit app |
| `.streamlit/config.toml` | Streamlit theme configuration |

## Working Principle

The system works in the following steps:

1. Data collection: Medical images such as X-rays and MRI scans are collected and stored.
2. Data preprocessing: Images are resized and converted into grayscale format for easier analysis.
3. Feature extraction: Important image features are extracted using image processing techniques.
4. Model training: The machine learning algorithm learns patterns from labeled images.
5. Prediction: When a new image is uploaded, the model predicts whether the image indicates disease or a normal condition.
6. Result display: The prediction result and confidence score are displayed in the user interface.

## System Architecture

```text
Input Image
    |
    v
Image Preprocessing
    |
    v
Feature Extraction
    |
    v
Machine Learning Model
    |
    v
Prediction
    |
    v
Result Display
```

## Modules

### 1. Data Collection Module

Collects medical images for training the model.

### 2. Image Preprocessing Module

Resizes images and converts them into a machine-readable format.

### 3. Model Training Module

Trains the machine learning algorithm using labeled images.

### 4. Prediction Module

Predicts whether a newly uploaded medical image is Normal or Disease.

### 5. User Interface Module

Provides an interface to upload medical images and view prediction results.

### 6. Login Module

Provides authentication to access the system.

## Input

The system accepts medical image files such as:

- JPG
- PNG

Example inputs:

- MRI scan image
- X-ray image

## Output

The system displays:

- Prediction result: Normal or Disease
- Confidence score

Example:

```text
Disease detected with 87% confidence
```

## Advantages

- Fast diagnosis
- Easy-to-use interface
- Reduces manual workload
- Improves decision-making
- Cost-effective solution
- Useful for medical research
- Can be extended for multiple diseases

## Limitations

- Accuracy depends on dataset quality.
- Requires labeled medical images.
- Cannot replace medical professionals.
- Works best with clear images.

## Future Enhancements

- Multi-disease classification
- Deep learning model integration
- Heatmap visualization
- Cloud deployment
- Mobile app development
- Real-time hospital integration

## Applications

- Healthcare sector
- Medical research
- Radiology labs
- Telemedicine
- Medical education
- AI healthcare startups

## Installation

Make sure Python is installed on your system. Python 3.10 or above is recommended.

Clone or open the project folder:

```text
medAI_Vision
```

Install the required Python libraries from the project directory:

```bash
pip install -r requirement.txt
```

## Quick Start

If synthetic data and a trained model already exist, start the application directly:

```bash
run_app.bat
```

or:

```bash
streamlit run ui.py
```

Then open the application in your browser:

```text
http://127.0.0.1:8501
```

Login using:

```text
Username: admin
Password: admin123
```

## Dataset Format

Place medical images inside the `dataset` folder and enter their filenames and labels in `Labels.csv`.

Example:

```csv
filename,label
normal_001.jpg,Normal
disease_001.jpg,Disease
normal_002.png,Normal
disease_002.png,Disease
```

Only two labels are supported by the current model:

- Normal
- Disease

For demo testing without a real medical image dataset, generate synthetic images and labels:

```bash
python generate_synthetic_data.py
```

This creates balanced synthetic Normal and Disease images in `dataset` and writes matching rows to `Labels.csv`. Synthetic data is useful for verifying the application pipeline, but it does not represent real diagnostic accuracy.

The default synthetic generator creates:

- 60 Normal images
- 60 Disease images
- 120 total labeled images

## Train the Model

Train the Random Forest model after adding labeled images:

```bash
python train_model.py
```

The trained model will be saved to:

```text
models/medai_random_forest.joblib
```

## Run the Application

```bash
streamlit run ui.py
```

On Windows, you can also start the app with:

```bash
run_app.bat
```

Then open:

```text
http://127.0.0.1:8501
```

## How to Use the Application

1. Start the Streamlit application.
2. Open `http://127.0.0.1:8501` in a browser.
3. Login using the demo username and password.
4. Upload an X-ray or MRI image in JPG, JPEG, or PNG format.
5. Click `Analyze Scan`.
6. View the prediction result, confidence score, and class probabilities.

## Model Details

The current model uses a Random Forest Classifier. Each uploaded image is processed before prediction.

Preprocessing steps:

1. Read the uploaded image.
2. Convert it to grayscale.
3. Resize it to 128 x 128 pixels.
4. Apply Gaussian blur.
5. Apply histogram equalization.
6. Extract pixel, histogram, edge, and shape-based features.
7. Send the features to the trained Random Forest model.

The model outputs:

- Predicted class: Normal or Disease
- Confidence score
- Probability for each class

## Common Commands

Generate synthetic demo data:

```bash
python generate_synthetic_data.py
```

Train the model:

```bash
python train_model.py
```

Run the app:

```bash
streamlit run ui.py
```

Run the app on Windows:

```bash
run_app.bat
```

## Troubleshooting

If the application does not open, check that Streamlit is running and open:

```text
http://127.0.0.1:8501
```

If the model is missing, train it first:

```bash
python train_model.py
```

If training fails because images are missing, make sure every filename in `Labels.csv` exists inside the `dataset` folder.

If dependencies are missing, reinstall them:

```bash
pip install -r requirement.txt
```

## Conclusion

MedAI Vision demonstrates how machine learning can be applied in healthcare to assist in medical image diagnosis. The project helps in understanding the practical implementation of AI techniques for solving real-world problems. It provides a foundation for developing advanced medical AI systems that can improve healthcare efficiency and support medical professionals.

## Disclaimer

This project is intended for educational and demonstration purposes only. It should not be used as a replacement for professional medical diagnosis, clinical judgment, or advice from qualified healthcare professionals.
