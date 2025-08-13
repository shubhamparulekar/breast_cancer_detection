# Breast Cancer Detection using Deep Learning

This project uses a deep learning model to classify breast cancer histology images as either benign or malignant. It leverages transfer learning with a pre-trained Convolutional Neural Network (CNN) to achieve high accuracy in detection, providing a tool to assist radiologists in diagnosing breast cancer.

---

## 📋 Table of Contents

- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Implementation](#implementation)
- [Performance Metrics](#performance-metrics)
- [Results](#results)
- [Website Screenshots](#website-screenshots)
- [How to Use](#how-to-use)
- [Contributors](#contributors)
- [References](#references)

---

## 📖 About the Project

Breast cancer is one of the most common cancers among women worldwide. Early and accurate detection is crucial for effective treatment. This project aims to develop an end-to-end deep learning model that can accurately classify breast cancer from histopathological images.

The system is designed to:
- Preprocess mammogram images to enhance quality.
- Use a CNN model to extract features and classify images.
- Distinguish between malignant and benign tumors.
- Provide a user-friendly web interface for uploading images and viewing results.

---

## 📊 Dataset

The model was trained on the **Breast Cancer Histopathological Database (BreakHis)**. This dataset consists of 9,109 microscopic images of breast tumor tissue collected from 82 patients.

- **Classes:** Benign and Malignant
- **Data Split:**
  - **Training set:** 1000 images per class
  - **Validation set:** 250 images per class
- **Link:** You can find more information and download the dataset [here](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images).

---

## 🔬 Methodology

The project follows a standard machine learning workflow, from data preprocessing to model evaluation.
<img width="772" height="427" alt="image" src="https://github.com/user-attachments/assets/b5dbbc73-87a1-4e86-a024-ec2d3d135501" />

### 1. Data Preprocessing
- **Image Augmentation:** To prevent overfitting and increase the diversity of the training data, techniques like rotation, shifting, and flipping were applied.
- **Normalization:** Pixel values were scaled to a standard range.
<img width="1174" height="646" alt="image" src="https://github.com/user-attachments/assets/7049d117-36ae-44ef-a645-d26f0b88684a" />


### 2. Model Architecture
A **Convolutional Neural Network (CNN)** was used for classification. The project implements a transfer learning approach using the **DenseNet201** architecture, pre-trained on the ImageNet dataset.

<img width="750" height="380" alt="image" src="https://github.com/user-attachments/assets/21e07e7f-c903-4f84-8327-c7960e3b1430" />

The model architecture includes:
1.  **Input Layer:** Takes histology images as input.
2.  **Base Model (DenseNet201):** The pre-trained convolutional base for feature extraction.
3.  **Global Average Pooling:** To reduce the spatial dimensions of the feature maps.
4.  **Dropout Layer:** To prevent overfitting.
5.  **Batch Normalization:** To stabilize and speed up the training process.
6.  **Dense Layer (Output):** A fully connected layer with a `softmax` activation function to output the probability for each class (Benign or Malignant).

<img width="729" height="359" alt="image" src="https://github.com/user-attachments/assets/3375f239-04f0-4922-aa5e-e3074662aa74" />




---

## ⚙️ Implementation

### Hardware
- **Processor:** Standard CPU or a cloud-based GPU for faster training.
- **Memory:** At least 8GB RAM.

### Software & Dependencies
The model was built using Python and the Keras library with a TensorFlow backend.
- `python`
- `tensorflow`
- `keras`
- `numpy`
- `matplotlib`
- `scikit-learn`

---

## 📈 Performance Metrics

To evaluate the model's performance, the following metrics were used:

- **Accuracy:** The proportion of correctly classified images.
- **Precision:** The ratio of true positive predictions to the total positive predictions.
- **Recall (Sensitivity):** The ratio of true positive predictions to all actual positive cases.
- **F1-Score:** The harmonic mean of Precision and Recall, providing a single score that balances both.
- **ROC Curve & AUC:** The Receiver Operating Characteristic (ROC) curve plots the true positive rate against the false positive rate. The Area Under the Curve (AUC) represents the model's ability to distinguish between classes.
- **Confusion Matrix:** A table that visualizes the performance of the classification model by showing the counts of true positive, true negative, false positive, and false negative predictions.
<img width="877" height="227" alt="image" src="https://github.com/user-attachments/assets/2208ac09-65bc-4507-a702-a2ec32036cd8" />

---

## 🏆 Results

The model achieved high performance in classifying breast cancer histology images.

| Metric    | Score |
| :-------- | :---- |
| Accuracy  | 98.3% |
| Precision | 0.65  |
| Recall    | 0.95  |
| F1-Score  | 0.77  |
| ROC-AUC   | 0.692 |

### Confusion Matrix
The confusion matrix below shows the number of correct and incorrect predictions for each class.
<img width="418" height="393" alt="image" src="https://github.com/user-attachments/assets/deaeabed-c90b-4603-8650-baaf3c72ae0b" />


### Accuracy and Loss Curves
The training and validation accuracy and loss curves over epochs.

#### Accuracy vs. Epochs
<img width="488" height="213" alt="image" src="https://github.com/user-attachments/assets/4dcb596d-e80f-4188-918f-dd93147ab88a" />

#### Loss vs. Epochs
<img width="488" height="222" alt="image" src="https://github.com/user-attachments/assets/e1527b61-1c40-460e-a6be-46a248c6b0e8" />


---

## 🖥️ Website Screenshots

Here are some screenshots of the web application interface.

#### Main Page
<img width="1918" height="957" alt="image" src="https://github.com/user-attachments/assets/b44c4df4-e17d-4afd-98d4-62e80dfae29f" />

<img width="1057" height="938" alt="Screenshot from 2025-08-13 18-33-08" src="https://github.com/user-attachments/assets/e6bfcc17-90fe-45f3-a666-a958161eeec0" />


#### Results Page
<img width="910" height="265" alt="image" src="https://github.com/user-attachments/assets/78573fa7-c08a-45ed-b295-b4aa9f01092a" />
<img width="911" height="264" alt="image" src="https://github.com/user-attachments/assets/c528b57f-75fb-49df-b8a4-1a04cf2302a1" />



---

## 🚀 How to Use

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/breast-cancer-detection.git](https://github.com/your-username/breast-cancer-detection.git)
    ```
2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the application:**
    ```bash
    python app.py
    ```
4.  Open your web browser and navigate to `http://127.0.0.1:5000`.
5.  Upload a histology image to get a classification result.

---

## 🧑‍💻 Contributors

- **Gaurav Mhatre**
- **Mihir Gadkar**
- **Shubham Parulekar**

---

## 📚 References

[1] Shen, L., Margolies, L. R., Rothstein, J. H., Fluder, E., McBride, R., & Sieh, W. (2019). Deep Learning to Improve Breast Cancer Detection on Screening Mammography. *Scientific Reports*.

[2] Vakka, A. R., Soni, B., & Reddy, S. (2020). Breast cancer detection by leveraging Machine Learning. *The Korean Institute of Communication and Information Sciences (KICS)*.

[3] Gardezi, S. J. S., Elazab, A., Lei, B., & Wang, T. (2019). Breast Cancer Detection and Diagnosis Using Mammographic Data: Systematic Review. *Journal of Medical Internet Research*.

