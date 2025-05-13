
# 📌 DeepFocusAI: Automated Optimal Focus Detection for Wafer Inspection

## 🚀 Project Overview

**DeepFocusAI** utilizes advanced deep learning techniques to automatically determine optimal focus adjustments in semiconductor wafer inspection systems. By predicting precise focus heights, this solution significantly enhances defect detection capabilities and manufacturing accuracy, ultimately contributing to improved semiconductor production efficiency.

---

## 🌟 Key Highlights

- **Synthetic Wafer Image Generation:** Realistic wafer inspection images generated from wafer map data with varying levels of simulated defocus.
- **Automated Sharpness Quantification:** Uses the Variance of Laplacian metric to objectively evaluate image sharpness.
- **CNN-based Regression Model:** Built and trained a customized convolutional neural network (CNN) using TensorFlow/Keras, capable of accurately predicting optimal focus scores.
- **Outstanding Performance:** Achieved very low validation MAE (0.0154) and validation MSE (0.0007), indicating robust and reliable predictive performance.

---

## 🛠️ Technologies Utilized

- **Programming & Data Science:** Python, TensorFlow, Keras, OpenCV, NumPy, Pandas, Matplotlib
- **Image Processing & Computer Vision:** Gaussian Blur simulations, Laplacian variance for sharpness evaluation
- **Modeling & Machine Learning:** CNN regression modeling, supervised learning techniques

---

## 📁 Dataset Source

This project uses wafer map data from the publicly available **WM-811K** dataset on Kaggle:

- **WM-811K: Wafer Map Failure Patterns**  
  📎 [https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)

The raw wafer maps were converted into synthetic optical-style images, and various levels of Gaussian blur were applied to simulate defocus conditions. This synthetic dataset was then labeled using the variance of the Laplacian to represent sharpness scores.

---

## 📂 Repository Structure

```bash
DeepFocus-AI/
├── synthetic_images/
│   ├── sharp/                      # Original synthetic images (in-focus)
│   └── blurred/                    # Artificially blurred images (simulated defocus)
├── notebooks/
│   └── DeepFocus-AI.ipynb           # Jupyter notebook outlining project workflow
├── models/
│   └── deepfocusAI_model.h5        # Final trained CNN model
├── dataset/
│   └── synthetic_focus_dataset.csv # Generated labeled dataset
├── README.md                       # Project introduction & details
└── requirements.txt                # Project dependencies
├── .gitignore                      # Files and folders to ignore in version control
├── LICENSE                         # MIT license for open source usage
```

---

## 🚩 Quick Start Guide

### 🔸 Clone Repository

```bash
git clone https://github.com/<your-github-username>/DeepFocus-AI.git
cd DeepFocus-AI
```

### 🔸 Set Up Python Environment (Conda recommended)

```bash
conda create -n deepfocus-ai python=3.10
conda activate deepfocus-ai
pip install -r requirements.txt
```

### 🔸 Launch Notebook

```bash
jupyter notebook notebooks/DeepFocus-AI.ipynb
```

---

## 📊 Results Summary

| Metric                                    | Result    |
|-------------------------------------------|-----------|
| **Validation Mean Absolute Error (MAE)**  | `0.0154`  |
| **Validation Mean Squared Error (MSE)**   | `0.0007`  |

---

## 🔮 Future Directions

- **Model Refinement:** Explore advanced architectures such as ResNet, EfficientNet, or attention-based mechanisms.
- **Hyperparameter Optimization:** Improve model robustness through detailed hyperparameter tuning and cross-validation.
- **Real-world Integration:** Test the model performance with actual semiconductor inspection systems.

---

## 🧑‍💻 About Me

**Abhir Iyer**  
📍 Bloomington, IN, USA  

I’m a graduate student pursuing a **Master of Science in Data Science** at Indiana University Bloomington, passionate about leveraging data-driven techniques and machine learning to solve impactful real-world problems. With hands-on experience in Data Science internships, I specialize in predictive analytics, computer vision, and advanced statistical modeling.

- **Website**: [abhir.com](https://abhir.com)  
- **LinkedIn**: [linkedin.com/in/abhir-iyer](https://linkedin.com/in/abhir-iyer)  
- **Email**: [abhiyer@iu.edu](mailto:abhiyer@iu.edu)  

---

## 🎖️ Certifications

- **Google Data Analytics Professional Certificate** – Coursera, 2023  
- **Google Crash Course on Python** – Coursera, 2021  

---

## ⭐ Support & Contributions

Feel free to connect if you'd like to collaborate or discuss potential improvements!

⭐ If you found this project interesting or useful, please consider giving it a star ⭐
