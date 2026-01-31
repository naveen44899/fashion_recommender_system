#  Fashion Recommender System using Deep Learning

An end-to-end **image-based fashion recommender system** built using **Deep Learning (CNN – ResNet50)** and **Computer Vision**. The application recommends visually similar fashion products when a user uploads an image.

---

##  Project Overview

This project uses **transfer learning** with a pre-trained **ResNet50** model to extract deep visual features from fashion images. These features are compared using a **K-Nearest Neighbors (KNN)** algorithm to recommend similar fashion items.

The project is deployed as an **interactive web application using Streamlit**.

---

##  How It Works

1. User uploads a fashion image
2. Image is preprocessed and passed through **ResNet50 (without top layer)**
3. Deep feature embeddings are extracted and normalized
4. Similarity is calculated using **Euclidean distance**
5. Top 5 visually similar fashion products are displayed

---

##  Tech Stack

* **Python**
* **TensorFlow / Keras**
* **ResNet50 (CNN – Transfer Learning)**
* **Scikit-learn (KNN)**
* **NumPy & Pickle**
* **Streamlit (Web App)**

---

## 📂 Project Structure

```
├── app.py                  # Streamlit application
├── feature_list.pkl        # Extracted image feature vectors
├── filenames.pkl           # Image file paths
├── uploads/                # Uploaded images
├── data/                   # Fashion dataset (optional)
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
```

---

## 📦 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git https://github.com/naveen44899/fashion_recommender_system.git

```

### 2️⃣ Create Virtual Environment (Optional but Recommended)

```bash
conda activate venv/ # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Then open your browser at:

```
http://localhost:8501
```

---

## 📸 Demo

* Upload a fashion product image
* Get top 5 visually similar fashion recommendations

*(Add screenshots or demo video link here)*

---


---

## 🎯 Learning Outcomes

* Practical experience with **CNNs & Transfer Learning**
* Feature extraction using pretrained models
* Image similarity search
* Building & deploying ML-powered web apps

---



---

## 📬 Contact

If you have suggestions or feedback, feel free to connect with me on **LinkedIn**.

---

⭐ If you like this project, consider giving it a star!
