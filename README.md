# Spam Classification ML Pipeline

## 📌 Project Overview
This project implements a machine learning pipeline to classify messages as **Spam** or **Not Spam** using a **Random Forest Classifier**. The pipeline consists of the following stages:

1. **Data Ingestion**  
2. **Data Preprocessing**  
3. **Feature Engineering**  
4. **Model Building**  
5. **Model Evaluation**  
6. **Model Deployment**  

---

## 📂 Directory Structure
```
├── data/                 # Dataset storage
│   ├── raw_data.csv      # Original dataset
│   ├── processed_data.csv # Cleaned & preprocessed dataset
│
├── notebooks/            # Jupyter notebooks for experimentation
│
├── src/                  # Source code for ML pipeline
│   ├── data_ingestion.py  # Data loading scripts
│   ├── preprocessing.py   # Data cleaning & transformation
│   ├── feature_engineering.py # Feature extraction & selection
│   ├── model_building.py  # Model training script
│   ├── evaluation.py      # Model performance evaluation
│   ├── deployment.py      # Model deployment script
│
├── README.md             # Project documentation
├── requirements.txt      # Dependencies
└── train.py              # Main script to execute the pipeline
```

---

## 🔹 1. Data Ingestion
- Load the dataset (CSV format)
- Handle missing values (if any)
- Split data into training and testing sets

---

## 🔹 2. Data Preprocessing
- Remove special characters, punctuation, and stopwords
- Convert text to lowercase
- Tokenization & Lemmatization
- Perform stemming (if needed)

---

## 🔹 3. Feature Engineering
- Convert text into numerical representations using **TF-IDF Vectorization**
- Optional: Use **Word Embeddings (Word2Vec, GloVe)**
- Extract additional text features such as word count and sentence length

---

## 🔹 4. Model Building (Random Forest)
- Train a **Random Forest Classifier** using the extracted features
- Hyperparameter tuning using GridSearchCV

---

## 🔹 5. Model Evaluation
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- ROC-AUC Score

---

## 🔹 6. Model Deployment
- Save the trained model using **Pickle or Joblib**
- Deploy the model as a REST API using Flask or FastAPI
- Serve predictions via an endpoint

---

## 🚀 How to Run the Pipeline
### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Training Pipeline
```bash
python train.py
```

### 3️⃣ Deploy the Model
```bash
python src/deployment.py
```

---

## 📊 Results & Performance
- The model achieves an **F1-score of ~95%** on the test dataset.
- Feature importance analysis shows **TF-IDF features** contribute significantly.

---

## 📌 Future Improvements
✅ Implement deep learning models (LSTMs, Transformers) for better performance  
✅ Deploy the model as a REST API using FastAPI or Flask  
✅ Use **Active Learning** to continuously improve the model with new data  
✅ Integrate the model with a real-time email filtering system  

---

## 📝 Authors
- **Your Name**
- Contact: [Your Email]

