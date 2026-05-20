# 🌾 Smart Soil Nutrient and Crop Recommendation System using Virtual IoT and Machine Learning

##  Overview

This project presents a **Virtual IoT-based Smart Agriculture System** that predicts the most suitable crop, required nutrient, and fertilizer based on soil conditions. The system simulates IoT soil sensors and uses Machine Learning to assist farmers in making better agricultural decisions.

The proposed solution is cost-effective, scalable, and useful for precision farming.

---

## Problem Statement

Farmers often face challenges in:

* Selecting the right crop for their soil
* Identifying nutrient deficiencies
* Applying correct fertilizers
* Maintaining soil health

Traditional soil testing is expensive and time-consuming. This project provides a **low-cost, AI-driven solution**.

---

##  Proposed Solution

The system uses:

* Virtual IoT sensors to simulate soil nutrient data
* Machine Learning models for crop and nutrient prediction
* A web-based dashboard for visualization and decision support

---

##  Features

✔ Crop recommendation
✔ Nutrient deficiency detection
✔ Fertilizer suggestion
✔ Soil health analysis
✔ Data visualization (NPK graph)
✔ Confusion matrix for model evaluation
✔ Accuracy and cross-validation
✔ User-friendly dashboard

---

##  Technologies Used

* Python
* Flask
* Machine Learning (Random Forest)
* Pandas, NumPy, Scikit-learn
* Matplotlib
* HTML, CSS

---

##  Machine Learning Details

The system uses the **Random Forest algorithm** for classification.

### Model Performance:

* Crop Prediction Accuracy: ~88%
* Cross Validation Accuracy: ~83%
* Nutrient Prediction Accuracy: High due to strong feature correlation.

---

## System Architecture

The system consists of:

1. Virtual IoT soil sensors
2. Data preprocessing
3. Machine learning model
4. Prediction and visualization
5. Decision support dashboard

---

##  Workflow

1. User inputs soil NPK values
2. Data is preprocessed and encoded
3. ML model predicts crop and nutrient
4. Fertilizer recommendation is generated
5. Soil health is evaluated
6. Results are displayed on dashboard

---

##  Project Structure

```
SoilNutrient/
│
├── app.py
├── model.py
├── generate_data.py
├── Crop_recommendation.csv
├── model.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   ├── graph.png
│   └── confusion_matrix.png
│
└── README.md
```

---

##  How to Run the Project

### Step 1: Install dependencies

```
pip install flask pandas numpy scikit-learn matplotlib
```

### Step 2: Generate dataset

```
python generate_data.py
```

### Step 3: Train model

```
python model.py
```

### Step 4: Run web app

```
python app.py
```

### Step 5: Open browser

Go to:

```
http://127.0.0.1:5000
```

---

##  Results

The system successfully predicts:

* Crop type
* Nutrient deficiency
* Fertilizer requirement
* Soil quality

Visualization and confusion matrix improve interpretability.

---

##  Applications

* Smart farming
* Precision agriculture
* Soil health monitoring
* Agricultural advisory systems

---

##  Future Scope

* Integration with real IoT sensors
* Cloud-based monitoring
* Mobile application
* Weather prediction
* Satellite and remote sensing
* Real-time alerts for farmers

---

##  References

* Research papers on precision agriculture
* Machine learning in agriculture
* IoT-based smart farming systems

---


