# Heart Disease Prediction API (End-to-End Deployment)

### Objective
To develop, serialize, and deploy a machine learning classification model as a live REST API using Flask, GitHub, and Render, enabling the prediction of heart disease risks based on clinical parameters.

### Dataset Link
[Heart Disease Dataset on Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)[cite: 6]

### Libraries Used
* **Pandas:** Data loading, missing value verification, and DataFrame formatting.
* **Scikit-Learn:** 80/20 train-test splitting, Random Forest model training, and accuracy evaluation.
* **Pickle:** Model serialization and deserialization.
* **Flask & Gunicorn:** REST API development and WSGI server deployment.

### Methodology
1. **Data Preprocessing:** Loaded clinical data, verified zero missing values, and partitioned it into an 80% training set and a 20% testing set.
2. **Model Training:** Built a Random Forest Classifier to handle non-linear clinical thresholds. Evaluated the model using the standard accuracy score.
3. **Serialization:** Exported the trained model to a `model.pkl` binary file.
4. **API Development:** Created a Flask application with a `/predict` POST endpoint accepting JSON payloads and returning explicit prediction strings[cite: 6].

### Render Deployment URL
**[INSERT YOUR LIVE RENDER URL HERE ONCE DEPLOYED]**

---

### Conclusion
The Random Forest model performed exceptionally well during the evaluation phase, delivering a robust accuracy score on the unseen 20% testing partition[cite: 6]. During cloud deployment via Render, the primary challenges involved ensuring environment consistency—specifically matching Scikit-learn library versions between the local training environment and the cloud server—and correctly configuring the Gunicorn WSGI server to bind to the appropriate web ports[cite: 6]. Overcoming these hurdles underscores the critical importance of MLOps in modern machine learning projects[cite: 6]. MLOps practices automate model packaging, manage version control via GitHub, and streamline the deployment pipeline[cite: 6]. By serving predictions through a live REST API, MLOps transforms static Jupyter notebooks into scalable, publicly accessible web services that can be integrated directly into healthcare management systems[cite: 6].