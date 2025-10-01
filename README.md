# Bank Marketing Campaign – Predicting Term Deposit Subscription

## Business Objective
Customer service teams are tasked with calling potential clients to encourage them to sign up for long-term deposits.  
Currently, only about **11% of contacted clients subscribe**, which increases customer acquisition costs.  

The goal is to build a **binary classification model** that predicts whether a client will subscribe (`y = yes/no`), allowing customer service to prioritize **likely subscribers**.  
 

---

## Dataset
**Source**: [UCI Machine Learning Repository – Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)  

- **Rows**: ~41K  
- **Features**:
  - Personal Information: age, job, marital, education  
  - Banking Information: default, housing loan, personal loan  
  - Campaign Details: contact type, number of contacts (`campaign`), days since last contact (`pdays`), previous outcome (`poutcome`)  
  - Socioeconomic Indicators: employment variation rate, consumer confidence index, euribor3m, number of employees  
- **Target**: `y` (binary: `yes` if client subscribed, else `no`)  

---

## Data Preparation
- Removed campaign outliers (campaign > 20, duration > 2000)  
- Engineered features:  
  - `never_contacted` = 1 if `pdays == 999` else 0  
  - `have_loan` = 1 if `housing == yes` or `loan == yes`  
- Encodings:  
  - Nominal → OneHotEncoder (job, default, housing, loan)  
  - Ordinal → OrdinalEncoder (education hierarchy)  
- Balanced training data with SMOTENC

  <p float="center">
        <img src="./images/subscription distribution.png" width="45%"> 
    </p> 
    <p float="center">
        <img src="./images/Subscription based on newly contacted.png" width="50%"> 
    </p> 


---

## Modeling Approach
Models compared:
- Baseline(Dummy)
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Tree  
- Support Vector Machine (SVM, RBF kernel)  
- Random Forest (recall-optimized)  

### Pipeline
1. Preprocessing (encoding + scaling)  
2. Balancing (SMOTE / SMOTENC)  
3. Classifier  
4. Evaluation and threshold tuning  

---

## Installation

- Clone the repository:
   ```bash
   git clone https://github.dev/premkumargit/classification_customercare_targets.git
   cd job_recommandation
   ```
---

## Usage
- Start Jupyter Notebook as below
  ``` bash
    jupyter notebook
  ```
- Open the notebook either in Jupyterlab or jupyter notebook [prompt_III.ipynb](https://github.com/premkumargit/classification_customercare_targets/blob/main/prompt_III.ipynb) and run all the cell

---

## Evaluation Metrics and Improvements
- Accuracy: overall correctness  
- Recall (priority): percentage of true subscribers correctly identified  
- Precision: percentage of predicted subscribers that are actual subscribers  
- F1-score: balance of recall and precision  
- **Precision vs Recall:** Use case: Customer service teams are tasked with calling potential clients to encourage them to sign up for long-term deposits. 
    - Precision (avoiding false positives) is less important, because even if the model mistakenly flags some uninterested customers, calling them just means a wasted call. The downside is relatively low.
    - Recall (minimizing false negatives) is critical, because missing true positives means you fail to reach clients who would have signed up. That directly reduces revenue and defeats the campaign’s purpose.
- **Unbalanced dataset:** This is highly unbalanced dataset. only 11% of total subscribed for long deposite
    - Metrics like accuracy become misleading — the model seems good, but it’s useless for the minority class.
    - Classifiers like Logistic Regression, SVM,  k-NN are affected since they rely on balanced decision boundaries. 

Plots generated:
- Confusion Matrix  
  <p float="center">
        <img src="./images/confustion matrix.png" width="65%"> 
    </p>   

---

## Key Results
- By considering the use case, the model was trained for better **recall** rather than accuracy.  
- **RandomForestClassifier** and **Decision Tree** performed better, with ~3 seconds training time, delivering higher recall and accuracy. Precision was acceptable, with about a 50% increase in false positives.  
- **SVM** achieved strong performance after RandomForest, but required significantly more training time and resources.  
- **SMOTE** did not improve model performance in this setup.  

---

