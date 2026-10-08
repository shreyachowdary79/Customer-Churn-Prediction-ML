# Customer-Churn-Prediction-ML
Customer churn prediction using Python, exploratory data analysis, feature engineering, and machine learning. Compares six classification models using accuracy, recall, F1-score, and ROC-AUC to identify customers at risk of leaving.
# Customer Churn Prediction & Analysis

**Machine Learning | Python | Exploratory Data Analysis | Customer Retention**

A data analytics and machine learning project focused on understanding customer churn patterns, identifying potential churn drivers, and comparing classification models to predict customer attrition. The project combines exploratory analysis, feature preparation, class-imbalance handling, and model evaluation to demonstrate how customer data can support data-driven retention strategies.

## Project Overview

Customer churn occurs when customers stop using a company's products or services. Identifying customers who are likely to leave helps businesses investigate potential issues, improve customer experience, and develop targeted retention strategies.

This project analyzes credit card customer data to explore patterns associated with churn and evaluates multiple machine learning classification algorithms to predict whether a customer is likely to exit.

### Business Objectives

* Understand customer characteristics associated with churn.
* Explore relationships between customer demographics, account activity, and churn.
* Prepare customer data for machine learning.
* Address class imbalance during model development.
* Compare classification models using multiple evaluation metrics.
* Identify analytical opportunities to improve customer retention.

## Dataset

**Source:** [Credit Card Customer Churn Prediction — Kaggle](https://www.kaggle.com/datasets/rjmanoj/credit-card-customer-churn-prediction/data)

The dataset contains customer information, account characteristics, and a target variable indicating whether a customer exited.

### Key Features

| Feature         | Description                              |
| --------------- | ---------------------------------------- |
| CreditScore     | Customer's credit score                  |
| Geography       | Customer's geographical region           |
| Gender          | Customer's gender                        |
| Age             | Customer's age                           |
| Tenure          | Number of years as a customer            |
| Balance         | Customer's account balance               |
| NumOfProducts   | Number of products held                  |
| HasCrCard       | Whether the customer has a credit card   |
| IsActiveMember  | Whether the customer is an active member |
| EstimatedSalary | Estimated customer salary                |
| Exited          | Target variable indicating customer exit |

Identifiers such as `RowNumber`, `CustomerId`, and `Surname` are also present in the source dataset and should be evaluated for exclusion from model features to avoid irrelevant or misleading predictions.

**Target variable:** `Exited`

* `0` — Customer did not exit
* `1` — Customer exited

## Technology Stack

* **Programming:** Python
* **Data Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn, XGBoost
* **Data Preparation:** Feature engineering, categorical encoding, feature scaling
* **Imbalanced Data Handling:** SMOTE and class weighting
* **Development Environment:** Jupyter Notebook

## Project Workflow

```text
Customer Churn Dataset
          |
          v
   Data Exploration
          |
          v
 Data Cleaning & EDA
          |
          v
 Feature Preparation
          |
          v
 Train-Test Split
          |
          v
 Class Imbalance Handling
          |
          v
 Classification Models
          |
          v
 Model Evaluation
          |
          v
 Churn Insights & Recommendations
```

## Exploratory Data Analysis

Exploratory Data Analysis (EDA) is used to investigate customer characteristics and identify patterns associated with churn.

Key areas of analysis include:

* Customer age and churn distribution
* Geographical and demographic patterns
* Relationship between account balance and churn
* Customer tenure and product ownership
* Active membership and churn behavior
* Credit score and estimated salary distributions
* Relationships among numerical and categorical variables

Visualizations help reveal potential churn patterns and guide subsequent feature preparation and modeling.

## Data Preparation

The project prepares customer data for classification through preprocessing and feature engineering.

Key techniques include:

* Inspecting data types and missing values
* Exploring feature distributions and relationships
* Encoding categorical variables
* Scaling numerical features where required
* Separating features from the target variable
* Addressing class imbalance using SMOTE or class weighting

**Important:** Resampling must be applied only to the training data, after the train-test split, to avoid data leakage and overly optimistic evaluation results.

## Machine Learning Models

Six classification algorithms are compared to evaluate their ability to identify customers who exit.

| Model                        | Purpose                                          |
| ---------------------------- | ------------------------------------------------ |
| Logistic Regression          | Baseline classification model                    |
| Random Forest                | Ensemble learning using decision trees           |
| K-Nearest Neighbors (KNN)    | Classification based on neighboring observations |
| Support Vector Machine (SVM) | Classification using decision boundaries         |
| XGBoost                      | Gradient-boosted decision trees                  |
| Gradient Boosting            | Sequential ensemble learning                     |

## Model Evaluation

The models are evaluated using multiple metrics rather than relying solely on accuracy.

* **Accuracy:** Overall proportion of correct predictions.
* **Recall:** Proportion of actual churned customers correctly identified.
* **F1 Score:** Balance between precision and recall.
* **ROC-AUC:** Ability to distinguish between churned and non-churned customers across classification thresholds.

Recall is particularly important when missing a customer who is likely to churn is costly. Precision should also be considered because retention campaigns have limited budgets.

### Model Comparison

The following table summarizes the reported results from the project.

| Model                  | Accuracy | Recall | F1 Score | ROC-AUC |
| ---------------------- | -------: | -----: | -------: | ------: |
| Logistic Regression    |   0.7037 | 0.6832 |   0.4730 |  0.7641 |
| Random Forest          |   0.8620 | 0.4144 |   0.5390 |  0.8524 |
| K-Nearest Neighbors    |   0.7523 | 0.6678 |   0.5121 |  0.7766 |
| Support Vector Machine |   0.7857 | 0.6627 |   0.5462 |  0.8225 |
| XGBoost                |   0.8330 | 0.6096 |   0.5870 |  0.8418 |
| Gradient Boosting      |   0.8170 | 0.7003 |   0.5984 |  0.8598 |

*Metrics are reported from the existing model evaluation and should be reproduced against the final code and dataset before being treated as verified results.*

### Key Results

* **Gradient Boosting** achieved the highest reported F1 score of approximately 0.5984 and ROC-AUC of approximately 0.8598.
* **XGBoost** achieved the second-highest reported F1 score, approximately 0.5870.
* **Random Forest** achieved the highest reported accuracy at 86.20%, but its recall was lower than that of Gradient Boosting.
* **Gradient Boosting** identified a larger proportion of actual churn cases than the other listed models, based on the reported recall of approximately 70.03%.

Based on these results, Gradient Boosting is the strongest candidate among the evaluated models when balancing precision and recall. The final model choice should also consider business costs, validation results, and performance on unseen customers.

## Business Insights & Recommendations

The analysis can support customer retention planning in several ways:

1. **Identify at-risk customers:** Use predicted churn probabilities to prioritize customers who may need additional attention.
2. **Investigate customer activity:** Explore whether inactivity and product usage patterns are associated with higher churn.
3. **Segment customers:** Compare churn behavior across age groups, regions, tenure, and product ownership.
4. **Design targeted retention strategies:** Develop suitable interventions for customer segments with elevated churn risk.
5. **Monitor model performance:** Track precision, recall, and ROC-AUC as new customer data becomes available.

These are recommended applications of the analysis, not claims that a retention campaign has already been implemented or that churn has been reduced.

## How to Run the Project

1. Clone or download the repository.
2. Install Python and the required libraries.
3. Open the Jupyter Notebook included in the repository.
4. Ensure the dataset is available at the file path specified in the notebook.
5. Run the notebook cells in sequence to explore the data, train the models, and review evaluation metrics.

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn jupyter
```

## Skills Demonstrated

* Data cleaning and preprocessing
* Exploratory data analysis
* Statistical analysis and data visualization
* Feature engineering and categorical encoding
* Machine learning classification
* Imbalanced dataset handling
* Model evaluation and comparison
* Translating analytical results into business recommendations

## Future Improvements

* Perform cross-validation and systematic hyperparameter tuning.
* Evaluate precision-recall curves and confusion matrices.
* Compare feature importance and explain model predictions.
* Add a reproducible preprocessing pipeline.
* Validate model performance on unseen data.
* Build an interactive dashboard to communicate churn insights.
* Develop a cost-sensitive retention strategy using predicted churn probabilities.

## Conclusion

This project demonstrates an end-to-end approach to customer churn analysis, from exploratory data analysis and preprocessing to classification model comparison and business interpretation.

By combining Python, statistical exploration, and machine learning evaluation, the project illustrates how customer data can help businesses understand attrition patterns and make better-informed retention decisions.

---

**Focus Areas:** Data Analytics | Python | Machine Learning | Customer Churn | Predictive Analytics | Customer Retention
