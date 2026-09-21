# Restaurant Intelligence & Predictive Analytics System

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![Visualization](https://img.shields.io/badge/Plotly%20%7C%20Seaborn-Interactive-green.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

An end-to-end machine learning and data analytics project developed as part of the **Cognifyz Technologies Machine Learning Internship Program**. This repository contains predictive models, content-based filtering algorithms, multi-class classifiers, and spatial intelligence pipelines evaluated on global dining establishment records[cite: 1, 3, 5].

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Internship Tasks & Deliverables](#-internship-tasks--deliverables)
  - [Task 1: Restaurant Rating Prediction](#task-1-predict-restaurant-ratings)
  - [Task 2: Content-Based Restaurant Recommender](#task-2-restaurant-recommendation-system)
  - [Task 3: Primary Cuisine Classification](#task-3-cuisine-classification)
  - [Task 4: Location-Based Spatial Analysis](#task-4-location-based-analysis)
- [Dataset Architecture](#-dataset-architecture)
- [Key Performance Metrics & Findings](#-key-performance-metrics--findings)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [Technologies & Libraries Used](#-technologies--libraries-used)
- [Author & Acknowledgments](#-author--acknowledgments)

---

## 📖 Project Overview

The primary objective of this project is to leverage real-world restaurant metadata to extract predictive behavioral patterns, automate consumer decision-making, and uncover market opportunities. The system covers four core operational pillars:
1. **Continuous Target Prediction:** Forecasting continuous restaurant aggregate ratings from structured business attributes[cite: 1].
2. **Personalized Search & Discovery:** Powering user-personalized restaurant recommendations using text vectorization and cosine similarity matching[cite: 1, 3].
3. **Multiclass Supervised Classification:** Classifying establishments into cuisine categories based on operational profile indicators[cite: 1, 4].
4. **Geographical Intelligence:** Mapping spatial clustering, rating variance, and density patterns across global metropolitan regions[cite: 1, 5].

---

## 🚀 Internship Tasks & Deliverables

### Task 1: Predict Restaurant Ratings
* **Objective:** Predict the continuous `Aggregate rating` (0.0 – 5.0) using operational and geographical features[cite: 1, 2].
* **Methodology:**
  * Cleaned missing cuisine rows and converted boolean fields (`Has Table booking`, `Has Online delivery`, `Is delivering now`) into binary numeric values.
  * One-hot encoded high-cardinality nominal predictors (`City`)[cite: 2].
  * Trained and compared **Linear Regression**, **Decision Tree Regressor**, and **Random Forest Regressor** with an 80/20 train-test split[cite: 2].
  * Evaluated feature importances to determine top drivers influencing customer ratings[cite: 2].

### Task 2: Restaurant Recommendation System
* **Objective:** Build an intelligent recommendation engine matching diner preferences (city, desired cuisine, spending capability, rating thresholds)[cite: 1, 3].
* **Methodology:**
  * Aggregated textual metadata including `Restaurant Name`, sanitized `Cuisines`, `City`, and `Price range`.
  * Vectorized multi-attribute token strings using **TF-IDF Vectorization** (`TfidfVectorizer`) removing standard English stop words.
  * Calculated vector closeness via **Cosine Similarity** matrices[cite: 3].
  * Filtered candidates by minimum rating and maximum price boundary, returning ranked matches prioritized by similarity and total review volume (`Votes`)[cite: 3].

### Task 3: Cuisine Classification
* **Objective:** Predict the restaurant's primary cuisine category based on price tier, location coordinates, delivery services, and voting patterns[cite: 1, 4].
* **Methodology:**
  * Extracted normalized `primary_cuisine` labels and focused on the top 10 most frequent classes (e.g., North Indian, Chinese, Fast Food, Bakery, Cafe).
  * Built preprocessing pipelines including standard feature scaling (`StandardScaler`) and categorical dummy encoding[cite: 4].
  * Benchmarked **Logistic Regression** against **Random Forest Classifier**[cite: 4].
  * Evaluated weighted precision, recall, and F1-score to detect class imbalance challenges[cite: 4].

### Task 4: Location-Based Analysis
* **Objective:** Perform spatial exploration across geographical latitude and longitude coordinates[cite: 1, 5].
* **Methodology:**
  * Filtered anomalous and zeroed coordinates[cite: 5].
  * Plotted restaurant coordinate distributions across domestic and international cities using interactive geo-scatter plots[cite: 5].
  * Aggregated mean price tiers, average customer satisfaction scores, and cuisine variety across localities and cities[cite: 1, 5].

---

## 📊 Dataset Architecture

* **Source File:** `Dataset.csv`[cite: 2]
* **Total Records:** 9,551 restaurants[cite: 2]
* **Attributes:** 21 columns[cite: 2]

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `Restaurant ID` | Integer | Unique identifier for each establishment[cite: 2] |
| `Restaurant Name` | String | Commercial title of the restaurant[cite: 2] |
| `Country Code` | Integer | Geographic country indicator[cite: 2] |
| `City` / `Locality` | String | City and specific neighborhood zone[cite: 2] |
| `Longitude` / `Latitude` | Float | GPS geospatial coordinates[cite: 2] |
| `Cuisines` | String | Comma-separated food varieties served[cite: 2] |
| `Average Cost for two` | Integer | Estimated cost for a dining pair[cite: 2] |
| `Price range` | Integer | Ordinal pricing category (1 to 4)[cite: 2] |
| `Has Table booking` | Categorical | Availability of table reservation (`Yes`/`No`)[cite: 2] |
| `Has Online delivery` | Categorical | Online delivery capability (`Yes`/`No`)[cite: 2] |
| `Aggregate rating` | Float | Target rating variable (0.0 to 5.0)[cite: 2] |
| `Votes` | Integer | Cumulative customer review count[cite: 2] |

---

## 📈 Key Performance Metrics & Findings

### 1. Rating Prediction (Regression Task)
| Model | Mean Squared Error (MSE) | R² Score |
| :--- | :---: | :---: |
| Linear Regression | 1.4012 | 0.3881[cite: 2] |
| Decision Tree Regressor | 0.1776 | 0.9224[cite: 2] |
| **Random Forest Regressor** | **0.0942** | **0.9588**[cite: 2] |

> **Key Takeaway:** Random Forest demonstrated the highest predictive power ($R^2 \approx 95.9\%$)[cite: 2]. Top feature importance analysis revealed that **review count (`Votes`)**, **geospatial location (`Longitude`, `Latitude`)**, and **`Price range`** are the strongest determinants of a restaurant's rating[cite: 2].

### 2. Primary Cuisine Classification (Top 10 Classes)
| Classifier | Test Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 44.55% | 0.3298 | 0.4455 | 0.3012[cite: 4] |
| **Random Forest Classifier** | **44.90%** | **0.3881** | **0.4490** | **0.4021**[cite: 4] |

> **Key Takeaway:** Model performance reflected inherent real-world multi-class imbalance where dominant classes (e.g., North Indian) showed higher recall, while specialized categories presented misclassification overlap without textual menu data[cite: 4].

---

## 📁 Project Structure

```text
├── Dataset.csv                       # Master restaurant dataset
├── Task 1 - Rating Prediction.ipynb  # Task 1: Regression modeling & feature importance
├── Task 2 - Recommendation.ipynb     # Task 2: Content-based filtering engine
├── Task 3 - Cuisine Classifier.ipynb # Task 3: Multi-class cuisine categorization
├── Task 4 - Location Analysis.ipynb  # Task 4: Geospatial & locality analytics
├── requirements.txt                  # Python dependencies
└── README.md                         # Complete project documentation
