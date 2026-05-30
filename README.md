# 📊 Predictive Modeling for Business Score Classification Using GAMs and Decision Trees

> **Classifying online business scores (High vs. Low) across three field-segmented subsets using Lasso feature selection, Generalised Additive Models (GAM), and Decision Trees — with ROC curves, confusion matrices, and cross-validation**

[![Language](https://img.shields.io/badge/Language-R-276DC3?style=for-the-badge&logo=r)](https://www.r-project.org/)
[![Models](https://img.shields.io/badge/Models-GAM%20%7C%20Decision%20Tree%20%7C%20Lasso-orange?style=for-the-badge)]()
[![Framework](https://img.shields.io/badge/Framework-Tidymodels-1a9850?style=for-the-badge)]()

---

## 🔍 Project Overview

This project applies advanced machine learning to **classify online business scores** as either **High** (score > 3) or **Low** (score ≤ 3) using the `student_merge_platform_business_file_final15` dataset. The analysis spans three field-segmented data subsets, applying a full ML pipeline: exploratory data analysis, Lasso-based feature selection, GAM modelling for non-linear relationships, and Decision Tree classification with hyperparameter tuning.

The core research question: *Which model — GAM or Decision Tree — better predicts business performance scores, and what factors drive those predictions?*

---

## 📊 Key Results at a Glance

### Decision Tree Model Performance

| Subset | Accuracy | Sensitivity | Specificity |
|---|---|---|---|
| Subset 1 (Field 59) | **62%** | **94%** | 29% |
| Subset 2 (Field 18) | **56%** | **72%** | 46% |
| Subset 3 (Field 45) | **65%** | **78%** | **55%** |

### GAM Model Performance

| Model | Subset | Adj. R² | Deviance Explained | Key Predictors |
|---|---|---|---|---|
| GAM_MODEL1 | Field 59 | 0.0255 | **2.24%** | `state_NJ`, `city_Philadelphia`, `field_cat_X49` |
| GAM_MODEL2 | Field 18 | 0.0213 | **1.90%** | `state_NJ`, `state_AZ`, `CEO_grd_yr_X2000` |
| GAM_MODEL3 | Field 45 | 0.0157 | **1.61%** | `CEO_grd_yr_X2009`, `CEO_grd_yr_X2004` |

> **Key finding:** Decision Trees outperform GAMs for this classification task. Subset 1's Decision Tree achieved 94% sensitivity — critical for catching high-risk business classifications. GAMs, while interpretable, explained only 1.6–2.2% of score variance, suggesting that non-linear additive terms alone are insufficient for this dataset's complexity.

---

## 🎯 Problem Statement

Online business platforms generate large volumes of performance scores that need to be categorised for strategic decision-making. Manual score classification is inconsistent and unscalable. This project builds a **binary classifier** to automate business score prediction, enabling:

- Automated identification of high vs. low performing businesses
- Data-driven regional and leadership insights for strategic planning
- Scalable scoring across different business field categories

**Target variable:** `category_score`
- `high` → score > 3
- `low` → score ≤ 3

---

## 📦 Dataset

| Property | Detail |
|---|---|
| **File** | `student_merge_platform_business_file_final15.csv` |
| **Target variable** | `category_score` (binary: high / low) |
| **Train/Test split** | 75% / 25% (stratified by `category_score`) |
| **Key predictors** | State, city, field category, CEO graduation year, CEO school category, gender, rural/metropolitan classification |
| **Preprocessing** | KNN imputation (numeric), mode imputation (categorical), normalisation, correlation filtering (threshold 0.9), dummy encoding, zero-variance removal |

### Data Segmentation Strategy
Three subsets created by filtering on `field_cat` to test model performance across different business field contexts:

```
Subset 1 (Field 59): field_cat ∈ {59, 51, 28, 57, 45, 76, 75, 69, 55, 49}
Subset 2 (Field 18): field_cat ∈ {5, 46, 60, 59, 48, 41, 62, 58, 33}
Subset 3 (Field 45): field_cat ∈ {67, 61, 43, 23, 47, 18, 10, 3}
```

---

## 🔬 Methodology

### Step 1 — Exploratory Data Analysis
- Missing value summary using `colSums(is.na())` and custom `summarize_blanks_zeros()` function
- `SmartEDA` for automated categorical variable exploration
- Key EDA insights:
  - Strong state-level variation in business scores (PA, FL, AZ most prominent)
  - Gender disparity in dataset — males more prevalent
  - Predominantly metropolitan business locations

### Step 2 — Target Variable Engineering
```r
category_score = case_when(
  score <= 3 ~ "low",
  score >  3 ~ "high"
)
```
Original numeric score (1–5) converted to binary classification for model compatibility.

### Step 3 — Lasso Feature Selection
- `tidymodels` workflow with `multinom_reg()` and `glmnet` engine
- `mixture = 1` (full Lasso penalty), `penalty` tuned via 5-fold CV
- 30-level penalty grid searched from 10⁻³ to 10¹
- `select_by_one_std_err()` used for robust lambda selection
- Top 10 variables per subset extracted using `vip()` (variable importance plots)

**Top predictors identified across subsets:**

| Subset | Top Variables |
|---|---|
| Subset 1 | `city_Tucson`, `state_AZ`, `state_NJ`, `city_Philadelphia`, `state_PA`, `field_cat_X49`, `CEO_grd_yr_X2015` |
| Subset 2 | `state_NJ`, `state_AZ`, `CEO_grd_yr_X2000`, `state_FL`, `state_PA`, `city_Tampa` |
| Subset 3 | `CEO_grd_yr_X2009`, `CEO_grd_yr_X2004`, `state_TN`, `CEO_sch_cat_X210`, `field_cat_X61` |

### Step 4 — Generalised Additive Models (GAM)
- `mgcv::gam()` with `family="binomial"` and `method="REML"`
- Lasso-selected top 10 features used as formula inputs per subset
- Model summaries printed; smooth term plots generated
- Error handling implemented for subsets with no plottable smooth terms

### Step 5 — Decision Tree Classification
- `tidymodels` workflow with `rpart` engine
- Three hyperparameters tuned simultaneously: `cost_complexity`, `tree_depth`, `min_n`
- 10-fold stratified cross-validation for tuning
- Random grid search with 5 candidates (`grid_random()`)
- Best model selected by accuracy; final fit evaluated on held-out test set
- Outputs: ROC curve, confusion matrix mosaic, confusion matrix heatmap

---

## 📈 Visualisations Produced

- Lasso regularisation path plots (ROC-AUC vs. penalty)
- Variable importance (VIP) bar charts — top 10 features per subset
- Score and category score distribution plots (pre/post binarisation)
- GAM smooth term plots per subset
- Decision Tree ROC curves with AUC
- Confusion matrix mosaic and heatmap for each subset
- Missing values barplot

---

## 🏆 Model Comparison: GAM vs. Decision Tree

| Criterion | GAM | Decision Tree |
|---|---|---|
| **Accuracy** | Not directly comparable | 56–65% |
| **Sensitivity** | Low | Up to **94%** |
| **Deviance Explained** | 1.61–2.24% | — |
| **Interpretability** | High (coefficient-level) | Medium (tree structure) |
| **Non-linearity handling** | Yes (smooth terms) | Yes (recursive splits) |
| **Winner** | ❌ | ✅ |

**Conclusion:** Decision Trees outperform GAMs for accuracy and sensitivity. However, GAMs provide valuable interpretability through coefficient-level insights into geographic and leadership effects. Future work should explore **Random Forests** or **gradient boosting** to combine Decision Tree strength with ensemble robustness.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **R 4.2** | Core programming language |
| `tidymodels` | End-to-end ML workflow (split, recipe, tune, fit, evaluate) |
| `glmnet` | Lasso regularisation engine |
| `mgcv` | Generalised Additive Models (GAM) with REML |
| `rpart` | Decision Tree engine |
| `vip` | Variable importance plots |
| `tidyverse` | Data manipulation pipeline |
| `ggplot2` | Visualisations |
| `naniar` | Missing data visualisation |
| `SmartEDA` | Automated EDA |
| `viridis` | Colour palettes |
| `embed` | Factor encoding support |
| `RColorBrewer` | Colour schemes |
| `knitr` / `kableExtra` | Publication-quality tables |
| `xtable` | LaTeX/HTML table output |

---

## 📁 Repository Structure

```
PREDICTIVE-MODELING-FOR-BUSINESS-SCORE-CLASSIFICATION-USING-GAMS-AND-DECISION-TREES/
│
├── AA_Assignment2_Done.Rmd    # Full analysis source code
└── README.md
```

---

## 🚀 How to Reproduce

```r
# Install required packages
install.packages(c("tidyverse", "naniar", "knitr", "xtable", "RColorBrewer",
                   "SmartEDA", "ggplot2", "gam", "viridis", "glmnet",
                   "vip", "embed", "tidymodels"))

# Place the dataset CSV in your working directory:
# "student_merge_platform_business_file_final15.csv"

# Open AA_Assignment2_Done.Rmd in RStudio and knit to Word
# or run chunks sequentially for interactive analysis
```

---

## 💡 Key Insights & Conclusions

1. **Geography is the strongest predictor** — State variables (`state_NJ`, `state_AZ`, `state_PA`) appear as top predictors across all three subsets, confirming that regional economic conditions and regulatory environments are primary drivers of business score classification.

2. **Leadership era matters** — CEO graduation year variables (`CEO_grd_yr_X2000`, `CEO_grd_yr_X2015`) consistently appear in Lasso-selected features, suggesting that the business environment and leadership training of a particular era significantly influences long-term business performance.

3. **Decision Trees beat GAMs for this task** — While GAMs offer interpretability, their low deviance explained (1.6–2.2%) suggests the relationships are not well-captured by additive smooth terms. Decision Trees achieve up to 94% sensitivity, making them far more practically useful for catching high-scoring businesses.

4. **High sensitivity is the priority** — With Decision Tree Subset 1 achieving 94% sensitivity at the cost of 29% specificity, the model is calibrated to avoid false negatives (missing high performers) — appropriate for business intelligence applications where identifying top performers is critical.

5. **Lasso as a pipeline, not just a model** — Rather than using Lasso predictions directly, this project uses Lasso purely for feature selection — feeding its top 10 variables into GAM and Decision Tree models. This two-stage approach improves interpretability while maintaining predictive power.

6. **Ensemble methods recommended for future work** — Random Forests or XGBoost could substantially improve on the 56–65% accuracy range, combining the Decision Tree's strength with ensemble variance reduction.

---

## 👨‍💻 Author

**Siddharth Shekhar Singh**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/siddharth-shekhar-singh)

---

*This project was completed as part of an Advanced Analytics and Machine Learning assignment. Full source code is available in the .Rmd file above.*
