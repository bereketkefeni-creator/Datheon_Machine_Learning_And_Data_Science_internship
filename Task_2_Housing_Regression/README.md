# California Housing Price Prediction

## Project Overview

This project develops and evaluates machine learning regression models to predict California district median house values using demographic, geographic, and housing-related features.

The objective is to build a reliable regression pipeline, compare different modeling approaches, investigate feature relationships, and improve predictive performance through feature engineering and hyperparameter tuning.

This project was completed as part of the Datheon Machine Learning & Data Science Internship.

## Dataset

**Dataset:** California Housing Dataset

**Shape:** 20,640 rows × 10 columns

**Target variable:** `median_house_value`

The dataset contains information about California housing districts, including:

- Geographic coordinates
- Median housing age
- Total rooms and bedrooms
- Population and households
- Median income
- Ocean proximity

### Data Quality Findings

- 207 missing values were found in `total_bedrooms`.
- No duplicate rows were detected.
- `ocean_proximity` contains five categories.
- The target variable is capped at $500,001.

The target represents district-level median house value, not the price of an individual house.

## Project Workflow

### 1. Exploratory Data Analysis

- Examined dataset dimensions and data types.
- Investigated missing values and duplicate records.
- Analyzed numerical feature distributions.
- Examined categorical feature frequencies.
- Investigated relationships between predictors and the target.

### 2. Data Preprocessing

The dataset was split into training and testing sets using an 80/20 split.

Preprocessing was implemented using Scikit-learn pipelines and a ColumnTransformer.

**Numerical features:**
- Missing values were imputed using the median.
- Features were standardized using StandardScaler.

**Categorical features:**
- Missing values were imputed using the most frequent category.
- One-hot encoding was applied.
- Unknown categories were handled using `handle_unknown="ignore"`.

Preprocessing was fitted on the training data to prevent information leakage from the test set.

### 3. Feature Engineering

Three additional features were created:

| Feature | Calculation |
|---|---|
| Rooms per household | Total rooms / Households |
| Bedrooms per room | Total bedrooms / Total rooms |
| Population per household | Population / Households |

These features provide additional information about housing density and household composition.

### 4. Multicollinearity Analysis

Correlation analysis and Variance Inflation Factor (VIF) were used to investigate relationships among numerical predictors.

Strong correlations were observed among total rooms, total bedrooms, households, and population.

VIF analysis also indicated substantial multicollinearity among some predictors.

These findings motivated experiments with regularized regression models.

### 5. Model Development

The following regression models were implemented:

1. Linear Regression
2. Feature-Engineered Linear Regression
3. Ridge Regression
4. Lasso Regression
5. Random Forest Regression
6. Tuned Random Forest Regression

Ridge and Lasso were used to investigate the effect of regularization.

Random Forest was introduced to capture nonlinear relationships and feature interactions.

### 6. Hyperparameter Tuning

GridSearchCV was used to tune Ridge and Lasso regularization parameters.

RandomizedSearchCV was used to optimize Random Forest hyperparameters.

The Random Forest search used five-fold cross-validation and RMSE as its optimization metric.

**Best Random Forest parameters:**

- `max_depth`: 30
- `max_features`: sqrt
- `min_samples_leaf`: 3
- `min_samples_split`: 6
- `n_estimators`: 300

**Best cross-validation RMSE:** $49,893.45

## Model Evaluation

Models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

### Model Comparison

| Model | MAE ($) | RMSE ($) | R² |
|---|---:|---:|---:|
| Linear Regression | 50,670.49 | 70,059.19 | 0.6300 |
| Feature-Engineered Linear Regression | 49,645.49 | 69,127.04 | 0.6353 |
| Ridge Regression | 49,652.14 | 69,134.97 | 0.6353 |
| Lasso Regression | 49,646.43 | 69,128.09 | 0.6353 |
| Random Forest | 31,766.87 | 49,499.52 | 0.8130 |
| Tuned Random Forest | 32,538.54 | 49,189.78 | 0.8154 |

### Best Model

The tuned Random Forest achieved:

- **MAE:** $32,538.54
- **RMSE:** $49,189.78
- **R²:** 0.8154

The model explains approximately 81.54% of the variation in the test target values.

Although the original Random Forest achieved a lower MAE, the tuned model achieved a lower RMSE and slightly higher R².

The tuning objective was RMSE, so the tuned model was selected based on its cross-validation performance.

## Feature Importance

The tuned Random Forest identified the following important features:

| Feature | Importance |
|---|---:|
| Median Income | 0.3034 |
| Ocean Proximity: Inland | 0.1465 |
| Population per Household | 0.1014 |
| Bedrooms per Room | 0.0949 |
| Longitude | 0.0807 |
| Latitude | 0.0737 |

Median income was the most influential feature according to the model's impurity-based feature importance.

Feature importance indicates predictive contribution within this model; it does not establish causation.

## Visualizations

The `assets/` directory contains visualizations of the experiments, including:

- Exploratory data analysis
- Correlation analysis
- Model comparison
- Feature importance
- Actual vs. predicted values
- Residual analysis

These visualizations help evaluate model performance and identify potential weaknesses.

## Limitations

Several limitations should be considered:

1. **Target-value ceiling:** The target is capped at $500,001, limiting the model's ability to learn differences among higher-value districts.
2. **Prediction uncertainty:** Prediction errors become more dispersed at higher predicted values.
3. **Multicollinearity:** Some housing and population variables contain overlapping information.
4. **Model overfitting:** The Random Forest showed a noticeable difference between training and test performance.
5. **Feature importance:** Impurity-based importance can be influenced by correlated predictors and should not be interpreted as causation.
6. **Evaluation methodology:** The test set was examined during exploratory model comparisons. Therefore, it should not be described as a completely untouched final holdout set.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Project Structure

```text
Task_2_Housing_Regression/
│
├── data/
│   └── housing.csv
│
├── notebooks/
│   └── housing_regression.ipynb
│
├── assets/
│   ├── model_comparison.csv
│   ├── model_comparison_rmse.png
│   ├── model_comparison_r2.png
│   ├── model_comparison_mae.png
│   ├── feature_importance.png
│   ├── random_forest_actual_vs_predicted.png
│   └── random_forest_residuals.png
│
└── README.md
```

## How to Run

1. Clone the repository.
2. Install the required Python libraries.
3. Open the notebook using Jupyter Notebook or JupyterLab.
4. Run the notebook cells sequentially.

Install the main dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

## Conclusion

This project explored multiple regression techniques for California housing-value prediction.

The experiments showed that feature engineering provided modest improvements to linear models, while Random Forest captured nonlinear relationships more effectively.

Hyperparameter tuning produced a small improvement in RMSE, although it did not improve every evaluation metric.

The project demonstrates a complete regression workflow involving data exploration, preprocessing, feature engineering, multicollinearity analysis, model comparison, hyperparameter optimization, and error analysis.