# House Price Prediction using Linear Regression

A project implementing and comparing **Simple Linear Regression** and **Multiple Linear Regression** using both **Ordinary Least Squares (OLS)** and **Gradient Descent (GD)** on the King County housing dataset.

---

## Project Overview

The goal of this project is to predict residential house prices in King County, Washington (including Seattle) using linear regression models. 

We compare two fundamental approaches to fitting linear models:
1. **Ordinary Least Squares (OLS)**: The closed-form analytical method used in scikit-learn's `LinearRegression`.
2. **Gradient Descent (GD)**: The iterative numerical optimization method used in scikit-learn's `SGDRegressor`.

We also run diagnostic tests on the core linear regression assumptions:
- Linearity between features and price
- Independence of residual errors
- Homoscedasticity (constant variance of errors)
- Normality of error distribution
- Multicollinearity among features (using correlation matrices and Variance Inflation Factor)

---

## Project Structure

```text
Linear-Regression/
├── .gitignore                      # Git ignore rules (.venv, cache, checkpoints)
├── .python-version                 # Pinned Python version (3.14)
├── pyproject.toml                  # Project dependencies managed by uv
├── uv.lock                         # Locked reproducible dependencies
├── README.md                       # Project documentation
│
├── data/
│   ├── raw/
│   │   └── kc_house_data.csv       # Original King County House Sales dataset
│   └── processed/
│       ├── kc_house_simple_lr.csv  # Subset for Simple Linear Regression (sqft_living, price)
│       └── kc_house_multiple_lr.csv# 7 selected features for Multiple Linear Regression
│
├── notebooks/
│   ├── 01_simple_linear_regression_ols.ipynb   # Simple LR using OLS
│   ├── 02_simple_linear_regression_gd.ipynb    # Simple LR using Gradient Descent
│   ├── 03_multiple_linear_regression_ols.ipynb # Multiple LR using OLS + assumption tests
│   └── 04_multiple_linear_regression_gd.ipynb  # Multiple LR using Gradient Descent
│
└── src/
    └── linear_regression/
        └── __init__.py             # Package init
```

---

## Notebooks Walkthrough

| Notebook | Focus | Summary |
| :--- | :--- | :--- |
| [`01_simple_linear_regression_ols.ipynb`](notebooks/01_simple_linear_regression_ols.ipynb) | Simple Linear Regression (OLS) | Predicts price using living area (`sqft_living`). Covers train/test split, model fitting, evaluation (MAE, RMSE, R2), regression plotting, and assumption testing. |
| [`02_simple_linear_regression_gd.ipynb`](notebooks/02_simple_linear_regression_gd.ipynb) | Simple Linear Regression (GD) | Solves the same single-feature problem using `StandardScaler` + `SGDRegressor`. Shows why scaling is necessary and compares results directly with OLS. |
| [`03_multiple_linear_regression_ols.ipynb`](notebooks/03_multiple_linear_regression_ols.ipynb) | Multiple Linear Regression (OLS) | Expands the model to 7 features (living area, bedrooms, floors, grade, lot size, year built, waterfront). Checks all 5 regression assumptions including VIF. |
| [`04_multiple_linear_regression_gd.ipynb`](notebooks/04_multiple_linear_regression_gd.ipynb) | Multiple Linear Regression (GD) | Trains the 7-feature model using Gradient Descent. Analyzes feature importance from standardized weights and compares all 4 models in a final benchmark. |

---

## Results Summary Across Models

| Model | Method | Features | MAE | RMSE | R2 Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Simple LR | OLS | `sqft_living` | $177,868 | $276,559 | 0.4941 |
| Simple LR | Gradient Descent | `sqft_living` (scaled) | $179,025 | $276,504 | 0.4943 |
| Multiple LR | OLS | 7 features | $146,474 | $232,223 | 0.6433 |
| Multiple LR | Gradient Descent | 7 features (scaled) | $144,952 | $233,288 | 0.6400 |

### Key Takeaways:
- **Feature Addition**: Moving from a single feature (`sqft_living`) to 7 features increased R2 from 0.49 to 0.64, explaining 15% more variance and reducing average prediction errors by over $30,000.
- **Method Equivalence**: Gradient Descent finds virtually the exact same solution as OLS once features are properly scaled.
- **Assumptions in Practice**: Real-world house prices have high-value luxury outliers that violate homoscedasticity and normality, showing where linear models struggle with extreme values.

---

## Dataset Reference

- **Dataset**: King County House Sales Dataset (`kc_house_data.csv`)
- **Records**: 21,613 houses sold between May 2014 and May 2015 in King County, WA.
- **Source**: Available on [Kaggle: House Sales in King County, USA](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction).

---

## Getting Started

### Prerequisites
- Python >= 3.14
- [`uv`](https://github.com/astral-sh/uv) (recommended) or standard `pip`

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/Linear-Regression.git
cd Linear-Regression
```

### 2. Setup Environment

#### Using `uv` (Fastest):
```bash
uv sync
```

#### Using standard `pip`:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate.bat
# On Mac/Linux:
source .venv/bin/activate

pip install -e .
```

### 3. Running the Notebooks
```bash
# If using uv:
uv run jupyter notebook

# Or in an activated virtual environment:
jupyter notebook
```
Open any notebook in the `notebooks/` folder to run the code.
