# Life Churn Analysis

This project analyzes customer churn data and builds predictive models using Python.  
It uses **Poetry** for dependency management and can be run in **Jupyter Lab**.

---

## 📦 Setup

### 1. Clone the repository
```bash
git clone https://github.com/KaiWH1/Churn-Life.git
cd Churn-Life
```

### 2. Install dependencies
```bash
poetry install
```
If you only want to manage dependencies (no packaging mode):
```bash
poetry install --no-root
```

### 3. Create a Jupyter kernel
```bash
poetry run python -m ipykernel install --user --name=Kai-Wa-Ho-Quiz
```

### 4. Launch Jupyter Lab
```bash
poetry run jupyter lab
```

---

## 📂 Data

Place your **`merged_data.xlsx`** file in the `/data` folder before running the notebook or scripts.  
It should contain:
- A **"Churned Customers"** sheet
- A combined sheet with all customers

---

## 📊 Analysis

The analysis includes:
- Data cleaning & preprocessing
- Churn trend visualization
- Churn reason distribution
- Feature impact analysis
- Predictive modeling (Logistic Regression, Random Forest, XGBoost, Decision Tree, EBM)
- Regression analysis for churn timing

---

## ▶ How to Run

1. Open **`churn_analysis.ipynb`** (or your analysis notebook) in Jupyter Lab.
2. Select the **Kai-Wa-Ho-Quiz** kernel.
3. Run all cells (`Cell > Run All`).
4. View plots and model results directly in the notebook.

---

## 🛠 Dependencies

Main Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- scikit-learn
- xgboost
- interpret
- lifelines
- statsmodels
- poetry

(See `pyproject.toml` for the full list.)

---

## 📌 Notes
- Requires **Python 3.9 – 3.12** (per `pyproject.toml` settings)
- Tested on macOS with Poetry 1.8+
- If new packages are needed, add them via:
```bash
poetry add package_name
```
Then restart your Jupyter kernel.

---

## 📄 License
This project is for educational and research purposes.