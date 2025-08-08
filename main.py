import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import os # type: ignore
import statsmodels.api as sm # type: ignore
import matplotlib.pyplot as plt # type: ignore
import xgboost as xgb # type: ignore
import seaborn as sns # type: ignore
import plotly.express as px # type: ignore
from lifelines import KaplanMeierFitter, CoxPHFitter # type: ignore
from itertools import combinations, product # type: ignore
from interpret import show # type: ignore
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree # type: ignore
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score # type: ignore
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor # type: ignore
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV, RidgeCV # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from statsmodels.stats.outliers_influence import variance_inflation_factor # type: ignore


st.sidebar.title("Navigation")
section = st.sidebar.selectbox("Choose Section", ["Data Exploration", "Modeling", "Extra"])

customer_data = pd.read_csv("customer_info.csv")
engagement_data = pd.read_csv("engagement.csv")
labels = pd.read_csv("labels.csv")
policy_data = pd.read_csv("policy_data.csv")
transaction = pd.read_csv("transactions.csv")

# 1. Merging the dataset
dfs = [customer_data, engagement_data, labels, policy_data]

merged_df = dfs[0]
for df in dfs[1:]:
    merged_df = pd.merge(merged_df, df, on = 'customer_id', how = 'left')

merged_df['last_login_date'] = pd.to_datetime(merged_df['last_login_date'])
merged_df['policy_start_date'] = pd.to_datetime(merged_df['policy_start_date'])

merged_df['days_between_policy_last_login'] = (
    merged_df['last_login_date'] - merged_df['policy_start_date']
).dt.days

# Add churn_date and days_between_policy_churn calculation
merged_df['churn_date'] = pd.to_datetime(merged_df['churn_date'], errors='coerce')
merged_df['days_between_policy_churn'] = (
    merged_df['churn_date'] - merged_df['policy_start_date']
).dt.days

def classify_churn_return(row):
    if pd.isna(row['days_between_policy_churn']):
        return "Not Returning"
    elif row['days_between_policy_last_login'] > row['days_between_policy_churn']:
        return "Possibly Returning"
    else:
        return "Possibly Churn"

merged_df['churn_behavior'] = merged_df.apply(classify_churn_return, axis=1)

# st.title("Merged Data Overview")

# st.write(merged_df)

# 2. Check NAs
cols_to_check = [
    col for col in merged_df.columns if col not in ['churn_date', 'churn_reason']
]
na_counts = merged_df[cols_to_check].isna().sum()
# print(na_counts)
# No NA

if section == "Data Exploration":
    # 3. Churn Rate
    pattern_dfs = {}

    drop_cols = ['churned','churn_date', 'churn_reason', 'last_login_date', 'policy_start_date', 'customer_id', 'churn_behavior', 'transaction_date', 'transaction_type', 'amount', 'payment_status', 'days_overdue']

    churn_vars = [col for col in merged_df.columns if col not in drop_cols]

    for col in churn_vars:
        if merged_df[col].dtype in ['object', 'bool']:
            grouped = merged_df.groupby(col)["churned"].mean().reset_index()
            grouped.columns = ["value", "churn_rate"]
        else:
            try:
                binned, bins = pd.qcut(merged_df[col], q=5, retbins=True, duplicates='drop')
                binned = pd.cut(merged_df[col], bins=bins, include_lowest=True)
                grouped = merged_df.groupby(binned)["churned"].mean().reset_index()
                grouped.columns = ["value", "churn_rate"]
            except ValueError:
                grouped = merged_df.groupby(col)["churned"].mean().reset_index()
                grouped.columns = ["value", "churn_rate"]
        pattern_dfs[col] = grouped


    cleaned_dfs = []
    for feature, df in pattern_dfs.items():
        df = df.copy()
        df.columns = ["value", "churn_rate"]
        df["feature"] = feature
        cleaned_dfs.append(df)

    churn_pattern_df = pd.concat(cleaned_dfs, ignore_index=True)[["feature", "value", "churn_rate"]]

    st.title("Churn Pattern Summary - Single Segment")

    unique_features = ["All"] + churn_pattern_df["feature"].unique().tolist()
    selected_feature = st.selectbox("Select a feature to view churn pattern", unique_features)

    if selected_feature == "All":
        filtered_df = churn_pattern_df.copy()
    else:
        filtered_df = churn_pattern_df[churn_pattern_df["feature"] == selected_feature]
    filtered_df["value"] = filtered_df["value"].astype(str)

    st.dataframe(filtered_df.style.format({"churn_rate": "{:.2%}"}))


    st.title("Churn Pattern Summary - Cross Segments")

    min_sample_threshold = st.slider(
        "Minimum sample count per segment",
        min_value=500,
        max_value=1000,
        value=500,
        step=100
    )

    cross_segment_results = []

    # Can be adjusted to more if more computational resources are available
    for r in range(1, min(4, len(churn_vars)) + 1):
        for combo in combinations(churn_vars, r):
            try:
                group = merged_df.groupby(list(combo)).agg(
                    count = ('churned', 'count'),
                    churn_rate = ('churned', 'mean')
                ).reset_index()
                group = group[group["count"] > min_sample_threshold]
                if not group.empty:
                    group['variables'] = ', '.join(combo)
                    cross_segment_results.append(group)
            except Exception as e:
                continue

    if cross_segment_results:
        cross_df = pd.concat(cross_segment_results, ignore_index=True)

        value_cols = [col for col in cross_df.columns if col not in ['count', 'churn_rate', 'variables']]
        cross_df["segment_values"] = cross_df.apply(
        lambda row: " | ".join(f"{col}:{row[col]}" for col in value_cols if pd.notna(row[col])), axis=1
        )
        display_df = cross_df[["segment_values", "churn_rate"]].copy()
        top_display_df = display_df.sort_values("churn_rate", ascending=False).head(30)
        
        st.write("Top 30 High-Churn Cross Segments")
        st.dataframe(top_display_df.style.format({"churn_rate": "{:.2%}"}))
    else:
        st.write("No high churn cross segments found.")

    st.title("Churn Reason")

    selected_reason = st.selectbox(
        'Select a churn reason',
        merged_df['churn_reason'].dropna().unique(),
        key = 'reason_select'
    )

    reason_df = merged_df[merged_df['churn_reason'] == selected_reason]
    all_churned_df = merged_df[merged_df['churned'] == 1]
    profile_rows = []
    all_features = [col for col in merged_df.columns if col not in drop_cols]

    for col in all_features:
        try:
            if merged_df[col].dtype in ['object', 'bool']:
                for val in merged_df[col].dropna().unique():
                    p1 = (reason_df[col] == val).mean()
                    p2 = (all_churned_df[col] == val).mean()
                    diff = p1 - p2
                    profile_rows.append({
                        "Feature": f"{col} = {val}",
                        "Reason Percentage": p1,
                        "All Churned Percentage": p2,
                        "Difference": diff
                    })
        except Exception:
            continue

    profile_df = pd.DataFrame(profile_rows)
    profile_df[["Reason Percentage", "All Churned Percentage", "Difference"]] = profile_df[["Reason Percentage", "All Churned Percentage", "Difference"]].applymap(lambda x: f"{x:.0%}")

    st.dataframe(profile_df.sort_values("Difference", ascending=False))

    # 4. Feature Engineering
    # Policy Tenure
    merged_df['policy_tenure'] = (pd.to_datetime('2024-12-31') - merged_df['policy_start_date']).dt.days

    # Transaction Dataset
    transaction['transaction_date'] = pd.to_datetime(transaction['transaction_date'], errors='coerce')
    premium_tx = transaction[transaction['transaction_type'] == 'Premium'].copy()

    payment_features = premium_tx.groupby('customer_id').agg(
        num_transactions = ('payment_status', 'count'),
        num_failed_payments = ('payment_status', lambda x: (x != "Success").sum()),
        avg_days_overdue = ('days_overdue', lambda x: x[x > 0].mean()),
        max_days_overdue = ('days_overdue', 'max'),
        sum_success_payment = ('amount', lambda x: x[premium_tx.loc[x.index, 'payment_status'] == 'Success'].sum()),
    ).reset_index().fillna(0)

    payment_features['failed_payment_rate'] = payment_features['num_failed_payments'] / payment_features['num_transactions']
    payment_features['has_chronic_overdue'] = premium_tx[premium_tx['days_overdue'] > 0].groupby('customer_id').size().ge(3).astype(int).reindex(payment_features['customer_id']).fillna(0).astype(int).values

    merged_df = pd.merge(merged_df, payment_features, on = 'customer_id', how = 'left')

    st.title("Data Summary with Features")

    # Engagement Score
    def calculate_engagement_score(row):
        score = 0
        if row['login_frequency_30d'] >= 5:
            score += 2
        elif row['login_frequency_30d'] >= 2:
            score += 1
        
        if row['mobile_app_user'] == "Yes":
            score += 2
        
        if row['email_opens_6m'] >= .6:
            score += 2
        elif row['email_opens_6m'] >= .3:
            score += 1
        
        if row['customer_service_calls_12m'] >= 3:
            score -= 2
        elif row['customer_service_calls_12m'] >= 1:
            score -= 1
        
        if row['complaints_filed'] >= 2:
            score -= 2
        
        return score

    merged_df['engagement_score'] = merged_df.apply(calculate_engagement_score, axis=1)

    with pd.ExcelWriter('merged_data.xlsx', engine = 'openpyxl') as writer:
        merged_df.to_excel(writer, sheet_name = 'All Data', index = False)
        merged_df[merged_df['churn_reason'].notna()].to_excel(writer, sheet_name = "Churned Customers", index = False)

    st.dataframe(merged_df)

elif section == "Modeling":
    merged_df = pd.read_excel('merged_data.xlsx', sheet_name='All Data')
    st.title("Logistic Regression Model")
    
    num_vars = [
        'age', 'dependents', 'login_frequency_30d', 'email_opens_6m', 'customer_service_calls_12m',
        'complaints_filed', 'policy_amount', 'premium_amount', 'riders', 'days_between_policy_last_login',
        'engagement_score'
    ]

    cat_vars = [
        'gender', 'marital_status', 'income_bracket', 'employment_status', 'education_level',
        'mobile_app_user', 'policy_type', 'premium_frequency', 'payment_method', 'churn_behavior'
        
    ]

    for col in num_vars:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

    for col in cat_vars:
        merged_df[col] = merged_df[col].astype(str)

    merged_df = pd.get_dummies(merged_df, columns=cat_vars, drop_first=True)

    drop_due_to_vif = [
    'premium_amount',
    'policy_amount',
    'policy_tenure',
    'max_days_overdue',
    'avg_days_overdue',
    'age',
    'premium_frequency_Monthly',
    'num_transactions',
    'engagement_score',
    'email_opens_6m'
    ]

    x = merged_df.drop(columns = ['churned', 'customer_id', 'churn_date', 'policy_start_date', 
                                  'churn_reason', 'last_login_date', 'days_between_policy_churn']).drop(columns = drop_due_to_vif)

    # Remove one-hot encoded churn_behavior columns
    churn_behavior_cols = [col for col in x.columns if col.startswith('churn_behavior_')]
    x = x.drop(columns=churn_behavior_cols)
    x = x.dropna(axis=0)

    x = x.astype(float)

    y = merged_df['churned'].astype(int)
    y = y.loc[x.index]

    model = sm.Logit(y, x)
    results = model.fit()
    st.subheader("Logistic Regression Model Summary (Before VIF Removal)")
    st.write(results.summary())

    # Find significant variables
    p_values = results.pvalues
    sig_var = p_values[p_values < 0.05].index.tolist()

    x_del = x[sig_var]

    # Build dataset on significant variables
    x = x_del.copy()
    y = y.loc[x.index]

    sorted_indices = merged_df.loc[x.index].sort_values("policy_start_date").index
    x = x.loc[sorted_indices]
    y = y.loc[sorted_indices]

    n = len(x)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    x_train = x.iloc[:train_end]
    y_train = y.iloc[:train_end]

    x_val = x.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]

    x_test = x.iloc[val_end:]
    y_test = y.iloc[val_end:]




    # Fit a model
    log_reg_model = sm.Logit(y_train, x_train)
    log_reg_model = log_reg_model.fit()
    st.subheader("Logistic Regression Model Summary (After VIF Removal)")
    st.write(log_reg_model.summary())

    # Validation
    y_val_pred_prob = log_reg_model.predict(x_val)
    fpr, tpr, thresholds = roc_curve(y_val, y_val_pred_prob)
    auc_score = roc_auc_score(y_val, y_val_pred_prob)

    roc_df = pd.DataFrame({
        "Threshold": thresholds,
        "False Positive Rate": fpr,
        "True Positive Rate": tpr
    })
    st.dataframe(roc_df)

    fig_logit, ax_logit = plt.subplots()
    ax_logit.plot(fpr, tpr, label = f"AUC = {auc_score:.2%}")
    ax_logit.plot([0, 1], [0, 1], 'k--', label = "Random Guess")
    ax_logit.set_xlabel("False Positive Rate")
    ax_logit.set_ylabel("True Positive Rate")
    ax_logit.set_title("ROC Curve")
    ax_logit.legend(loc = "lower right")
    st.pyplot(fig_logit)

    # Test Performance
    threshold_result = []    
    y_test_prob = log_reg_model.predict(x_test)
    for threshold in np.arange(0, 1.01, 0.01):
        preds = (y_test_prob >= threshold).astype(int)
        churn_rate = preds.mean()
        threshold_result.append({"Threshold": round(threshold, 2), "Churn Rate": churn_rate})
    st.subheader("Churn Rate of Logistic Regression Model")
    threshold_df = pd.DataFrame(threshold_result)
    st.dataframe(threshold_df.style.format({"Churn Rate": "{:.2%}"}))
    st.markdown("Choose threshold = 0.34")

    y_test_pred = (y_test_prob >= 0.34).astype(int)

    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_test_pred))
    st.write("Accuracy:", accuracy_score(y_test, y_test_pred))
    st.write("Test AUC:", roc_auc_score(y_test, y_test_prob))
    st.text(classification_report(y_test, y_test_pred))

    churn_rate_log = y_test_pred.mean()
    st.write(f"Logistic Regression Churn Rate: Churn Rate: {churn_rate_log:.2%}")

    # Feature Importances for Logistic Regression
    importances = log_reg_model.params.abs().sort_values(ascending=False)
    st.subheader("Feature Importance - Logistic Regression")
    st.bar_chart(importances.head(15))



    # Random Forest Model
    st.header("Random Forest Model")
    rf_model = RandomForestClassifier(
        n_estimators = 1000,
        max_depth = 10,
        class_weight = 'balanced'
    )
    rf_model.fit(x_train, y_train)

    # Validation Performance
    y_val_rf_prob = rf_model.predict_proba(x_val)[:, 1]
    rf_val_auc = roc_auc_score(y_val, y_val_rf_prob)
    st.write(f"AUC on validation set: {rf_val_auc:.2%}")

    # ROC Curve for Random Forest
    fpr_rf, tpr_rf, _ = roc_curve(y_val,y_val_rf_prob)
    fig_rf, ax_rf = plt.subplots()
    ax_rf.plot(fpr_rf, tpr_rf, label = f"AUC = {rf_val_auc:.2%}")
    ax_rf.plot([0, 1], [0, 1], 'k--', label = "Random Guess")
    ax_rf.set_xlabel("False Positive Rate")
    ax_rf.set_ylabel("True Positive Rate")
    ax_rf.set_title("ROC Curve - Random Forest")
    ax_rf.legend(loc = "lower right")
    st.pyplot(fig_rf)

    # Test Performance
    y_test_rf_prob = rf_model.predict_proba(x_test)[:, 1]
    threshold_result = []
    for threshold in np.arange(0, 1.01, 0.01):
        preds = (y_test_rf_prob >= threshold).astype(int)
        churn_rate = preds.mean()
        threshold_result.append({"Threshold": round(threshold, 2), "Churn Rate": churn_rate})
    st.subheader("Churn Rate of Random Forest Model")
    threshold_df = pd.DataFrame(threshold_result)
    st.dataframe(threshold_df.style.format({"Churn Rate": "{:.2%}"}))
    st.markdown("Choose threshold = 0.39")

    y_test_rf_pred = (y_test_rf_prob >= 0.39).astype(int)

    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_test_rf_pred))
    st.write("Accuracy:", accuracy_score(y_test, y_test_rf_pred))
    st.write("Test AUC:", roc_auc_score(y_test, y_test_rf_prob))
    st.text(classification_report(y_test, y_test_rf_pred))

    churn_rate_rf = y_test_rf_pred.mean()
    st.write(f"Random Forest Model Churn Rate: Churn Rate: {churn_rate_rf:.2%}")

    importances = pd.Series(rf_model.feature_importances_, index = x_train.columns)
    importances = importances.sort_values(ascending = False).head(15)
    st.bar_chart(importances)



    # XGBoost Model
    st.header("XGBoost Model")
    xgb_model = xgb.XGBClassifier(
        n_estimators = 1000,
        max_depth = 5,
        learning_rate = 0.01,
        use_label_encoder = False,
        eval_metric = 'logloss'
    )
    

    xgb_model.fit(
        x_train, y_train, eval_set = [(x_val, y_val)], verbose = False
    )

    # Validation Performance
    y_val_xgb_prob = xgb_model.predict_proba(x_val)[:, 1]
    val_auc_xgb = roc_auc_score(y_val, y_val_xgb_prob)
    st.write(f"AUC on validation set: {val_auc_xgb:.2%}")

    fpr_xgb, tpr_xgb, _ = roc_curve(y_val, y_val_xgb_prob)
    fig_xgb, ax_xgb = plt.subplots()
    ax_xgb.plot(fpr_xgb, tpr_xgb, label = f"AUC = {val_auc_xgb:.2%}")
    ax_xgb.plot([0, 1], [0, 1], 'k--', label = "Random Guess")
    ax_xgb.set_xlabel("False Positive Rate")
    ax_xgb.set_ylabel("True Positive Rate")
    ax_xgb.set_title("ROC Curve - XGBoost")
    ax_xgb.legend(loc = "lower right")
    st.pyplot(fig_xgb)

    # Test Performance
    threshold_result = []
    y_test_xgb_prob = xgb_model.predict_proba(x_test)[:, 1]
    for threshold in np.arange(0, 1.01, 0.01):
        preds = (y_test_xgb_prob >= threshold).astype(int)
        churn_rate = preds.mean()
        threshold_result.append({"Threshold": round(threshold, 2), "Churn Rate": churn_rate})
    
    st.subheader("Churn Rate of XGBoost Model")
    threshold_df = pd.DataFrame(threshold_result)
    st.dataframe(threshold_df.style.format({"Churn Rate": "{:.2%}"}))
    st.markdown("Choose threshold = 0.18")

    y_test_xgb_pred = (y_test_xgb_prob >= 0.18).astype(int)

    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_test_xgb_pred))
    st.write("Accuracy:", accuracy_score(y_test, y_test_xgb_pred))
    st.write("Test AUC:", roc_auc_score(y_test, y_test_xgb_prob))
    st.text(classification_report(y_test, y_test_xgb_pred))

    churn_rate_xgb = y_test_xgb_pred.mean()
    st.write(f"XGBoost Model Churn Rate: Churn Rate: {churn_rate_xgb:.2%}")

    xgb_importances = pd.Series(xgb_model.feature_importances_, index = x_train.columns)
    xgb_importances = xgb_importances.sort_values(ascending = False).head(15)
    st.bar_chart(xgb_importances)



    # Decision Tree Model
    st.header("Decision Tree Model")
    dt_model = DecisionTreeClassifier(max_depth = 5, class_weight = 'balanced')
    dt_model.fit(x_train, y_train)

    # Validation Performance
    y_val_dt_prob = dt_model.predict_proba(x_val)[:, 1]
    dt_val_auc = roc_auc_score(y_val, y_val_dt_prob)
    st.write(f"AUC on validation set: {dt_val_auc:.2%}")

    # ROC Curve for Decision Tree
    fpr_dt, tpr_dt, _ = roc_curve(y_val, y_val_dt_prob)
    fig_dt, ax_dt = plt.subplots()
    ax_dt.plot(fpr_dt, tpr_dt, label = f"AUC ={dt_val_auc:.2%}")
    ax_dt.plot([0, 1], [0, 1], 'k--', label = "Random Guess")
    ax_dt.set_xlabel("False Positive Rate")
    ax_dt.set_ylabel("True Positive Rate")
    ax_dt.set_title("ROC Curve - Decision Tree")
    ax_dt.legend(loc = "lower right")
    st.pyplot(fig_dt)

    # Test Performance
    y_test_dt_prob = dt_model.predict_proba(x_test)[:, 1]
    threshold_result = []
    for threshold in np.arange(0, 1.01, 0.01):
        preds = (y_test_dt_prob >= threshold).astype(int)
        churn_rate = preds.mean()
        threshold_result.append({"Threshold": round(threshold, 2), "Churn Rate": churn_rate})
    st.subheader("Churn Rate of Decision Tree Model")
    threshold_df = pd.DataFrame(threshold_result)
    st.dataframe(threshold_df.style.format({"Churn Rate": "{:.2%}"}))
    st.markdown("Choose threshold = 0.52")
    y_test_dt_pred = (y_test_dt_prob >= 0.52).astype(int)

    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_test_dt_pred))
    st.write("Accuracy:", accuracy_score(y_test, y_test_dt_pred))
    st.write("Test AUC:", roc_auc_score(y_test, y_test_dt_prob))
    st.text(classification_report(y_test, y_test_dt_pred))

    churn_rate_dt = y_test_dt_pred.mean()
    st.write(f"Decision Tree Model Churn Rate: Churn Rate: {churn_rate_dt:.2%}")

    # Feature Importances for Decision Tree
    dt_importances = pd.Series(dt_model.feature_importances_, index = x_train.columns)
    dt_importances = dt_importances.sort_values(ascending = False).head(15)
    st.bar_chart(dt_importances)

    # Plot Tree
    st.subheader("Decision Tree Visualization")
    fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
    plot_tree(dt_model, feature_names = x_train.columns, class_names = ["No Churn", "Churn"], filled = True,rounded = True, fontsize = 8, ax = ax_tree)
    st.pyplot(fig_tree)



    # GAM/EBM Model
    st.header("Generalized Additive Model (GAM)")
    ebm_model = ExplainableBoostingClassifier(interactions = 4)
    ebm_model.fit(x_train, y_train)

    #validation performance
    y_val_ebm_prob = ebm_model.predict_proba(x_val)[:, 1]
    ebm_val_auc = roc_auc_score(y_val, y_val_ebm_prob)
    st.write(f"AUC on validation set: {ebm_val_auc:.2%}")

    # ROC Curve for EBM
    fpr_ebm, tpr_ebm, _ = roc_curve(y_val, y_val_ebm_prob)
    fig_ebm, ax_ebm = plt.subplots()
    ax_ebm.plot(fpr_ebm, tpr_ebm, label = f"AUC = {ebm_val_auc:.2%}")
    ax_ebm.plot([0, 1], [0, 1], 'k--', label = "Random Guess")
    ax_ebm.set_xlabel("False Positive Rate")
    ax_ebm.set_ylabel("True Positive Rate")
    ax_ebm.set_title("ROC Curve - EBM")
    ax_ebm.legend(loc = "lower right")
    st.pyplot(fig_ebm)

    # Test Performance
    y_test_ebm_prob = ebm_model.predict_proba(x_test)[:, 1]
    threshold_results = []
    for threshold in np.arange(0, 1.01, 0.01):
        preds = (y_test_ebm_prob >= threshold).astype(int)
        churn_rate = preds.mean()
        threshold_results.append({"Threshold": round(threshold, 2), "Churn Rate": churn_rate})
    st.subheader("Churn Rate of EBM Model")
    threshold_df_ebm = pd.DataFrame(threshold_results)
    st.dataframe(threshold_df_ebm.style.format({"Churn Rate": "{:.2%}"}))
    st.markdown("Choose threshold = 0.17")
    y_test_ebm_pred = (y_test_ebm_prob >= 0.17).astype(int)

    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_test_ebm_pred))
    st.write("Accuracy:", accuracy_score(y_test, y_test_ebm_pred))
    st.write("Test AUC:", roc_auc_score(y_test, y_test_ebm_prob))
    st.text(classification_report(y_test, y_test_ebm_pred)) 

    churn_rate_gam = y_test_ebm_pred.mean()
    st.write(f"GAM Model Churn Rate: Churn Rate: {churn_rate_gam:.2%}")

    #Retention Simulation
    st.header("Retention Simulation")
    model_option = st.selectbox("Select a model for retention simulation",
                                [
                                    "Logistic Regression Model",
                                    "XGBoost Model",
                                    "Random Forest Model",
                                    "Decision Tree Model",
                                    "GAM Model"
                                ])
    sim_df = x_test.copy()
    sim_df_original = sim_df.copy()

    st.markdown("Adjust Key Features for Retnetion Simulation")
    if 'customer_service_calls_12m' in sim_df.columns:
        new_calls = st.slider(
            "Simulate Change in Customer Service Calls (12 m)",
            min_value = -10,
            max_value = 10,
            value = 0
        )
    else:
        new_calls = 0

    if 'login_frequency_30d' in sim_df.columns:
        new_login_freq = st.slider(
            "Simulate Change in Login Frequency (30 d)",
            min_value = -10,
            max_value = 10,
            value = 0
        )
    else:
        new_login_freq = 0

    if 'complaints_filed' in sim_df.columns:
        new_complaints = st.slider(
            "Simulate Change in Complaints Filed",
            min_value = -5,
            max_value = 5,
            value = 0
        )
    else:
        new_complaints = 0

    if 'has_chronic_overdue' in sim_df.columns:
        new_chronic_overdue = st.slider(
            "Simulate Change in Chronic Overdue",
            min_value = -1,
            max_value = 1,
            value = 0
        )
    else:
        new_chronic_overdue = 0

    # Apply the changes to sim_df after sliders and before prediction
    if 'customer_service_calls_12m' in sim_df.columns:
        sim_df['customer_service_calls_12m'] = (sim_df_original['customer_service_calls_12m'] + new_calls).clip(lower=0)
    if 'login_frequency_30d' in sim_df.columns:
        sim_df['login_frequency_30d'] = (sim_df_original['login_frequency_30d'] + new_login_freq).clip(lower=0)
    if 'complaints_filed' in sim_df.columns:
        sim_df['complaints_filed'] = (sim_df_original['complaints_filed'] + new_complaints).clip(lower=0)
    if 'has_chronic_overdue' in sim_df.columns:
        sim_df['has_chronic_overdue'] = (sim_df_original['has_chronic_overdue'] + new_chronic_overdue).clip(lower=0)

    if model_option == "Logistic Regression Model":
        y_sim_pred_prob = log_reg_model.predict(sim_df)
        threshold = 0.34
        orig_rate = churn_rate_log
    elif model_option == "XGBoost Model":
        y_sim_pred_prob = xgb_model.predict_proba(sim_df)[:, 1]
        threshold = 0.18
        orig_rate = churn_rate_xgb
    elif model_option == "Random Forest Model":
        y_sim_pred_prob = rf_model.predict_proba(sim_df)[:, 1]
        threshold = 0.18
        orig_rate = churn_rate_rf
    elif model_option == "Decision Tree Model":
        y_sim_pred_prob = dt_model.predict_proba(sim_df)[:, 1]
        threshold = 0.52
        orig_rate = churn_rate_dt
    elif model_option == "GAM Model":
        y_sim_pred_prob = ebm_model.predict_proba(sim_df)[:, 1]
        threshold = 0.17
        orig_rate = churn_rate_gam

    st.metric("Original Churn Rate", "15%")
    st.metric("Original Predicted Churn Rate", f"{orig_rate:.2%}")
    st.metric("Simulated Predicted Churn Rate", f"{(y_sim_pred_prob >= threshold).mean():.2%}")
 




elif section == "Extra":
    churned_customers = pd.read_excel('merged_data.xlsx', sheet_name='Churned Customers')
    merged_df['polciy_date'] = pd.to_datetime(merged_df['policy_start_date'], errors='coerce')
    merged_df['policy_month'] = merged_df['policy_start_date'].dt.to_period('M')
    monthly_starts_df = merged_df.groupby('policy_month').agg(
        policy_start_count = ('customer_id', 'count'),
        monthly_churn_count = ('churned', 'sum')
    ).reset_index()

    monthly_starts_df['policy_month'] = monthly_starts_df['policy_month'].dt.to_timestamp()

    churned_customers['churn_month'] = churned_customers['churn_date'].dt.to_period('M')
    monthly_churn_df = churned_customers.groupby('churn_month').agg(churn_count = ('customer_id', 'count')).reset_index()
    monthly_churn_df['churn_month'] = monthly_churn_df['churn_month'].dt.to_timestamp()
    st.write("Monthly Churn Customer Count")
    st.write(monthly_churn_df)
    st.line_chart(monthly_churn_df.set_index("churn_month"))

    churn_reason_counts = churned_customers['churn_reason'].value_counts(normalize = True)
    st.write("Churn Reason Distribution")
    st.bar_chart(churn_reason_counts)

    st.write(monthly_starts_df)

    st.subheader("Churn Reason VS Key Features")

    selected_features = [
        'age', 'gender', 'marital_status', 'dependents', 'income_bracket', 'employment_status',
        'education_level', 'login_frequency_30d', 'mobile_app_user', 'email_opens_6m',
        'customer_service_calls_12m', 'complaints_filed', 'policy_type', 'premium_amount',
        'premium_frequency', 'payment_method', 'riders', 'days_between_policy_last_login',
        'days_between_policy_churn', 'policy_tenure'
    ]

    selected_col = st.selectbox("Select a feature to visualize vs churn reason", selected_features)

    try:
        st.write(f"**Churn Reason vs {selected_col}**")
        if churned_customers[selected_col].dtype in ['object', 'bool']:
            sorted_categories = sorted(churned_customers[selected_col].dropna().unique())
            fig = px.histogram(
                churned_customers,
                x=selected_col,
                color='churn_reason',
                barmode='group',
                category_orders={selected_col: sorted_categories}
            )
            st.plotly_chart(fig)

            # Normalize percentage within each selected_col
            ratio_df = churned_customers.groupby([selected_col, 'churn_reason']).size().reset_index(name='count')
            total_per_group = churned_customers.groupby([selected_col]).size().reset_index(name='total')
            merged_ratio = pd.merge(ratio_df, total_per_group, on=selected_col)
            merged_ratio['percentage'] = merged_ratio['count'] / merged_ratio['total']
            sorted_categories = merged_ratio[selected_col].drop_duplicates().sort_values()
            fig_pct = px.bar(
                merged_ratio,
                x=selected_col,
                y='percentage',
                color='churn_reason',
                category_orders={selected_col: sorted_categories.tolist()},
                barmode='stack',
                title=f"Normalized % Churn Reason by {selected_col}"
            )
            st.plotly_chart(fig_pct)
        else:
            fig = px.box(churned_customers, x='churn_reason', y=selected_col, points='all')
            st.plotly_chart(fig)
    except Exception as e:
        st.warning(f"Error plotting {selected_col}: {e}")


    # Churn Days Analysis
    st.header("Churn Days Analysis")
    
    # Data Preparation
    num_vars = [
        'age', 'dependents', 'login_frequency_30d', 'email_opens_6m', 'customer_service_calls_12m',
        'complaints_filed', 'policy_amount', 'premium_amount', 'riders', 'days_between_policy_last_login',
        'engagement_score'
    ]

    cat_vars = [
        'gender', 'marital_status', 'income_bracket', 'employment_status', 'education_level',
        'mobile_app_user', 'policy_type', 'premium_frequency', 'payment_method', 'churn_behavior'
    ]

    for col in num_vars:
        churned_customers[col] = pd.to_numeric(churned_customers[col], errors = 'coerce')
    for col in cat_vars:
        churned_customers[col] = churned_customers[col].astype(str)

    reg_df = churned_customers.copy()
    reg_df = pd.get_dummies(reg_df, columns=cat_vars, drop_first=True)
    for col in num_vars:
        reg_df[col] = pd.to_numeric(reg_df[col], errors = 'coerce')
    y = pd.to_numeric(reg_df['days_between_policy_churn'], errors='coerce')

    # Linear Regression Model
    reg_df = reg_df.sort_values('policy_start_date')

    x = reg_df.drop(columns = ['customer_id', 'churned', 'churn_date', 'policy_start_date', 'churn_reason', 'days_between_policy_churn', 'last_login_date', 'churn_month'])
    # Drop any datetime or period columns before regression
    y = pd.to_numeric(reg_df['days_between_policy_churn'], errors='coerce')

    n = len(x)
    train_end = int(n * 0.8)
    x_train = x.iloc[:train_end]
    y_train = y.iloc[:train_end]
    x_test = x.iloc[train_end:]
    y_test = y.iloc[train_end:]

    # Fill missing values with zero for linear regression
    x_train = x_train.fillna(0)
    y_train = y_train.fillna(0)
    x_test = x_test.fillna(0)
    y_test = y_test.fillna(0)

    # Linear Regression Model
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)
    y_pred = lin_reg.predict(x_test)

    st.subheader("Linear Regression Results")
    st.write("R-squared on test set:", r2_score(y_test, y_pred))
    st.write("Mean Squared Error on test set:", mean_squared_error(y_test, y_pred))

    # Plot of Actual vs Predicted
    plt.figure(figsize = (12, 6))
    plt.hist(y_test, bins=20, alpha=0.6, label='Actual')
    plt.hist(y_pred, bins=20, alpha=0.6, label='Predicted')
    plt.xlabel('Days Between Policy Start and Churn')
    plt.ylabel('Frequency')
    plt.title('Actual vs Predicted Days Between Policy Start and Churn')
    plt.legend()
    st.pyplot(plt)

    # Lasso Regression Model
    st.subheader("Lasso Regression Model")

    scaler = StandardScaler()
    x_train_scaler = scaler.fit_transform(x_train)
    x_test_scaler = scaler.transform(x_test)

    lasso = LassoCV(cv = 5, n_alphas = 100, max_iter = 10000)
    lasso.fit(x_train_scaler, y_train)
    y_lasso_pred = lasso.predict(x_test_scaler)

    st.write("Best alpha:", float(lasso.alpha_))
    st.write("R-squared on test set:", r2_score(y_test, y_lasso_pred))
    st.write("Mean Squared Error on test set:", mean_squared_error(y_test, y_lasso_pred))

    plt.figure(figsize = (12, 6))
    plt.hist(y_test, bins=20, alpha=0.6, label='Actual')
    plt.hist(y_lasso_pred, bins=20, alpha=0.6, label='Predicted')
    plt.xlabel('Days Between Policy Start and Churn')
    plt.ylabel('Frequency')
    plt.title('Actual vs Predicted Days Between Policy Start and Churn')
    plt.legend()
    st.pyplot(plt)
    

    # Ridge Regression Model
    st.subheader("Ridge Regression Model")
    
    ridge = RidgeCV(alphas = np.logspace(-3, 3, 50))
    ridge.fit(x_train_scaler, y_train)
    y_ridge_pred = ridge.predict(x_test_scaler)

    st.write("Best alpha:", float(ridge.alpha_))
    st.write("R-squared on test set:", r2_score(y_test, y_ridge_pred))
    st.write("Mean Squared Error on test set:", mean_squared_error(y_test, y_ridge_pred))

    plt.figure(figsize = (12, 6))
    plt.hist(y_test, bins=20, alpha=0.6, label='Actual')
    plt.hist(y_ridge_pred, bins=20, alpha=0.6, label='Predicted')
    plt.xlabel('Days Between Policy Start and Churn')
    plt.ylabel('Frequency')
    plt.title('Actual vs Predicted Days Between Policy Start and Churn')
    plt.legend()
    st.pyplot(plt)

    # XGBoost Regression Model
    st.subheader("XGBoost Regression Model")

    xgb_reg = xgb.XGBRegressor(
        n_estimators = 1000,
        max_depth = 5,
        learning_rate = 0.01,
        subsample = 0.8,
        colsample_bytree = 0.8,
        objective = 'reg:squarederror',
        n_jobs = -1
    )

    xgb_reg.fit(x_train, y_train, eval_set = [(x_test, y_test)], verbose = False)

    y_xgb_pred = xgb_reg.predict(x_test)

    st.write("R-squared on test set:", r2_score(y_test, y_xgb_pred))
    st.write("Mean Squared Error on test set:", mean_squared_error(y_test, y_xgb_pred))

    plt.figure(figsize = (12, 6))
    plt.hist(y_test, bins=20, alpha=0.6, label='Actual')
    plt.hist(y_xgb_pred, bins=20, alpha=0.6, label='Predicted')
    plt.xlabel('Days Between Policy Start and Churn')
    plt.ylabel('Frequency')
    plt.title('Actual vs Predicted Days Between Policy Start and Churn')
    plt.legend()
    st.pyplot(plt)

    # Random Forest Regression Model
    st.subheader("Random Forest Regression Model")

    x_train = x.iloc[:train_end].fillna(0)
    x_test = x.iloc[train_end:].fillna(0)
    rf_reg = RandomForestRegressor(
        n_estimators = 1000,
        max_depth = 10,
        min_samples_split= 2,
        min_samples_leaf= 1,
        max_features = 'sqrt',
        n_jobs = -1
    )

    rf_reg.fit(x_train, y_train)
    y_rf_pred = rf_reg.predict(x_test)

    st.write("R-squared on test set:", r2_score(y_test, y_rf_pred))
    st.write("Mean Squared Error on test set:", mean_squared_error(y_test, y_rf_pred))

    plt.figure(figsize = (12, 6))
    plt.hist(y_test, bins=20, alpha=0.6, label='Actual')
    plt.hist(y_rf_pred, bins=20, alpha=0.6, label='Predicted')
    plt.xlabel('Days Between Policy Start and Churn')
    plt.ylabel('Frequency')
    plt.title('Actual vs Predicted Days Between Policy Start and Churn')
    plt.legend()
    st.pyplot(plt)

    # Decision Tree Regression Model
    st.subheader("Decision Tree Regression Model")
    dt_reg = DecisionTreeRegressor(max_depth = 5, min_samples_split = 2, min_samples_leaf = 1)
    dt_reg.fit(x_train, y_train)
    y_dt_pred = dt_reg.predict(x_test)

    st.write("R-squared on test set:", r2_score(y_test, y_dt_pred))
    st.write("Mean Squared Error on test set:", mean_squared_error(y_test, y_dt_pred))

    plt.figure(figsize = (12, 6))
    plt.hist(y_test, bins=20, alpha=0.6, label='Actual')
    plt.hist(y_dt_pred, bins=20, alpha=0.6, label=' Predicted')
    plt.xlabel('Days Between Policy Start and Churn')
    plt.ylabel('Frequency')
    plt.title('Actual vs Predicted Days Between Policy Start and Churn')
    plt.legend()
    st.pyplot(plt)

    # GAM/EBM Regression Model
    st.subheader("GAM/EBM Regression Model")

    ebm_reg = ExplainableBoostingRegressor(interactions = 5)
    ebm_reg.fit(x_train, y_train)

    y_ebm_pred = ebm_reg.predict(x_test)

    st.write("R-squared on test set:", r2_score(y_test, y_ebm_pred))
    st.write("Mean Squared Error on test set:", mean_squared_error(y_test, y_ebm_pred))

    plt.figure(figsize = (12, 6))
    plt.hist(y_test, bins=20, alpha=0.6, label='Actual')
    plt.hist(y_ebm_pred, bins=20, alpha=0.6, label='Predicted')
    plt.xlabel('Days Between Policy Start and Churn')
    plt.ylabel('Frequency')
    plt.title('Actual vs Predicted Days Between Policy Start and Churn')
    plt.legend()
    st.pyplot(plt)