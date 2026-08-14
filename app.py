import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

# Metrics
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

# Set page config
st.set_page_config(
    page_title="Student Academic Failure Risk Classifier",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App custom styling
st.markdown("""
<style>
    .main-title {
        font-family: 'Inter', sans-serif;
        color: #1E3A8A;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #4B5563;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .winner-banner {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model and data paths dynamically
@st.cache_resource
def load_paths():
    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / "model"
    results_dir = base_dir / "results"
    sample_data_path = base_dir / "test_data.csv"
    
    # Check model filenames
    model_files = {
        "Logistic Regression": models_dir / "logistic_regression.pkl",
        "Decision Tree": models_dir / "decision_tree.pkl",
        "KNN": models_dir / "knn.pkl",
        "Naive Bayes": models_dir / "naive_bayes.pkl",
        "Random Forest": models_dir / "random_forest.pkl"
    }
    
    comparison_path = results_dir / "model_comparison.csv"
    observations_path = results_dir / "model_observations.csv"
    
    return model_files, sample_data_path, comparison_path, observations_path

# Load paths
model_files, sample_data_path, comparison_path, observations_path = load_paths()

@st.cache_resource
def load_trained_pipeline(model_path):
    if not model_path.exists():
        st.error(f"Model file not found at: {model_path.name}. Please ensure the models are trained and saved.")
        return None
    return joblib.load(model_path)

def get_model_features_weights(pipeline, selected_model_name):
    try:
        preprocessor = pipeline.named_steps['preprocessor']
        model = pipeline.named_steps['model']
        feature_names = preprocessor.get_feature_names_out()
        feature_names = [f.replace('num__', '').replace('cat__', '') for f in feature_names]
        
        weights = None
        is_coef = False
        
        if selected_model_name == "Logistic Regression" and hasattr(model, 'coef_'):
            weights = model.coef_[0]
            is_coef = True
        elif selected_model_name in ["Decision Tree", "Random Forest"] and hasattr(model, 'feature_importances_'):
            weights = model.feature_importances_
            is_coef = False
            
        if weights is not None:
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Value': weights
            })
            if is_coef:
                importance_df['Abs_Value'] = importance_df['Value'].abs()
                importance_df = importance_df.sort_values(by='Abs_Value', ascending=False).drop(columns=['Abs_Value'])
            else:
                importance_df = importance_df.sort_values(by='Value', ascending=False)
            return importance_df, is_coef
    except Exception as e:
        pass
    return None, False

# Header
st.markdown('<div class="main-title">Student Academic Failure Risk — ML Classification Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Compare multiple Machine Learning models for predicting Academic Failure Risk using student social media, AI usage, health, lifestyle, and academic indicators.</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/graduation-cap.png", width=90)
st.sidebar.header("Navigation & Configuration")

app_mode = st.sidebar.radio(
    "Go to:", 
    ["Predict (Batch Upload)", "Predict (Single Student)", "Global Model Comparison", "Exploratory Data Analysis (EDA)", "About the Dataset"]
)

# Initialize data variable
data_df = None
is_sample_data = False
selected_model_name = None

# Sidebar File Uploader (Only for Batch Upload)
if app_mode == "Predict (Batch Upload)":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Upload Test Data")
    uploaded_file = st.sidebar.file_uploader("Upload Test Data (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            data_df = pd.read_csv(uploaded_file)
            st.sidebar.success("Uploaded file successfully!")
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")
    else:
        # Check if sample test_data.csv exists
        if sample_data_path.exists():
            data_df = pd.read_csv(sample_data_path)
            is_sample_data = True
            st.sidebar.info("Using included sample test dataset.")
        else:
            st.sidebar.warning("No test data found. Please upload a CSV file.")

# Sidebar Model Selection (For Batch and Single Student modes)
if app_mode in ["Predict (Batch Upload)", "Predict (Single Student)"]:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Selection")
    selected_model_name = st.sidebar.selectbox(
        "Select Classifier Model:",
        ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest"]
    )

# 1. Main Tab: Predict (Batch Upload)
if app_mode == "Predict (Batch Upload)":
    if data_df is None:
        st.warning("Please upload a test dataset or place test_data.csv in the application root to begin.")
    else:
        # Load Selected Model
        model_path = model_files[selected_model_name]
        pipeline = load_trained_pipeline(model_path)
        
        if pipeline is not None:
            # Check validation
            target_col = "Academic_Failure_Risk"
            id_col = "Student_ID"
            
            # Predictors expected by preprocessor
            # Inspect the model's preprocessing feature names if possible
            # Or fall back to standard list
            required_cols = [
                'Age', 'Gender', 'Education_Level', 'Daily_Social_Media_Hours', 
                'Daily_AI_Tool_Usage_Hours', 'Sleep_Hours', 'Physical_Activity_Hours', 
                'Mental_Health_Score', 'Physical_Health_Score', 'Social_Isolation_Score', 
                'Burnout_Level', 'Academic_Performance_Score'
            ]
            
            # Missing columns verification
            missing_cols = [col for col in required_cols if col not in data_df.columns]
            
            if missing_cols:
                st.error(f"The uploaded CSV is missing critical feature columns: {missing_cols}")
            else:
                # Prepare features for prediction (drop target and ID if present)
                X_eval = data_df[required_cols].copy()
                
                # Check target presence
                has_target = target_col in data_df.columns
                
                # Make Predictions
                with st.spinner("Generating predictions..."):
                    preds = pipeline.predict(X_eval)
                    # Probabilities
                    probs = pipeline.predict_proba(X_eval)[:, 1]
                
                # Layout columns
                col_left, col_right = st.columns([1, 1])
                
                with col_left:
                    st.subheader("Dataset Overview")
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    col_stat1.metric("Row Count", data_df.shape[0])
                    col_stat2.metric("Column Count", data_df.shape[1])
                    col_stat3.metric("Data Source", "Uploaded CSV" if not is_sample_data else "Sample holdout")
                    
                    st.write("Preview of evaluation features:")
                    st.dataframe(data_df.head(5), height=180)
                    
                    if has_target:
                        st.write("Target Variable distribution (`Academic_Failure_Risk`):")
                        target_dist = data_df[target_col].value_counts()
                        target_dist_df = pd.DataFrame({
                            "Count": target_dist.values,
                            "Proportion": target_dist.values / len(data_df)
                        }, index=target_dist.index)
                        st.dataframe(target_dist_df.style.format({"Proportion": "{:.2%}"}))
                
                with col_right:
                    st.subheader(f"Current Model: {selected_model_name}")
                    st.info(f"Loaded classifier pipeline successfully. Ready for inference on {len(X_eval)} records.")
                    
                    # Display prediction distribution
                    pred_counts = pd.Series(preds).value_counts()
                    pred_0 = pred_counts.get(0, 0)
                    pred_1 = pred_counts.get(1, 0)
                    
                    fig, ax = plt.subplots(figsize=(6, 2.5))
                    sns.barplot(x=["Low/No Risk (0)", "Failure Risk (1)"], y=[pred_0, pred_1], palette=["#3B82F6", "#EF4444"], ax=ax)
                    ax.set_title("Distribution of Predicted Risk Labels")
                    ax.set_ylabel("Count")
                    sns.despine()
                    st.pyplot(fig)
                
                # Evaluation Metrics section (if target exists)
                st.markdown("---")
                if has_target:
                    st.subheader(f"Model Evaluation Metrics ({selected_model_name})")
                    y_true = data_df[target_col].astype(int)
                    
                    # Calculate Metrics
                    acc = accuracy_score(y_true, preds)
                    auc = roc_auc_score(y_true, probs)
                    prec = precision_score(y_true, preds, zero_division=0)
                    rec = recall_score(y_true, preds)
                    f1 = f1_score(y_true, preds)
                    mcc = matthews_corrcoef(y_true, preds)
                    
                    # Display metrics in cards
                    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
                    
                    m_col1.markdown(f'<div class="metric-card"><div class="metric-value">{acc:.2%}</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)
                    m_col2.markdown(f'<div class="metric-card"><div class="metric-value">{auc:.4f}</div><div class="metric-label">AUC-ROC</div></div>', unsafe_allow_html=True)
                    m_col3.markdown(f'<div class="metric-card"><div class="metric-value">{prec:.2%}</div><div class="metric-label">Precision</div></div>', unsafe_allow_html=True)
                    m_col4.markdown(f'<div class="metric-card"><div class="metric-value">{rec:.2%}</div><div class="metric-label">Recall</div></div>', unsafe_allow_html=True)
                    m_col5.markdown(f'<div class="metric-card"><div class="metric-value">{f1:.2%}</div><div class="metric-label">F1-Score</div></div>', unsafe_allow_html=True)
                    m_col6.markdown(f'<div class="metric-card"><div class="metric-value">{mcc:.4f}</div><div class="metric-label">MCC Score</div></div>', unsafe_allow_html=True)
                    
                    # Confusion Matrix & Classification Report Layout
                    c_col1, c_col2 = st.columns([1, 1])
                    
                    with c_col1:
                        st.subheader("Confusion Matrix")
                        cm = confusion_matrix(y_true, preds)
                        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                        sns.heatmap(
                            cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=["Low/No Risk (0)", "Failure Risk (1)"], 
                            yticklabels=["Low/No Risk (0)", "Failure Risk (1)"],
                            cbar=False, ax=ax_cm
                        )
                        ax_cm.set_xlabel("Predicted")
                        ax_cm.set_ylabel("Actual")
                        st.pyplot(fig_cm)
                        
                    with c_col2:
                        st.subheader("Classification Report")
                        rep_dict = classification_report(y_true, preds, output_dict=True)
                        rep_df = pd.DataFrame(rep_dict).transpose()
                        st.dataframe(rep_df.style.format(precision=4), height=200)
                else:
                    st.info("Evaluation target column `Academic_Failure_Risk` is not present in the dataset. Models can generate predictions, but comparative metrics cannot be calculated.")
                
                # Predictions Output Section
                st.markdown("---")
                st.subheader("Batch Prediction Results")
                
                # Construct result dataframe
                results_df = data_df.copy()
                results_df["Predicted_Risk_Label"] = preds
                results_df["Predicted_Risk_Meaning"] = np.where(preds == 1, "Academic Failure Risk", "Low/No Academic Failure Risk")
                results_df["Failure_Probability"] = probs
                
                # If ID exists, keep it visible
                cols_to_show = []
                if id_col in results_df.columns:
                    cols_to_show.append(id_col)
                cols_to_show.extend(["Predicted_Risk_Label", "Predicted_Risk_Meaning", "Failure_Probability"])
                # Append a few feature columns for context
                cols_to_show.extend(required_cols[:5])
                
                st.write("Sample predictions:")
                st.dataframe(results_df[cols_to_show].head(10), height=300)
                
                # Download predictions CSV
                pred_csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Complete Predictions (CSV)",
                    data=pred_csv,
                    file_name=f"predictions_{selected_model_name.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )
                
                # Model Interpretation Section (Feature Importances/Coefficients)
                importance_df, is_coef = get_model_features_weights(pipeline, selected_model_name)
                if importance_df is not None:
                    st.markdown("---")
                    st.subheader(f"Model Interpretation: {selected_model_name}")
                    
                    if is_coef:
                        st.write("This chart shows the **coefficients** (weights) assigned to each feature. Positive values (Red) increase predicted risk, negative values (Blue) reduce risk.")
                    else:
                        st.write("This chart shows the relative **feature importance** (influence) on model decision-making.")
                        
                    fig_imp, ax_imp = plt.subplots(figsize=(8, 4.5))
                    if is_coef:
                        colors = ['#EF4444' if val > 0 else '#3B82F6' for val in importance_df['Value']]
                        sns.barplot(data=importance_df, y='Feature', x='Value', palette=colors, ax=ax_imp)
                    else:
                        sns.barplot(data=importance_df, y='Feature', x='Value', color='#3B82F6', ax=ax_imp)
                        
                    ax_imp.set_title(f"Feature Influence for {selected_model_name}")
                    ax_imp.set_xlabel("Weight / Importance Score")
                    ax_imp.set_ylabel("Feature")
                    sns.despine()
                    st.pyplot(fig_imp)

# 2. Main Tab: Predict (Single Student)
elif app_mode == "Predict (Single Student)":
    st.subheader(f"Single Student Risk Diagnosis — Current Model: {selected_model_name}")
    
    # Load model
    model_path = model_files[selected_model_name]
    pipeline = load_trained_pipeline(model_path)
    
    if pipeline is not None:
        st.write("Fill out the demographics, digital behaviors, lifestyle, and health scores below to predict academic failure risk in real-time.")
        
        with st.form("single_student_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("##### 👤 Demographics")
                age = st.slider("Age (years)", 15, 25, 20)
                gender = st.selectbox("Gender", ["Male", "Female", "Non-binary"])
                edu_level = st.selectbox("Education Level", ["High School", "College", "University"])
                
            with col2:
                st.markdown("##### 📱 Digital Behavior")
                social_media = st.slider("Daily Social Media (hours)", 0.0, 15.0, 2.0, step=0.5)
                ai_usage = st.slider("Daily AI Tool Usage (hours)", 0.0, 15.0, 1.0, step=0.5)
                sleep_hours = st.slider("Nightly Sleep Duration (hours)", 3.0, 12.0, 7.0, step=0.5)
                physical_hours = st.slider("Daily Physical Activity (hours)", 0.0, 8.0, 1.0, step=0.5)
                
            with col3:
                st.markdown("##### 🩺 Health & Academics")
                mental_score = st.slider("Mental Health Score (0-100)", 0, 100, 75)
                physical_score = st.slider("Physical Health Score (0-100)", 0, 100, 75)
                isolation_score = st.slider("Social Isolation Score (0-100)", 0, 100, 30)
                burnout_level = st.selectbox("Burnout Level", ["Low", "Moderate", "High", "Severe"])
                academic_score = st.slider("Academic Performance Score (0-100)", 0, 100, 75)
                
            submit_button = st.form_submit_button("Run Diagnostic Prediction 🧠")
            
        if submit_button:
            # Construct a dataframe matching train columns
            single_student_data = pd.DataFrame([{
                'Age': age,
                'Gender': gender,
                'Education_Level': edu_level,
                'Daily_Social_Media_Hours': social_media,
                'Daily_AI_Tool_Usage_Hours': ai_usage,
                'Sleep_Hours': sleep_hours,
                'Physical_Activity_Hours': physical_hours,
                'Mental_Health_Score': mental_score,
                'Physical_Health_Score': physical_score,
                'Social_Isolation_Score': isolation_score,
                'Burnout_Level': burnout_level,
                'Academic_Performance_Score': academic_score
            }])
            
            # Predict
            pred_label = pipeline.predict(single_student_data)[0]
            pred_prob = pipeline.predict_proba(single_student_data)[0][1]
            
            # Display results
            st.markdown("### Prediction Diagnostic Report")
            
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                if pred_label == 1:
                    st.markdown("""
                    <div style="background-color: #FEE2E2; border-left: 5px solid #EF4444; padding: 20px; border-radius: 8px;">
                        <h4 style="color: #991B1B; margin:0 0 10px 0;">🚨 Predicted Academic Failure Risk: HIGH</h4>
                        <p style="color: #7F1D1D; margin:0;">The model predicts this student is at risk of academic failure.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background-color: #D1FAE5; border-left: 5px solid #10B981; padding: 20px; border-radius: 8px;">
                        <h4 style="color: #065F46; margin:0 0 10px 0;">✅ Predicted Academic Failure Risk: LOW</h4>
                        <p style="color: #064E3B; margin:0;">The student is classified as low/no academic risk.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display probability gauge/bar
                st.write("")
                st.metric("Academic Failure Probability", f"{pred_prob:.2%}")
                st.progress(float(pred_prob))
                
            with res_col2:
                st.markdown("#### 💡 Tailored Recommendations")
                recs = []
                if pred_label == 1:
                    if social_media > 4.0:
                        recs.append("- 📱 **Limit Social Media**: High daily screen time detected. Restricting social media to < 2 hours can improve focus.")
                    if sleep_hours < 6.5:
                        recs.append("- 💤 **Improve Sleep**: Nightly sleep duration is low. Target a consistent 7-8 hours to improve cognitive recovery.")
                    if mental_score < 50:
                        recs.append("- 🩺 **Counseling Services**: Student shows low mental wellness indicators. Recommend student counseling or peer mentorship.")
                    if burnout_level in ["High", "Severe"]:
                        recs.append("- ⚠️ **Burnout Intervention**: Severe workload exhaustion. Encourage meeting with academic advisors to space out assignments and establish self-care.")
                    if academic_score < 60:
                        recs.append("- 📚 **Academic Support**: Low academic performance average. Recommend tutoring or attending faculty office hours.")
                    
                    if not recs:
                        recs.append("- 💡 **General Advisory**: Standard peer counselling, tutoring, and academic advice recommended.")
                else:
                    recs.append("- ⭐ Keep doing what you're doing! A healthy balance of sleep, physical activity, and social connections keeps academic risk low.")
                    if social_media > 4.0:
                        recs.append("- 📱 *Note: Social media usage is on the higher side. Monitor to prevent interference with studies.*")
                        
                st.write("\n".join(recs))
                
            # Local feature weight visualization for this model
            importance_df, is_coef = get_model_features_weights(pipeline, selected_model_name)
            if importance_df is not None:
                st.markdown("---")
                st.subheader(f"How {selected_model_name} Evaluates Features")
                st.write("Understand which features are most influential in this model's logic overall:")
                fig_imp, ax_imp = plt.subplots(figsize=(7, 3.5))
                if is_coef:
                    colors = ['#EF4444' if val > 0 else '#3B82F6' for val in importance_df['Value']]
                    sns.barplot(data=importance_df, y='Feature', x='Value', palette=colors, ax=ax_imp)
                else:
                    sns.barplot(data=importance_df, y='Feature', x='Value', color='#3B82F6', ax=ax_imp)
                ax_imp.set_title(f"Feature Weights for {selected_model_name}")
                ax_imp.set_xlabel("Influence Score")
                sns.despine()
                st.pyplot(fig_imp)

# 3. Main Tab: Global Model Comparison
elif app_mode == "Global Model Comparison":
    st.subheader("Holdout Model Comparison & Cross-Validation Observations")
    
    if comparison_path.exists() and observations_path.exists():
        comp_df = pd.read_csv(comparison_path)
        obs_df = pd.read_csv(observations_path)
        
        # Display comparison table
        st.markdown("### Precomputed Classifier Performance Table")
        st.write("All metrics are calculated on the 20% stratified holdout test split (3,000 records). 5-Fold CV F1 represents the cross-validation score on the training split.")
        
        # Format the comparison table beautifully
        styled_comp_df = comp_df.copy()
        format_dict = {
            "Accuracy": "{:.2%}",
            "AUC": "{:.4f}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1": "{:.2%}",
            "MCC": "{:.4f}"
        }
        if "CV_F1_Mean" in styled_comp_df.columns:
            format_dict["CV_F1_Mean"] = "{:.2%}"
        if "CV_F1_Std" in styled_comp_df.columns:
            format_dict["CV_F1_Std"] = "{:.2%}"
            
        highlight_subset = ["F1", "MCC", "AUC"]
        if "CV_F1_Mean" in styled_comp_df.columns:
            highlight_subset.append("CV_F1_Mean")
            
        st.dataframe(
            styled_comp_df.style.format(format_dict).highlight_max(subset=highlight_subset, color="#D1FAE5"),
            height=250
        )
        
        # Dynamic Winner Card
        winner_row = obs_df[obs_df["ML Model Name"] == "Overall Winner"]
        if not winner_row.empty:
            winner_text = winner_row.iloc[0]["Observation about model performance"]
            st.markdown(f"""
            <div class="winner-banner">
                <h3>🏆 Best Overall Model</h3>
                <p>{winner_text}</p>
            </div>
            """, unsafe_allow_html=True)
            
        # Display observations table
        st.markdown("### Model Observations")
        st.dataframe(
            obs_df[obs_df["ML Model Name"] != "Overall Winner"],
            column_config={
                "ML Model Name": st.column_config.Column("Classifier Model", width="medium"),
                "Observation about model performance": st.column_config.Column("Observations & Analysis", width="large")
            },
            hide_index=True,
            height=300
        )
        
        # Visual performance comparison chart
        st.markdown("### Performance Comparison Chart")
        
        # Melt dataframe for plotting
        value_vars = ["Accuracy", "AUC", "Precision", "Recall", "F1"]
        if "CV_F1_Mean" in comp_df.columns:
            value_vars.append("CV_F1_Mean")
            
        melted_df = comp_df.melt(id_vars="ML Model Name", value_vars=value_vars)
        
        fig_chart, ax_chart = plt.subplots(figsize=(10, 5))
        sns.barplot(data=melted_df, x="variable", y="value", hue="ML Model Name", palette="Set2", ax=ax_chart)
        ax_chart.set_ylim(0.4, 1.02)
        ax_chart.set_title("Classifier Comparison across Key Evaluation Metrics")
        ax_chart.set_xlabel("Metric")
        ax_chart.set_ylabel("Score")
        ax_chart.legend(title="ML Model", bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig_chart)
        
    else:
        st.warning("Comparison results not found. Please ensure you have run `model/train_models.py` before launching the app.")

# 4. Main Tab: Exploratory Data Analysis (EDA)
elif app_mode == "Exploratory Data Analysis (EDA)":
    st.subheader("Exploratory Data Analysis (EDA) - Cleaned Student Dataset")
    
    base_dir = Path(__file__).resolve().parent
    clean_data_path = base_dir / "data" / "AI_SocialMedia_Student_Health_Dataset_clean.csv"
    
    if clean_data_path.exists():
        df_clean = pd.read_csv(clean_data_path)
        
        st.write("Below are insights drawn directly from the cleaned training and evaluation dataset (15,000 records).")
        
        eda_tab1, eda_tab2, eda_tab3 = st.tabs(["Correlation Heatmap", "Digital Habits & Performance", "Lifestyle & Burnout"])
        
        with eda_tab1:
            st.markdown("#### Numerical Feature Correlation Matrix")
            st.write("Correlation analysis helps us see if variables move together. Values close to +1 or -1 indicate strong linear relationships.")
            
            # Select numerical columns
            num_cols = df_clean.select_dtypes(include=['int64', 'float64']).drop(columns=["Student_ID", "Academic_Failure_Risk"], errors="ignore")
            # Include risk for correlation
            num_cols["Failure_Risk"] = df_clean["Academic_Failure_Risk"]
            corr_matrix = num_cols.corr()
            
            fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar=True, ax=ax_corr, annot_kws={"size": 8})
            ax_corr.set_title("Pearson Correlation Heatmap")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig_corr)
            
        with eda_tab2:
            st.markdown("#### Impact of Social Media vs. Academic Performance")
            st.write("Does high screen time negatively correlate with grades? Explore the scatter relationship grouped by risk label.")
            
            fig_scat, ax_scat = plt.subplots(figsize=(8, 4))
            sns.scatterplot(
                data=df_clean.sample(2000, random_state=42), 
                x="Daily_Social_Media_Hours", 
                y="Academic_Performance_Score", 
                hue="Academic_Failure_Risk", 
                palette=["#3B82F6", "#EF4444"], 
                alpha=0.6,
                ax=ax_scat
            )
            ax_scat.set_title("Academic Performance vs. Daily Social Media Screen Time (Sample of 2000 students)")
            ax_scat.set_xlabel("Daily Social Media Hours")
            ax_scat.set_ylabel("Academic Performance Score")
            st.pyplot(fig_scat)
            
            st.markdown("#### Distribution of Sleep Hours by Risk Status")
            fig_sleep, ax_sleep = plt.subplots(figsize=(8, 3.5))
            sns.boxplot(
                data=df_clean,
                x="Academic_Failure_Risk",
                y="Sleep_Hours",
                palette=["#3B82F6", "#EF4444"],
                ax=ax_sleep
            )
            ax_sleep.set_xticklabels(["Low/No Risk (0)", "Failure Risk (1)"])
            ax_sleep.set_title("Nightly Sleep Hours Distribution by Academic Risk")
            st.pyplot(fig_sleep)
            
        with eda_tab3:
            st.markdown("#### Burnout Level vs. Academic Failure Risk")
            st.write("Examine the proportion of students flagged at risk within each class of emotional/workload burnout.")
            
            burnout_risk = pd.crosstab(df_clean['Burnout_Level'], df_clean['Academic_Failure_Risk'], normalize='index') * 100
            
            fig_burn, ax_burn = plt.subplots(figsize=(8, 4))
            burnout_risk.plot(kind='bar', stacked=True, color=["#3B82F6", "#EF4444"], ax=ax_burn)
            ax_burn.set_title("Risk Share (%) across Student Burnout Levels")
            ax_burn.set_ylabel("Percentage (%)")
            ax_burn.set_xlabel("Burnout Level")
            ax_burn.legend(["Low/No Risk (0)", "Failure Risk (1)"], loc="lower left")
            plt.xticks(rotation=0)
            sns.despine()
            st.pyplot(fig_burn)
            
            st.markdown("#### Social Isolation Score vs. Mental Health")
            st.write("Higher social isolation score correlates with lower mental health wellness indicators.")
            
            fig_iso, ax_iso = plt.subplots(figsize=(8, 4))
            sns.regplot(
                data=df_clean.sample(1000, random_state=42),
                x="Social_Isolation_Score",
                y="Mental_Health_Score",
                scatter_kws={"alpha": 0.4, "color": "#1E3A8A"},
                line_kws={"color": "#EF4444"},
                ax=ax_iso
            )
            ax_iso.set_title("Relationship between Isolation and Mental Health (Sample of 1000 students)")
            ax_iso.set_xlabel("Social Isolation Score")
            ax_iso.set_ylabel("Mental Health Score")
            st.pyplot(fig_iso)
    else:
        st.warning("Cleaned dataset not found. Please ensure the CSV is placed at `data/AI_SocialMedia_Student_Health_Dataset_clean.csv`.")

# 5. Main Tab: About the Dataset
elif app_mode == "About the Dataset":
    st.subheader("AI, Social Media, Student Health, and Academic Performance Dataset")
    
    st.markdown("""
    This project uses a cleaned dataset mapping student lifestyle factors, digital habits, mental health, and physical wellness to their risk of academic failure.
    
    ### Feature Explanations:
    * **Student_ID**: Categorical student identifier (dropped from model features to prevent memorization/leakage).
    * **Age**: Integer representation of student age (range 15-25 years).
    * **Gender**: Categorical demographic identification (Male, Female, Non-binary).
    * **Education_Level**: Academic tier of the student (High School, College, University).
    * **Daily_Social_Media_Hours**: Float tracking the average self-reported hours spent on social platforms per day.
    * **Daily_AI_Tool_Usage_Hours**: Float tracking average daily hours utilizing artificial intelligence assistance tools.
    * **Sleep_Hours**: Average self-reported nightly sleep duration in hours.
    * **Physical_Activity_Hours**: Average daily hours spent in physical exercises.
    * **Mental_Health_Score**: Scale value measuring general mental wellbeing.
    * **Physical_Health_Score**: Scale value measuring general physical wellness.
    * **Social_Isolation_Score**: Scale value indicating self-reported level of isolation.
    * **Burnout_Level**: Categorical representation of academic/emotional exhaustion (Low, Moderate, High, Severe).
    * **Academic_Performance_Score**: General student grade average scaled between 0 and 100.
    
    ### Target Variable:
    * **Academic_Failure_Risk**: Binary indicator where `1` represents a student at risk of failing, and `0` indicates normal academic standing.
    """)
    
    # Load dataset statistics
    stats_df = data_df
    if stats_df is None:
        base_dir = Path(__file__).resolve().parent
        clean_path = base_dir / "data" / "AI_SocialMedia_Student_Health_Dataset_clean.csv"
        if clean_path.exists():
            stats_df = pd.read_csv(clean_path)
            
    if stats_df is not None:
        st.markdown("### Descriptive Statistics")
        st.write("Summary statistics for the numerical features in the dataset:")
        num_cols = stats_df.select_dtypes(include=['int64', 'float64']).drop(columns=["Student_ID", "Academic_Failure_Risk"], errors="ignore").columns
        st.dataframe(stats_df[num_cols].describe().T.style.format("{:.3f}"))
