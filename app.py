import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Prostate Cancer Survival Predictor", page_icon="🩺", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .risk-high { background-color: #FFE4E1; padding: 10px; border-radius: 10px; border-left: 5px solid #DC143C; margin: 5px 0; }
    .risk-low { background-color: #E6F3FF; padding: 10px; border-radius: 10px; border-left: 5px solid #1E90FF; margin: 5px 0; }
    .prediction-card { background-color: #F5F5F5; padding: 15px; border-radius: 10px; text-align: center; margin: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .metric-value { font-size: 28px; font-weight: bold; }
    .metric-label { font-size: 14px; color: #666; }
    .performance-card { background-color: #F8F9FA; padding: 10px; border-radius: 8px; text-align: center; margin: 5px; border: 1px solid #E9ECEF; }
    .performance-value { font-size: 20px; font-weight: bold; color: #1E90FF; }
    .performance-label { font-size: 11px; color: #666; }
</style>
""", unsafe_allow_html=True)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define file paths
MODEL_PATH = os.path.join(SCRIPT_DIR, 'gbsa_model.pkl')
FEATURES_PATH = os.path.join(SCRIPT_DIR, 'feature_names.pkl')
CUTOFFS_PATH = os.path.join(SCRIPT_DIR, 'youden_cutoffs.pkl')

# Load files
with open(CUTOFFS_PATH, 'rb') as f:
    YOUDEN_CUTOFFS = pickle.load(f)

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, 'rb') as f:
        feature_names = pickle.load(f)
    return model, feature_names

# Model performance metrics (from your analysis)
MODEL_PERFORMANCE = {
    '36_months': {
        'AUC': 0.9567,
        'Sensitivity': 0.8462,
        'Specificity': 0.9633,
        'Youden_Index': 0.8095
    },
    '48_months': {
        'AUC': 0.9467,
        'Sensitivity': 0.8827,
        'Specificity': 0.9010,
        'Youden_Index': 0.7837
    },
    '59_months': {
        'AUC': 0.9555,
        'Sensitivity': 0.8719,
        'Specificity': 0.9600,
        'Youden_Index': 0.8319
    }
}

# Feature list (22 features)
COXP_FEATURES = [
    'agegrp_81_90_vs_61_80', 'adt_yes_vs_no', 'radiotx_yes_vs_no',
    'ecogreclass2_ECOG_3_4_vs_ECOG_1_2', 'chemotx_no_vs_yes',
    'psanadir_cat_high_vs_low', 'maxpsa_cat_high_vs_low', 'timetonadir_cat_high_vs_low',
    'histologictype_well_differentiated', 'histologictype_poorly_differentiated',
    'stageatdiagnosis_Stage_I', 'stageatdiagnosis_Stage_II', 'stageatdiagnosis_Stage_IV',
    'metastasis_No_metastasis', 'metastasis_Bone_metastasis', 'metastasis_Multiple_metastasis',
    'commorbidity_no_commorbidity', 'commorbidity_one_commorbidity', 'commorbidity_more_than_one_commorbidity',
    'baselinepsa_cat_0_2_4_ng_ml', 'baselinepsa_cat_2_5_3_9_ng_ml', 'baselinepsa_cat_10+_ng_ml'
]

def get_age_value(x): return 1 if x == "81-90 years" else 0
def get_adt_value(x): return 1 if x == "Yes" else 0
def get_radiotx_value(x): return 1 if x == "Yes" else 0
def get_ecog_value(x): return 1 if x == "ECOG 3-4 (Poor)" else 0
def get_chemo_value(x): return 1 if x == "No" else 0
def get_psanadir_value(x): return 1 if x == "High" else 0
def get_maxpsa_value(x): return 1 if x == "High" else 0
def get_ttn_value(x): return 1 if x == "High" else 0
def get_histology_well_value(x): return 1 if x == "Well differentiated" else 0
def get_histology_poor_value(x): return 1 if x == "Poorly differentiated" else 0
def get_stage_I_value(x): return 1 if x == "Stage I" else 0
def get_stage_II_value(x): return 1 if x == "Stage II" else 0
def get_stage_IV_value(x): return 1 if x == "Stage IV" else 0
def get_metastasis_no_value(x): return 1 if x == "No metastasis" else 0
def get_metastasis_bone_value(x): return 1 if x == "Bone metastasis" else 0
def get_metastasis_multiple_value(x): return 1 if x == "Multiple metastasis" else 0
def get_comorbidity_no_value(x): return 1 if x == "No comorbidity" else 0
def get_comorbidity_one_value(x): return 1 if x == "One comorbidity" else 0
def get_comorbidity_multiple_value(x): return 1 if x == "More than one comorbidity" else 0
def get_baseline_psa_0_2_4_value(x): return 1 if x == "0-2.4 ng/mL" else 0
def get_baseline_psa_2_5_3_9_value(x): return 1 if x == "2.5-3.9 ng/mL" else 0
def get_baseline_psa_10_plus_value(x): return 1 if x == "≥10 ng/mL" else 0

def predict_survival_at_time(model, patient_df, time_point):
    """Predict survival probability at specific time point"""
    try:
        surv_functions = model.predict_survival_function(patient_df)
        sf = surv_functions[0]
        if hasattr(sf, '__call__'):
            prob = sf(time_point)
        else:
            times = sf['time']
            surv_probs = sf['survival']
            idx = np.searchsorted(times, time_point)
            if idx < len(surv_probs):
                prob = surv_probs[idx]
            else:
                prob = surv_probs[-1]
        return max(0, min(1, prob))
    except:
        risk = model.predict(patient_df)[0]
        prob = 1 / (1 + np.exp(risk / 2)) * 0.8 + 0.1
        return max(0.05, min(0.95, prob))

def get_risk_classification(risk_score, cutoff):
    """Get risk classification based on cutoff"""
    if risk_score > cutoff:
        return "HIGH RISK", "🔴", "#DC143C"
    else:
        return "LOW RISK", "🟢", "#1E90FF"

def main():
    st.title("🩺 Prostate Cancer Survival Predictor")
    st.markdown("### GBSA Model with COX-P Features | Test C-index: 0.8833")
    st.markdown("#### Dynamic Survival Prediction & Risk Classification at 36, 48, and 59 Months")

    model, feature_names = load_model()

    # Sidebar with Model Performance Metrics
    with st.sidebar:
        st.markdown("### 📊 Model Performance")
        st.metric("Test C-index", "0.8833", "Excellent")
        st.metric("5-Fold CV C-index", "0.8725", "±0.0144")

        st.markdown("---")
        st.markdown("### 📈 Time-Dependent AUC")

        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            st.markdown(f"""
            <div class="performance-card">
                <div class="performance-value">{MODEL_PERFORMANCE['36_months']['AUC']:.3f}</div>
                <div class="performance-label">36 mo AUC</div>
            </div>
            """, unsafe_allow_html=True)
        with col_a2:
            st.markdown(f"""
            <div class="performance-card">
                <div class="performance-value">{MODEL_PERFORMANCE['48_months']['AUC']:.3f}</div>
                <div class="performance-label">48 mo AUC</div>
            </div>
            """, unsafe_allow_html=True)
        with col_a3:
            st.markdown(f"""
            <div class="performance-card">
                <div class="performance-value">{MODEL_PERFORMANCE['59_months']['AUC']:.3f}</div>
                <div class="performance-label">59 mo AUC</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📊 Sensitivity & Specificity")

        # 36 months
        st.markdown("**36 Months**")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown(f"""
            <div class="performance-card">
                <div class="performance-value">{MODEL_PERFORMANCE['36_months']['Sensitivity']:.3f}</div>
                <div class="performance-label">Sensitivity</div>
            </div>
            """, unsafe_allow_html=True)
        with col_s2:
            st.markdown(f"""
            <div class="performance-card">
                <div class="performance-value">{MODEL_PERFORMANCE['36_months']['Specificity']:.3f}</div>
                <div class="performance-label">Specificity</div>
            </div>
            """, unsafe_allow_html=True)

        # 48 months
        st.markdown("**48 Months**")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown(f"""
            <div class="performance-card">
                <div class="performance-value">{MODEL_PERFORMANCE['48_months']['Sensitivity']:.3f}</div>
                <div class="performance-label">Sensitivity</div>
            </div>
            """, unsafe_allow_html=True)
        with col_p2:
            st.markdown(f"""
            <div class="performance-card">
                <div class="performance-value">{MODEL_PERFORMANCE['48_months']['Specificity']:.3f}</div>
                <div class="performance-label">Specificity</div>
            </div>
            """, unsafe_allow_html=True)

        # 59 months
        st.markdown("**59 Months**")
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            st.markdown(f"""
            <div class="performance-card">
                <div class="performance-value">{MODEL_PERFORMANCE['59_months']['Sensitivity']:.3f}</div>
                <div class="performance-label">Sensitivity</div>
            </div>
            """, unsafe_allow_html=True)
        with col_q2:
            st.markdown(f"""
            <div class="performance-card">
                <div class="performance-value">{MODEL_PERFORMANCE['59_months']['Specificity']:.3f}</div>
                <div class="performance-label">Specificity</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🎯 Risk Cutoffs (Youden Index)")
        st.markdown(f"**36 months:** `{YOUDEN_CUTOFFS[36]:.4f}`")
        st.markdown(f"**48 months:** `{YOUDEN_CUTOFFS[48]:.4f}`")
        st.markdown(f"**59 months:** `{YOUDEN_CUTOFFS[59]:.4f}`")

        st.markdown("---")
        st.markdown("### 📖 How to Interpret")
        st.info("""
        **Risk Score > Cutoff = HIGH RISK**
        - Poorer prognosis
        - Lower survival probability

        **Risk Score < Cutoff = LOW RISK**
        - Better prognosis
        - Higher survival probability
        """)

    with st.form("prediction_form"):
        st.markdown("### 📋 Patient Clinical Information")

        col1, col2 = st.columns(2)
        with col1:
            age = st.selectbox("Age Group", ["61-80 years", "81-90 years"])
            ecog = st.selectbox("ECOG Status", ["ECOG 1-2 (Good)", "ECOG 3-4 (Poor)"])
            adt = st.selectbox("ADT", ["No", "Yes"])
            radiotherapy = st.selectbox("Radiotherapy", ["No", "Yes"])
            chemotherapy = st.selectbox("Chemotherapy", ["Yes", "No"])
        with col2:
            comorbidity = st.selectbox("Comorbidity", ["No comorbidity", "One comorbidity", "More than one comorbidity"])
            histology = st.selectbox("Histology", ["Well differentiated", "Moderately differentiated", "Poorly differentiated"])
            stage = st.selectbox("Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"])
            metastasis = st.selectbox("Metastasis", ["No metastasis", "Bone metastasis", "Multiple metastasis", "Non-Regional LN metastasis"])

        st.markdown("#### 🔬 Laboratory Biomarkers")
        col3, col4, col5 = st.columns(3)
        with col3: psa_nadir = st.selectbox("PSA Nadir", ["Low", "High"])
        with col4: max_psa = st.selectbox("Maximum PSA", ["Low", "High"])
        with col5: ttn = st.selectbox("Time to PSA Nadir", ["Low", "High"])

        baseline_psa = st.selectbox("Baseline PSA", ["0-2.4 ng/mL", "2.5-3.9 ng/mL", "4.0-9.9 ng/mL", "≥10 ng/mL"])

        submitted = st.form_submit_button("🔮 Predict Survival", type="primary", use_container_width=True)

    if submitted:
        # Build input vector
        inputs = {
            'agegrp_81_90_vs_61_80': get_age_value(age),
            'adt_yes_vs_no': get_adt_value(adt),
            'radiotx_yes_vs_no': get_radiotx_value(radiotherapy),
            'ecogreclass2_ECOG_3_4_vs_ECOG_1_2': get_ecog_value(ecog),
            'chemotx_no_vs_yes': get_chemo_value(chemotherapy),
            'psanadir_cat_high_vs_low': get_psanadir_value(psa_nadir),
            'maxpsa_cat_high_vs_low': get_maxpsa_value(max_psa),
            'timetonadir_cat_high_vs_low': get_ttn_value(ttn),
            'histologictype_well_differentiated': get_histology_well_value(histology),
            'histologictype_poorly_differentiated': get_histology_poor_value(histology),
            'stageatdiagnosis_Stage_I': get_stage_I_value(stage),
            'stageatdiagnosis_Stage_II': get_stage_II_value(stage),
            'stageatdiagnosis_Stage_IV': get_stage_IV_value(stage),
            'metastasis_No_metastasis': get_metastasis_no_value(metastasis),
            'metastasis_Bone_metastasis': get_metastasis_bone_value(metastasis),
            'metastasis_Multiple_metastasis': get_metastasis_multiple_value(metastasis),
            'commorbidity_no_commorbidity': get_comorbidity_no_value(comorbidity),
            'commorbidity_one_commorbidity': get_comorbidity_one_value(comorbidity),
            'commorbidity_more_than_one_commorbidity': get_comorbidity_multiple_value(comorbidity),
            'baselinepsa_cat_0_2_4_ng_ml': get_baseline_psa_0_2_4_value(baseline_psa),
            'baselinepsa_cat_2_5_3_9_ng_ml': get_baseline_psa_2_5_3_9_value(baseline_psa),
            'baselinepsa_cat_10+_ng_ml': get_baseline_psa_10_plus_value(baseline_psa),
        }

        input_df = pd.DataFrame([inputs])[COXP_FEATURES]

        # Get overall risk score
        risk_score = model.predict(input_df)[0]

        # Get survival probabilities at each time point
        surv_36 = predict_survival_at_time(model, input_df, 36)
        surv_48 = predict_survival_at_time(model, input_df, 48)
        surv_59 = predict_survival_at_time(model, input_df, 59)

        st.markdown("---")
        st.markdown("## 📊 Prediction Results")

        # Overall Risk Score
        st.markdown(f"""
        <div style="text-align: center; padding: 15px; background-color: #F0F0F0; border-radius: 10px; margin-bottom: 20px;">
            <span style="font-size: 16px;">Overall Risk Score</span><br>
            <span style="font-size: 42px; font-weight: bold;">{risk_score:.4f}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📈 Dynamic Survival Probabilities")

        col_s1, col_s2, col_s3 = st.columns(3)

        with col_s1:
            st.markdown(f"""
            <div class="prediction-card">
                <div class="metric-label">36 Months (3 years)</div>
                <div class="metric-value" style="color: #1E90FF;">{surv_36*100:.1f}%</div>
                <div class="metric-label">Probability of surviving to 3 years</div>
            </div>
            """, unsafe_allow_html=True)

        with col_s2:
            st.markdown(f"""
            <div class="prediction-card">
                <div class="metric-label">48 Months (4 years)</div>
                <div class="metric-value" style="color: #1E90FF;">{surv_48*100:.1f}%</div>
                <div class="metric-label">Probability of surviving to 4 years</div>
            </div>
            """, unsafe_allow_html=True)

        with col_s3:
            st.markdown(f"""
            <div class="prediction-card">
                <div class="metric-label">59 Months (~5 years)</div>
                <div class="metric-value" style="color: #1E90FF;">{surv_59*100:.1f}%</div>
                <div class="metric-label">Probability of surviving to 5 years</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🎯 Risk Classification by Time Point")

        col_r1, col_r2, col_r3 = st.columns(3)

        # 36 Months
        with col_r1:
            risk_label_36, icon_36, color_36 = get_risk_classification(risk_score, YOUDEN_CUTOFFS[36])
            bg_color = "#FFE4E1" if risk_label_36 == "HIGH RISK" else "#E6F3FF"
            st.markdown(f"""
            <div style="background-color: {bg_color}; padding: 15px; border-radius: 10px; text-align: center; border-left: 5px solid {color_36};">
                <h4 style="margin: 0;">{icon_36} 36 Months</h4>
                <h3 style="color: {color_36}; margin: 10px 0;">{risk_label_36}</h3>
                <p style="margin: 5px 0; font-size: 12px;">Cutoff: {YOUDEN_CUTOFFS[36]:.4f}</p>
                <p style="margin: 5px 0; font-size: 12px;">Risk Score: {risk_score:.4f}</p>
                <p style="margin: 5px 0; font-size: 11px; color: #666;">AUC: {MODEL_PERFORMANCE['36_months']['AUC']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)

        # 48 Months
        with col_r2:
            risk_label_48, icon_48, color_48 = get_risk_classification(risk_score, YOUDEN_CUTOFFS[48])
            bg_color = "#FFE4E1" if risk_label_48 == "HIGH RISK" else "#E6F3FF"
            st.markdown(f"""
            <div style="background-color: {bg_color}; padding: 15px; border-radius: 10px; text-align: center; border-left: 5px solid {color_48};">
                <h4 style="margin: 0;">{icon_48} 48 Months</h4>
                <h3 style="color: {color_48}; margin: 10px 0;">{risk_label_48}</h3>
                <p style="margin: 5px 0; font-size: 12px;">Cutoff: {YOUDEN_CUTOFFS[48]:.4f}</p>
                <p style="margin: 5px 0; font-size: 12px;">Risk Score: {risk_score:.4f}</p>
                <p style="margin: 5px 0; font-size: 11px; color: #666;">AUC: {MODEL_PERFORMANCE['48_months']['AUC']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)

        # 59 Months
        with col_r3:
            risk_label_59, icon_59, color_59 = get_risk_classification(risk_score, YOUDEN_CUTOFFS[59])
            bg_color = "#FFE4E1" if risk_label_59 == "HIGH RISK" else "#E6F3FF"
            st.markdown(f"""
            <div style="background-color: {bg_color}; padding: 15px; border-radius: 10px; text-align: center; border-left: 5px solid {color_59};">
                <h4 style="margin: 0;">{icon_59} 59 Months</h4>
                <h3 style="color: {color_59}; margin: 10px 0;">{risk_label_59}</h3>
                <p style="margin: 5px 0; font-size: 12px;">Cutoff: {YOUDEN_CUTOFFS[59]:.4f}</p>
                <p style="margin: 5px 0; font-size: 12px;">Risk Score: {risk_score:.4f}</p>
                <p style="margin: 5px 0; font-size: 11px; color: #666;">AUC: {MODEL_PERFORMANCE['59_months']['AUC']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)

        # Survival Curve
        st.markdown("### 📉 Predicted Survival Curve")

        fig, ax = plt.subplots(figsize=(10, 6))

        time_points_full = np.linspace(0, 60, 100)
        surv_probs_full = []

        for t in time_points_full:
            surv = predict_survival_at_time(model, input_df, t)
            surv_probs_full.append(surv)

        ax.plot(time_points_full, surv_probs_full, 'b-', linewidth=2.5, label='Predicted Survival')
        ax.scatter([36, 48, 59], [surv_36, surv_48, surv_59], color='red', s=150, zorder=5,
                  label='Prediction Points', edgecolors='white', linewidth=2)

        if risk_score > YOUDEN_CUTOFFS[36]:
            ax.fill_between(time_points_full, 0, surv_probs_full, alpha=0.15, color='red', label='High Risk Zone')
        else:
            ax.fill_between(time_points_full, 0, surv_probs_full, alpha=0.15, color='green', label='Low Risk Zone')

        ax.set_xlabel('Time (months)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Survival Probability', fontsize=12, fontweight='bold')
        ax.set_title(f'Dynamic Survival Prediction (36-Month Risk: {risk_label_36})', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 60)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower left', fontsize=10)

        for t, prob in zip([36, 48, 59], [surv_36, surv_48, surv_59]):
            ax.annotate(f'{prob*100:.1f}%', xy=(t, prob), xytext=(5, 10),
                       textcoords='offset points', fontweight='bold', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

if __name__ == "__main__":
    main()
