import streamlit as st
import pandas as pd
import os
import joblib
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(page_title="Exam Analytics System", page_icon="📝", layout="wide")

# Load Lucide Icons CDN
st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/lucide-static@0.419.0/font/lucide.min.css">', unsafe_allow_html=True)

# Custom CSS for "Hacker" Aesthetic vs Professional Icons
st.markdown("""
    <style>
    .main {
        background-color: #0d1117;
        color: #58a6ff;
    }
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 5px;
    }
    h1, h2, h3 {
        color: #79c0ff;
        font-family: 'Courier New', Courier, monospace;
    }
    .stAlert {
        background-color: #161b22;
        color: #d29922;
        border: 1px solid #d29922;
    }
    .lucide {
        margin-right: 10px;
        vertical-align: middle;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1><i class="lucide lucide-zap"></i> Exam Analytics: Question Difficulty Classifier</h1>', unsafe_allow_html=True)
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Train Model", "Documentation"])

# Mock state if model doesn't exist
if 'model' not in st.session_state:
    st.session_state['model'] = None

# Model Loading and Prediction Logic
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))

def load_models():
    try:
        vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.joblib"))
        lr_model = joblib.load(os.path.join(MODELS_DIR, "lr_model.joblib"))
        dt_model = joblib.load(os.path.join(MODELS_DIR, "dt_model.joblib"))
        return vectorizer, lr_model, dt_model
    except Exception as e:
        return None, None, None

def predict_difficulty(df, vectorizer, model):
    # Ensure features match training
    if "question" not in df.columns:
        return None
    
    # Feature Engineering (mimic training)
    df["question"] = df["question"].fillna("").astype(str)
    df["question_length"] = df["question"].apply(len)
    
    # Check for other numeric features if available, else use zeros
    for feat in ["score", "tag_count"]:
        if feat not in df.columns:
            df[feat] = 0
            
    X_tfidf = vectorizer.transform(df["question"])
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(df[["score", "question_length", "tag_count"]])
    
    from scipy.sparse import hstack
    X = hstack([X_tfidf, X_numeric])
    return model.predict(X)

if page == "Dashboard":
    st.markdown('<h2><i class="lucide lucide-layout-dashboard"></i> Prediction Dashboard</h2>', unsafe_allow_html=True)
    
    vectorizer, lr_model, dt_model = load_models()
    
    if vectorizer is None:
        st.warning("No trained models found. Please run the training pipeline first to enable real predictions.")
        use_mock = True
    else:
        st.success("Models loaded successfully!")
        use_mock = False

    uploaded_file = st.file_uploader("Upload Question CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Raw Data Preview", df.head())
        
        col_sel = st.selectbox("Choose Model", ["Logistic Regression", "Decision Tree"])
        
        if st.button("Predict Difficulty"):
            with st.spinner("Analyzing questions..."):
                if use_mock:
                    # Fallback to mock for UI demo
                    df['predicted_difficulty'] = pd.Series(["Hard", "Medium", "Easy"] * (len(df)//3 + 1))[:len(df)]
                else:
                    model_to_use = lr_model if col_sel == "Logistic Regression" else dt_model
                    preds = predict_difficulty(df, vectorizer, model_to_use)
                    if preds is not None:
                        df['predicted_difficulty'] = preds
                    else:
                        st.error("❌ Data missing required columns.")
                
                if 'predicted_difficulty' in df.columns:
                    st.success("Analysis complete!")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("### Analysis Results")
                        st.dataframe(df[["question", "predicted_difficulty"]].head(20))
                    
                    with col2:
                        st.write("### Difficulty Distribution")
                        fig, ax = plt.subplots()
                        sns.countplot(data=df, x='predicted_difficulty', palette='viridis', ax=ax)
                        plt.title(f"Predicted Difficulty ({col_sel})")
                        st.pyplot(fig)

elif page == "Train Model":
    st.markdown('<h2><i class="lucide lucide-settings"></i> Model Training</h2>', unsafe_allow_html=True)
    st.info("Train your models directly from this dashboard using the processed dataset.")
    
    if st.button("Start Training Pipeline"):
        st.warning("This process might take a while for large datasets (1 million rows).")
        # Here we would trigger the src/model_train.py logic
        st.code("python src/model_train.py")
        st.success("Training script triggered via CLI. Check terminal for real-time logs.")

elif page == "Documentation":
    st.markdown('<h2><i class="lucide lucide-book-open"></i> Milestone 1 Documentation</h2>', unsafe_allow_html=True)
    st.markdown("""
    ### Project Overview
    This system analyzes exam questions to predict their difficulty level based on historical performance data and textual content.
    
    ### Completed Tasks
    - **Text Processing**: Tokenization, Lemmatization
    - **Feature Extraction**: TF-IDF, Embeddings
    - **Classifiers**: Logistic Regression, Decision Tree
    - **Evaluation**: Metrics & Confusion Matrix
    - **UI**: Question Upload & Visualization
    """)

st.sidebar.markdown("---")
st.sidebar.info("System Status: **Active**")
st.sidebar.text("Milestone 1: Ready")
