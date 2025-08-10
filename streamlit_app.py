"""
Streamlit application for Prudential Life Insurance Underwriting Assessment

This module provides a web interface for interacting with the trained model
using Streamlit, which can be easily deployed to Streamlit Cloud.
"""

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import sys
import io
from typing import List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from PIL import Image

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

# Set page config
st.set_page_config(
    page_title="Prudential Life Insurance Assessment",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .feature-importance {
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
        background-color: #f0f2f6;
    }
    .ai-explanation {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border-left: 4px solid #4e79a7;
    }
    </style>
""", unsafe_allow_html=True)

def generate_ai_explanation(pred_class: int, top_features: List[Tuple[str, float]]) -> str:
    """Generate an AI explanation for the prediction.

    Business rule: Only Class 1 => reject; otherwise => approve.
    """
    try:
        from src.genai import generate_underwriting_explanation

        decision = "reject" if pred_class == 1 else "approve"

        explanation = generate_underwriting_explanation(
            decision=decision,
            pred_class=pred_class,
            top_features=top_features,
            probs=None,
            sample=None,
        )
        return explanation
    except Exception as e:
        st.warning(f"Could not generate AI explanation: {str(e)}")
        return "AI 설명을 생성할 수 없습니다. (AI explanation not available)"

def get_top_contributing_features(sample: pd.Series, n: int = 5) -> List[Tuple[str, float]]:
    """Get top n contributing features from a sample"""
    # Convert to numeric, coercing errors to NaN, then drop NaN values
    numeric_values = pd.to_numeric(sample, errors='coerce').dropna()
    
    # Get absolute values and sort
    if not numeric_values.empty:
        # Sort by absolute value and get top n
        top_features = numeric_values.abs().sort_values(ascending=False).head(n)
        # Return list of (feature_name, value) tuples
        return [(feat, sample[feat]) for feat in top_features.index]
    return []

def load_model_and_data():
    """Load the trained model and test data"""
    try:
        from src.data import load_test, ID_COL, load_train
        from src.persist import load_pipeline
        from src.config import TARGET_COL
        from src.shap_utils import (
            prime_shap_cache,
            waterfall_figure_for_instance,
            _extract_estimator,
            _is_lgbm
        )
        
        with st.spinner("Loading model and data..."):
            # Load test data (excluding ID column)
            X_test = load_test(drop_id=True)
            
            # Load the model
            pipeline = load_pipeline("data/processed/final_pipe.joblib")
            
            # Get feature names from the model or column names
            if hasattr(pipeline, 'feature_names_in_'):
                feature_names = list(pipeline.feature_names_in_)
            else:
                feature_names = [col for col in X_test.columns if col != ID_COL and col != TARGET_COL]
            
            return X_test, pipeline, feature_names
            
    except Exception as e:
        st.error(f"Error loading the model or data: {str(e)}")
        st.stop()

def predict_sample(model, sample):
    """Make prediction for a single sample"""
    try:
        # Convert sample to DataFrame if it's not already
        if not isinstance(sample, pd.DataFrame):
            sample = pd.DataFrame([sample])
        
        # Ensure we're only using features that the model expects
        if hasattr(model, 'feature_names_in_'):
            missing_cols = set(model.feature_names_in_) - set(sample.columns)
            if missing_cols:
                for col in missing_cols:
                    sample[col] = 0  # Add missing columns with default value 0
            # Reorder columns to match model's expected order
            sample = sample[model.feature_names_in_]
        
        # Make prediction
        pred_proba = model.predict_proba(sample)[0]
        pred_class = np.argmax(pred_proba) + 1  # Assuming classes start from 1
        
        return pred_class, pred_proba, sample
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

def main():
    """Main Streamlit app function"""
    # App title and description
    st.title("Prudential Life Insurance Underwriting Assessment")
    st.markdown("""
    이 애플리케이션은 보험 인수 심사(Underwriting)를 위한 머신러닝 모델을 시각화한 데모입니다. 
    아래에서 테스트 케이스를 선택하거나 직접 값을 조정하여 모델의 예측 결과를 확인해보세요.
    """)
    
    # Load model and data
    X_test, pipeline, feature_names = load_model_and_data()
    
    # Add a sample selector
    st.sidebar.header("Test Case Selection")
    sample_idx = st.sidebar.slider(
        "Select a test case:", 
        min_value=0, 
        max_value=len(X_test)-1, 
        value=0,
        help="Select a test case to see the model's prediction"
    )
    
    # Get the selected sample
    sample = X_test.iloc[[sample_idx]]
    
    # Make prediction
    pred_class, pred_proba, processed_sample = predict_sample(pipeline, sample)
    
    if pred_class is not None and processed_sample is not None:
        # Prepare container for SHAP top contributors (LightGBM-only)
        shap_top_features: List[Tuple[str, float]] = []
        
        st.subheader("Model Prediction & Analysis")
        
        # First row: Predicted Class
        st.markdown("#### Prediction")
        st.metric("Predicted Class", f"Class {pred_class}")
        
        # Second row: Equal width columns for Class Probabilities and SHAP
        col_proba, col_shap = st.columns(2)
        
        with col_proba:
            # Display probability distribution
            st.markdown("#### Class Probabilities")
            for i, prob in enumerate(pred_proba, 1):
                st.progress(float(prob), text=f"Class {i}: {prob:.2%}")
        
        # SHAP Waterfall chart section in the right column
        with col_shap:
            st.markdown("#### Feature Impact Analysis")
            try:
                from src.shap_utils import (
                    _extract_estimator,
                    _is_lgbm,
                    waterfall_figure_for_instance,
                    top_contributors_for_instance,
                )
                
                clf = _extract_estimator(pipeline)
                if _is_lgbm(clf):
                    with st.spinner("Generating SHAP Waterfall chart..."):
                        # Create a copy of the sample with the ID column if it exists
                        sample_for_shap = sample.copy()
                        
                        # Compute SHAP top contributors for AI explanation
                        try:
                            contrib_info = top_contributors_for_instance(
                                pipeline, sample_for_shap, index=0, top_k=10
                            )
                            shap_top_features = contrib_info.get("top_features", [])
                        except Exception:
                            shap_top_features = []

                        # Generate the waterfall plot
                        fig = waterfall_figure_for_instance(
                            pipeline,
                            sample_for_shap,
                            index=0,  # First (and only) row
                            top_k=10  # Show top 10 features
                        )
                        
                        # Convert matplotlib figure to PIL Image
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', bbox_inches='tight')
                        plt.close(fig)  # Close the figure to free memory
                        buf.seek(0)
                        img = Image.open(buf)
                        
                        # Display the image in Streamlit
                        st.image(img, use_container_width=True)
                        
                        # Add a caption explaining the plot
                        st.markdown("""
                        <div style="font-size: 0.9em; color: #666; margin-top: -10px; margin-bottom: 1.5rem;">
                            This waterfall plot shows how each feature contributes to the final prediction.
                            Features are ordered by their impact on the prediction, with the most influential
                            features at the top. Blue bars indicate features that increase the prediction,
                            while red bars indicate features that decrease it.
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("SHAP Waterfall charts are only available for LightGBM models.")
            except Exception as e:
                st.warning(f"Could not generate SHAP Waterfall chart: {str(e)}")
        
        # Add AI explanation section at the bottom
        st.markdown("---")
        st.markdown("### AI Explanation")
        with st.spinner("Generating AI explanation..."):
            explanation = generate_ai_explanation(pred_class, shap_top_features)
            st.markdown(f"""
            <div class="ai-explanation">
                {explanation}
            </div>
            """, unsafe_allow_html=True)
    
    # Add some information in the sidebar
    with st.sidebar:
        st.markdown("---")
        st.header("About")
        st.markdown("""
        **Prudential Life Insurance Assessment**  
        이 애플리케이션은 보험 가입자의 위험 등급을 예측하는 모델을 보여줍니다.
        
        **사용 방법:**
        1. 왼쪽 슬라이더로 테스트 케이스 선택
        2. 모델 예측 결과 확인
        3. 다른 테스트 케이스로 실험해보기
        """)
        
        st.markdown("---")
        st.markdown("### Model Information")
        st.code("""
        Model: XGBoost Classifier
        Features: 128
        Classes: 8
        Accuracy: ~0.65
        """)

if __name__ == "__main__":
    main()
