"""
Gradio application for Prudential Life Insurance Underwriting Assessment

This module provides a web interface for interacting with the trained model.
It can be run locally or deployed to platforms like Hugging Face Spaces.
"""

import os
import sys
import pandas as pd
import gradio as gr
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from src.mockup_app import launch_demo_with_dataset
from src.data import load_train, load_test
from src.persist import load_pipeline

# Load data and model
print("Loading data and model...")
X_test = load_test()  # We only need test data for the demo
pipeline = load_pipeline("data/processed/final_pipe.joblib")

# Set up the Gradio interface
print("Setting up Gradio interface...")
title = "Prudential Life Insurance Underwriting Assessment"
description = """
이 데모는 Prudential 생명보사의 보험 인수 심사(Underwriting)를 위한 머신러닝 모델을 보여줍니다.
아래 슬라이더를 조정하여 다른 테스트 케이스를 탐색하고 모델의 예측을 확인해보세요.
"""

# Create the Gradio interface
iface = launch_demo_with_dataset(
    pipeline=pipeline,
    X=X_test,
    class_names=[f"Class {i+1}" for i in range(8)]
)

# Update the interface with custom title and description
iface.title = title
iface.description = description

# Add additional examples
examples = [
    [0],  # First example
    [100],  # Second example
    [500],  # Third example
]

iface.examples = examples

# For local testing
if __name__ == "__main__":
    print("Starting Gradio server...")
    iface.launch(server_name="0.0.0.0", server_port=7860, share=False)
