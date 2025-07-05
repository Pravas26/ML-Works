# Student Performance Classifier

This app predicts if a student will **pass** or **fail** based on hours studied, attendance, and assignment completion using a Decision Tree model.

## Features
- Decision Tree Classifier
- Tree visualization saved as PNG
- Feature importance (via tree structure)
- Overfitting control using pruning (`max_depth`)
- Fully console-based (no Streamlit or GUI)

## Run Locally

```bash
pip install -r requirements.txt
python train.py
python predict.py
