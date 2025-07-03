# Bangalore House Price Predictor

This is a Linear Regression-based ML project that predicts house prices based on location, total area, and BHK.

## Files
- `train.py`: Trains and saves the model
- `app.py`: Streamlit app for prediction
- `housing.csv`: Dataset (place inside `data/`)
- `model/linear_model.pkl`: Trained model
- `requirements.txt`: Python dependencies

## How to Run

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
