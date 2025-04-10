```markdown
# Crop Price Prediction

This project predicts the modal price of crops using machine learning.

## Overview

The project trains a Linear Regression model on historical crop data to predict prices based on input features like commodity, variety, grade, location, and min/max prices.

## Project Structure

```
Crop_Price_Prediction/
├── backend/
│   └── predict.py         # Script for making price predictions
├── processdata/
│   └── CLEAN_CROP.csv     # Cleaned and preprocessed crop data
├── rawdata/
│   └── CROP.csv           # Raw crop price data (input)
├── traindata/
│   ├── commodityenconder.pkl
│   ├── districtenconder.pkl
│   ├── model.pkl
│   ├── stateenconder.pkl
│   └── varietyenconder.pkl
├── README.md              # This file
└── requirements.txt       # Project dependencies
```

## Dependencies

Install required libraries:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should contain:

```
pandas
joblib
scikit-learn
mlxtend
matplotlib
seaborn
numpy
```

(For specific versions, see the previous response.)

## Setup and Usage

1.  **Place your raw data file (`CROP.csv`) in the `rawdata` directory.**
2.  **Run the training script (e.g., `your_training_script_name.py`) to train the model:**
    ```bash
    python your_training_script_name.py
    ```
    This will generate `CLEAN_CROP.csv` in `processdata` and save the model and encoders in `traindata`.
3.  **Run the prediction script:**
    ```bash
    cd backend
    python predict.py
    ```
    Follow the prompts to enter crop details or generate random inputs for prediction.

## Data Sources

The project uses `CROP.csv` in the `rawdata` directory.

## Model

A Linear Regression model is used for prediction.

## Evaluation (in training script)

The training script calculates R-squared and Mean Squared Error to evaluate the model.

## Author

AVIJIT BHADRA
