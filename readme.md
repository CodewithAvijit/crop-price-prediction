```
# Crop Price Prediction

This project predicts the modal price of crops using machine learning.

## 🧠 Overview

A Linear Regression model is trained on historical crop data to predict prices based on features like commodity, variety, grade, location, and minimum/maximum prices.
```
## 📁 Project Structure
```
Crop_Price_Prediction/
├── backend/
│   └── predict.py              # Script for making price predictions
├── processdata/
│   └── CLEAN_CROP.csv          # Cleaned and preprocessed crop data
├── rawdata/
│   └── CROP.csv                # Raw crop price data
├── traindata/
│   ├── commodityencoder.pkl
│   ├── districtencoder.pkl
│   ├── model.pkl
│   ├── stateencoder.pkl
│   └── varietyencoder.pkl
├── README.md                   # Project documentation
└── requirements.txt            # Project dependencies
```

## 📦 Dependencies

Install the required libraries using:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
pandas
joblib
scikit-learn
mlxtend
matplotlib
seaborn
numpy
```

## ⚙️ Setup and Usage

1. **Place the raw data file** `CROP.csv` into the `rawdata/` directory.

2. **Run the training script** to process the data and train the model:

   ```bash
   python your_training_script_name.py
   ```

   This will:
   - Generate `CLEAN_CROP.csv` in the `processdata/` folder
   - Save the model and encoders in the `traindata/` folder

3. **Run the prediction script**:

   ```bash
   cd backend
   python predict.py
   ```

   Follow the prompts to enter crop details or let it generate random inputs for prediction.

## 📊 Model Details

- **Algorithm**: Linear Regression  
- **Evaluation Metrics**: R-squared and Mean Squared Error (computed in training script)

## 📚 Data Source

- `rawdata/CROP.csv` — primary source of historical crop price data

## 👨‍💻 Author

**Avijit Bhadra**