import pandas as pd
import joblib
import numpy as np
import random

def predict_price(data):
    """
    Predicts the modal price of a crop based on the provided features.

    Args:
        data (dict): A dictionary containing the input features:
                     'Commodity', 'Variety', 'Grade', 'State', 'District',
                     'Min_Price', 'Max_Price'.

    Returns:
        float: The predicted modal price.
    """
    # Load the encoders
    state_encoder = joblib.load("traindata/stateenconder.pkl")
    district_encoder = joblib.load("traindata/districtenconder.pkl")
    commodity_encoder = joblib.load("traindata/commodityenconder.pkl")
    variety_encoder = joblib.load("traindata/varietyenconder.pkl")
    model = joblib.load("traindata/model.pkl")

    # Encode the input features
    encoded_state = state_encoder.transform([data['State']])[0]
    encoded_district = district_encoder.transform([data['District']])[0]
    encoded_commodity = commodity_encoder.transform([data['Commodity']])[0]
    encoded_variety = variety_encoder.transform([data['Variety']])[0]

    grade_faq = 1 if data['Grade'].upper() == 'FAQ' else 0
    grade_small = 1 if data['Grade'].upper() == 'SMALL' else 0

    # Create the input array for prediction
    input_features = np.array([[encoded_commodity, encoded_variety, grade_faq, grade_small,
                                encoded_state, encoded_district, data['Min_Price'], data['Max_Price']]])

    # Make the prediction
    predicted_price = model.predict(input_features)[0]
    return predicted_price

def get_random_input():
    """Generates a random input dictionary."""
    # Load unique values from the training data (assuming CLEAN_CROP.csv is available)
    try:
        df = pd.read_csv("processdata/CLEAN_CROP.csv")
        commodities = df['Commodity'].unique().tolist()
        varieties = df['Variety'].unique().tolist()
        states = df['State'].unique().tolist()
        districts = df['District'].unique().tolist()
        grades = ['FAQ', 'Small']
        min_price = random.randint(500, 5000)
        max_price = random.randint(min_price + 100, 8000)

        random_input = {
            'Commodity': random.choice(commodities),
            'Variety': random.choice(varieties),
            'Grade': random.choice(grades),
            'State': random.choice(states),
            'District': random.choice(districts),
            'Min_Price': float(min_price),
            'Max_Price': float(max_price)
        }
        return random_input
    except FileNotFoundError:
        print("Error: CLEAN_CROP.csv not found. Cannot generate random input.")
        return None

def get_user_input():
    """Gets input from the user."""
    user_input = {}
    user_input['Commodity'] = input("Enter Commodity: ")
    user_input['Variety'] = input("Enter Variety: ")
    user_input['Grade'] = input("Enter Grade ('FAQ' or 'Small'): ")
    user_input['State'] = input("Enter State: ")
    user_input['District'] = input("Enter District: ")
    try:
        user_input['Min_Price'] = float(input("Enter Minimum Price: "))
        user_input['Max_Price'] = float(input("Enter Maximum Price: "))
    except ValueError:
        print("Invalid price input. Please enter numbers.")
        return None
    return user_input

if __name__ == '__main__':
    while True:
        print("\nChoose input method:")
        print("1. Random Input")
        print("2. User Input")
        print("3. Exit")

        choice = input("Enter your choice (1, 2, or 3): ")

        if choice == '1':
            input_data = get_random_input()
            if input_data:
                predicted_price = predict_price(input_data)
                print("\nUsing Random Input:")
                for key, value in input_data.items():
                    print(f"{key}: {value}")
                print(f"Predicted Modal Price: {predicted_price:.2f}")

        elif choice == '2':
            input_data = get_user_input()
            if input_data:
                predicted_price = predict_price(input_data)
                print("\nUsing User Input:")
                for key, value in input_data.items():
                    print(f"{key}: {value}")
                print(f"Predicted Modal Price: {predicted_price:.2f}")

        elif choice == '3':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")