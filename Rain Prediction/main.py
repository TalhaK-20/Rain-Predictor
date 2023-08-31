import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def random_data_CSV():
    np.random.seed(42)
    num_samples = 100
    temperature = np.random.randint(20, 35, num_samples)
    humidity = np.random.randint(40, 80, num_samples)
    rainy = np.random.choice([0, 1], num_samples, p=[0.7, 0.3])  # Simulate rainy days (1) and non-rainy days (0)

    data = pd.DataFrame({'temperature': temperature, 'humidity': humidity, 'rainy': rainy})

    data.to_csv('Rain_Random_Data.csv', index=False)


def Prediction():
    data = pd.read_csv('Rain_Random_Data.csv')
    X = data[['temperature', 'humidity']]
    y = data['rainy']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    temp = int(input("Enter the value of temperature in Celsius = "))
    humid = int(input("Enter the value of humidity = "))

    new_data = pd.DataFrame({'temperature': [temp], 'humidity': [humid]})
    predicted_rain = model.predict(new_data)
    print(
        f'Will it rain for temperature = {new_data["temperature"].iloc[0]} and '
        f'humidity = {new_data["humidity"].iloc[0]}'
        f'?'f'{" Yes" if predicted_rain[0] else " No"}')


def repeat():
    while True:
        print("Welcome to the Simple Rain Prediction Software")
        print("        Developed by Talha Khalid\n")
        print("** Important Note ** \nFor making this software, Talha Khalid has used "
              "random generated data for temperature and humidity to train the model.")

        Prediction()

        user = input("Do you want to use it again. Press Y/N = ").lower()
        print("\n")
        if user != 'y' and user == "n":
            print("Thanks for using. Have a lovely day :)")
            break

        elif user != 'n' and user != 'y':
            print("You neither press Y nor N so I am Shutting Down the software..."
                  " Thanks for using. Have a lovely day :)")
            break


repeat()
