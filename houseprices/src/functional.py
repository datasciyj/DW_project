import pandas as pd
import pickle


MODEL_FILE_LOCATION = "..." # global file location of pkl model file

def load_data(file):
    return pd.read_csv(file)


def load_model(file):
    return pickle.load(open(file, 'rb'))


def create_features(df):
    df["A1"] = df["A1"] ** 2
    df["A2"] = df["A1"] ** 3
    df["A3"] = df["A1"] ** 3
    return df


def make_predictions(df, model):
    return model.predict(df)


if __name__ == "__main__":
    file = input() # get user input for data file location
    data = load_data(file)
    model = load_model(MODEL_FILE_LOCATION)
    make_predictions(data, model)
