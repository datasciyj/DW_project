import pandas as pd
import pickle


DATA_FILE_LOCATION = "..."
MODEL_FILE_LOCATION = "..." # global file location of pkl model file

def load_data():
    return pd.read_csv(DATA_FILE_LOCATION)


def load_model():
    return pickle.load(open(MODEL_FILE_LOCATION, 'rb'))


def create_features():
    df = load_data()
    df["A1"] = df["A1"] ** 2
    df["A2"] = df["A1"] ** 3
    df["A3"] = df["A1"] ** 3
    return df


def make_predictions():
    data = create_features()
    model = load_model()
    return model.predict(data)


def run():
    return make_predictions()


if __name__ == "__main__":
    run()
