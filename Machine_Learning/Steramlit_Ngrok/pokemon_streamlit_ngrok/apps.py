import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path


@st.cache_data
def load_data():
    current_dir = Path(__file__).parent
    csv_path = current_dir / "Pokemon_Complete_Gen1_to_Gen9.csv"
    df = pd.read_csv(csv_path)
    return df

@st.cache_resource
def train_model(df):
    df = df.copy()

    df["type2"] = df["type2"].fillna("")
    df["label"] = df["type2"].apply(lambda x: "Single Type" if x == "" else "Dual Type")

    X = df[["hp", "attack", "defense", "speed"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5))
        ]
    )

    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    return model, accuracy


def main():
    st.set_page_config(
        page_title="Pokemon KNN Classification",
        layout="centered"
    )

    st.title("⚡ Pokemon Type Classification")
    st.write("Simple KNN application using Streamlit.")

    df = load_data()
    model, accuracy = train_model(df)

    st.write("Input Pokemon stats:")

    hp = st.slider(
        "HP",
        int(df["hp"].min()),
        int(df["hp"].max()),
        int(df["hp"].mean())
    )

    attack = st.slider(
        "Attack",
        int(df["attack"].min()),
        int(df["attack"].max()),
        int(df["attack"].mean())
    )

    defense = st.slider(
        "Defense",
        int(df["defense"].min()),
        int(df["defense"].max()),
        int(df["defense"].mean())
    )

    speed = st.slider(
        "Speed",
        int(df["speed"].min()),
        int(df["speed"].max()),
        int(df["speed"].mean())
    )

    if st.button("Predict"):
        input_data = pd.DataFrame(
            {
                "hp": [hp],
                "attack": [attack],
                "defense": [defense],
                "speed": [speed]
            }
        )

        prediction = model.predict(input_data)[0]

        st.success(f"Prediction: **{prediction}**")
        st.info(f"Model Accuracy: **{accuracy * 100:.2f}%**")

    if st.checkbox("Show Pokemon Dataset"):
        st.dataframe(df)


if __name__ == "__main__":
    main()
