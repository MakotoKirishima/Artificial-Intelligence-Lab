import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(
    page_title="Klasifikasi Iris dengan Streamlit",
    page_icon="🌸",
    layout="centered",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load Iris dataset directly from scikit-learn.

    This avoids local CSV/file path errors and makes the app easier to deploy
    on Streamlit Community Cloud through GitHub.
    """
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df = df.rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
            "target": "species",
        }
    )
    df["species"] = df["species"].map(dict(enumerate(iris.target_names)))
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    feature_columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]

    X = df[feature_columns]
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    return model, accuracy, feature_columns


def main() -> None:
    st.title("🌸 Aplikasi Klasifikasi Bunga Iris")
    st.write("Masukkan nilai fitur bunga iris untuk memprediksi spesiesnya.")

    df = load_data()
    model, accuracy, feature_columns = train_model(df)

    with st.form("prediction_form"):
        sepal_length = st.slider(
            "Sepal Length (cm)",
            min_value=float(df["sepal_length"].min()),
            max_value=float(df["sepal_length"].max()),
            value=float(df["sepal_length"].mean()),
        )
        sepal_width = st.slider(
            "Sepal Width (cm)",
            min_value=float(df["sepal_width"].min()),
            max_value=float(df["sepal_width"].max()),
            value=float(df["sepal_width"].mean()),
        )
        petal_length = st.slider(
            "Petal Length (cm)",
            min_value=float(df["petal_length"].min()),
            max_value=float(df["petal_length"].max()),
            value=float(df["petal_length"].mean()),
        )
        petal_width = st.slider(
            "Petal Width (cm)",
            min_value=float(df["petal_width"].min()),
            max_value=float(df["petal_width"].max()),
            value=float(df["petal_width"].mean()),
        )

        submitted = st.form_submit_button("Prediksi")

    if submitted:
        input_data = pd.DataFrame(
            [[sepal_length, sepal_width, petal_length, petal_width]],
            columns=feature_columns,
        )
        prediction = model.predict(input_data)[0]

        st.success(f"🌼 Prediksi jenis iris: **{prediction}**")
        st.info(f"Akurasi model pada data uji: **{accuracy * 100:.2f}%**")

    with st.expander("Tampilkan Dataset Iris"):
        st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
