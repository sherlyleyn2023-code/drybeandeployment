# app.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.title("Dry Bean Classification (Random Forest)")

# --- Upload dataset ---
uploaded_file = st.file_uploader("Upload Dry Bean Excel (.xlsx)", type=["xlsx"])
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Gagal membaca file Excel: {e}")
        st.stop()

    st.success("Dataset berhasil di-upload!")
    st.write("Preview data:")
    st.dataframe(df.head())

    # --- Pisah fitur & target ---
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # --- Split train/test & scaling ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=123
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    rf = RandomForestClassifier(n_estimators=500, random_state=123)
    rf.fit(X_train_scaled, y_train)

    st.write("Masukkan nilai fitur untuk prediksi satu data point:")
    input_data = {}
    for col in X.columns:
        val = st.number_input(col, float(df[col].median()))
        input_data[col] = val

    if st.button("Prediksi"):
        input_scaled = scaler.transform(pd.DataFrame([input_data]))
        pred_class = rf.predict(input_scaled)[0]
        pred_proba = rf.predict_proba(input_scaled)
        st.write(f"**Prediksi kelas:** {pred_class}")
        st.write("**Probabilitas masing-masing kelas:**")
        st.dataframe(pd.DataFrame(pred_proba, columns=rf.classes_))
else:
    st.info("Silakan upload file Dry Bean Excel (.xlsx) untuk memulai.")
