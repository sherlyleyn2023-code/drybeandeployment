import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.title("Dry Bean Classification (Random Forest)")

# Upload file Excel
uploaded_file = st.file_uploader("Upload Dry_Bean_Dataset.xlsx", type="xlsx")
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=500, random_state=123)
    rf.fit(X_scaled, y)

    st.write("Masukkan nilai fitur untuk prediksi:")
    input_data = {}
    for col in X.columns:
        input_data[col] = st.number_input(col, float(df[col].median()))
    
    if st.button("Prediksi"):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        pred_class = rf.predict(input_scaled)[0]
        pred_proba = rf.predict_proba(input_scaled)
        st.write(f"**Prediksi kelas:** {pred_class}")
        st.dataframe(pd.DataFrame(pred_proba, columns=rf.classes_))
