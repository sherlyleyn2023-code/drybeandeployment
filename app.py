import pandas as pd
import xgboost as xgb
import streamlit as st

# --- Load dataset ---
df = pd.read_excel("Dry_Bean_Dataset.xlsx")

# Pisah fitur dan target
X = df.drop("Class", axis=1)
y = df["Class"]

# --- Train XGBoost model ---
model = xgb.XGBClassifier(
    n_estimators=500,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=123
)
model.fit(X, y)

# --- Streamlit UI ---
st.title("Dry Bean Classification (XGBoost)")
st.write("Masukkan nilai fitur untuk prediksi satu data point:")

# Input user
input_data = {col: st.number_input(col, float(df[col].median())) for col in X.columns}

if st.button("Prediksi"):
    input_df = pd.DataFrame([input_data])
    pred_class = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)

    st.write(f"**Prediksi kelas:** {pred_class}")
    st.write("**Probabilitas masing-masing kelas:**")
    st.dataframe(pd.DataFrame(pred_proba, columns=model.classes_))
