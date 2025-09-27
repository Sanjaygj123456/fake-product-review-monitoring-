import streamlit as st
import pandas as pd
import joblib

# Load model & vectorizer
clf = joblib.load("fake_review_model/logreg_model.joblib")
vectorizer = joblib.load("fake_review_model/tfidf_vectorizer.joblib")

st.title("🕵️ Fake Product Review Monitoring System")
st.write("Upload customer reviews and detect which ones are **fake**.")

# --- Option 1: Paste a single review ---
st.subheader("🔍 Check a Single Review")
user_input = st.text_area("Enter a product review:")
if st.button("Analyze Review"):
    if user_input.strip():
        X = vectorizer.transform([user_input])
        proba = clf.predict_proba(X)[0,1]
        label = "FAKE" if proba > 0.6 else "GENUINE"
        st.write(f"**Prediction:** {label} (probability = {proba:.2f})")
    else:
        st.warning("Please enter a review first.")

# --- Option 2: Upload CSV of reviews ---
st.subheader("📂 Bulk Review Analysis (CSV Upload)")
uploaded_file = st.file_uploader("Upload a CSV file with a column named 'review_text'", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "review_text" not in df.columns:
        st.error("CSV must contain a 'review_text' column.")
    else:
        X = vectorizer.transform(df["review_text"])
        df["pred_proba"] = clf.predict_proba(X)[:,1]
        df["pred_label"] = df["pred_proba"].apply(lambda x: "FAKE" if x > 0.6 else "GENUINE")
        st.success("Predictions completed ✅")
        st.dataframe(df[["review_text", "pred_label", "pred_proba"]])

        # Download button
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results as CSV", csv, "predictions.csv", "text/csv")
