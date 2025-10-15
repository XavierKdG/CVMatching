import streamlit as st
import subprocess

st.title("🧠 CV–Vacature Matching Tool")

st.sidebar.header("Kies een stap:")
choice = st.sidebar.radio(
    "Wat wil je doen?",
    ["Label data", "Preprocess data", "Train TF-IDF model"]
)

if choice == "Label data":
    st.write("🔹 Dit maakt en balanceert de gelabelde dataset.")
    if st.button("▶️ Start labeling"):
        subprocess.run(["python", "labeling.py"])
        st.success("✅ Labeling en balancing voltooid!")

elif choice == "Preprocess data":
    st.write("🔹 Dit maakt de tekst schoon (lowercase, stopwords verwijderen, etc.)")
    if st.button("▶️ Start preprocessing"):
        subprocess.run(["python", "preprocessing.py"])
        st.success("✅ Preprocessing voltooid!")

elif choice == "Train TF-IDF model":
    st.write("🔹 Dit traint het model met TF-IDF en RandomForest of Neural Network.")
    if st.button("▶️ Start training"):
        subprocess.run(["python", "train_model.py"])
        st.success("✅ Modeltraining voltooid!")
