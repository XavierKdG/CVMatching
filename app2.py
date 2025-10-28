import streamlit as st
import subprocess

# --- Pagina-instellingen ---
st.set_page_config(page_title="CV–Vacature Matching Tool", layout="centered")

# --- Titel ---
st.title("CV–Vacature Matching Tool")
st.markdown("---")

# --- Sidebar navigatie ---
st.sidebar.header("Navigatie")
choice = st.sidebar.radio(
    "Kies een stap:",
    ["Label data", "Preprocess data", "Train TF-IDF model"]
)

# --- Label data ---
if choice == "Label data":
    st.subheader("Stap 1: Label data")
    st.write("Dit script maakt en balanceert de gelabelde dataset.")
    
    if st.button("Start labeling"):
        with st.spinner("Bezig met labeling... even geduld."):
            result = subprocess.run(["python", "labeling.py"], capture_output=True, text=True)
        
        if result.returncode == 0:
            st.success("Labeling en balancing voltooid.")
        else:
            st.error("Er is een fout opgetreden bij labeling:")
            st.code(result.stderr)

# --- Preprocess data ---
elif choice == "Preprocess data":
    st.subheader("Stap 2: Preprocess data")
    st.write("Dit script maakt de tekst schoon (lowercase, stopwords verwijderen, enzovoort).")
    
    if st.button("Start preprocessing"):
        with st.spinner("Bezig met preprocessing... even geduld."):
            result = subprocess.run(["python", "preprocessing.py"], capture_output=True, text=True)
        
        if result.returncode == 0:
            st.success("Preprocessing voltooid.")
        else:
            st.error("Er is een fout opgetreden bij preprocessing:")
            st.code(result.stderr)

# --- Train model ---
elif choice == "Train TF-IDF model":
    st.subheader("Stap 3: Train TF-IDF model")
    st.write("Dit script traint het model met TF-IDF en een RandomForest of een neuraal netwerk.")
    
    if st.button("Start modeltraining"):
        with st.spinner("Bezig met modeltraining... even geduld."):
            result = subprocess.run(["python", "train_model.py"], capture_output=True, text=True)
        
        if result.returncode == 0:
            st.success("Modeltraining voltooid.")
        else:
            st.error("Er is een fout opgetreden bij modeltraining:")
            st.code(result.stderr)
