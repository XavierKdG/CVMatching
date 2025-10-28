import sys
from pathlib import Path
import streamlit as st
from qdrant_client import QdrantClient
import PyPDF2
import spacy
from spacy import displacy
from sentence_transformers import SentenceTransformer

# --- 1. ADD PROJECT ROOT TO PATH ---
# This allows us to import from the 'src' folder
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT.parent))

# --- 2. PAGE CONFIG ---
st.set_page_config(page_title="CV Matcher", page_icon="📄", layout="wide")

# --- 3. MODEL & CLIENT LOADING ---
# Use Streamlit's cache to load models only once
@st.cache_resource
def load_models():
    """Loads the spaCy NER and SentenceTransformer models."""
    ner_model_path = PROJECT_ROOT.parent / "models/model-best"
    
    try:
        ner_model = spacy.load(ner_model_path)
    except Exception as e:
        st.error(f"Error loading NER model from {ner_model_path}: {e}")
        st.error("Please make sure you have trained a model and it exists at 'models/model-best'.")
        st.stop()
        
    # This is the model used for creating embeddings for search
    # Replace with the *exact* model name you used in 'src/upload_to_qdrant.py'
    similarity_model_name = 'all-MiniLM-L6-v2' 
    similarity_model = SentenceTransformer(similarity_model_name)
    
    return ner_model, similarity_model

@st.cache_resource
def get_qdrant_client():
    """Initializes and returns the Qdrant client."""
    return QdrantClient(url="http://localhost:6333")

NER_MODEL, SIMILARITY_MODEL = load_models()
QDRANT_CLIENT = get_qdrant_client()

# --- 4. CONFIGURATION ---
# *** IMPORTANT ***
# Replace with the collection name you defined in your config.yml 
# for 'upload_to_qdrant.py' (e.g., 'job_descriptions')
COLLECTION_NAME = "job_collection"

# --- 5. HELPER FUNCTIONS ---

def read_pdf(file):
    """Extracts text from an uploaded PDF file."""
    try:
        pdf = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def infer_vector(text):
    """Creates a vector embedding from text using the SentenceTransformer."""
    return SIMILARITY_MODEL.encode(text)

def render_entities(doc, title):
    """Renders named entities using spaCy's displacy in a nice box."""
    html = displacy.render(doc, style="ent", page=False)
    # Add some styling to make it look nice in Streamlit
    html = html.replace("\n", " ")
    st.subheader(title)
    st.write(
        f'<div style="border: 1px solid #e6e9ef; border-radius: 5px; padding: 10px; margin-bottom: 20px;">{html}</div>',
        unsafe_allow_html=True
    )

# --- 6. UI LAYOUT ---
st.title("📄 CV & Job Matcher")
st.write("Upload your CV to find matching jobs, or paste a job description to see its extracted skills and titles.")

col1, col2 = st.columns([0.6, 0.4])

with col1:
    st.header("Find Matching Jobs")
    uploaded_file = st.file_uploader("1. Upload Your CV", type=["pdf", "txt"])
    
    st.header("Analyze a Job Posting")
    job_text_input = st.text_area("2. Paste a Job Description (Optional)", height=200)

with col2:
    st.header("Analysis & Results")
    
    # --- 7. LOGIC & PROCESSING ---

    # --- Part A: Process Job Description Input ---
    if job_text_input:
        with st.container(border=True):
            job_doc = NER_MODEL(job_text_input)
            render_entities(job_doc, "Job Description Entities")

    # --- Part B: Process CV Upload and Find Matches ---
    if uploaded_file is not None:
        with st.container(border=True):
            # 1. Extract CV Text
            if uploaded_file.type == "application/pdf":
                cv_text = read_pdf(uploaded_file)
            else:
                cv_text = str(uploaded_file.read(), "utf-8")
            
            if not cv_text:
                st.error("Could not extract text from your CV.")
                st.stop()

            # 2. Display CV Entities
            cv_doc = NER_MODEL(cv_text)
            render_entities(cv_doc, "Your CV Entities")

            # 3. Find and Display Matches
            st.subheader(f"Top 10 Matches from Qdrant ({COLLECTION_NAME})")
            with st.spinner("Processing CV and searching for matches..."):
                try:
                    # 3a. Create the vector
                    vector = infer_vector(cv_text)

                    # 3b. Search Qdrant
                    search_results = QDRANT_CLIENT.search(
                        collection_name=COLLECTION_NAME,
                        query_vector=vector.tolist(),
                        limit=10
                    )
                    
                    # 3c. Display results
                    if not search_results:
                        st.write("No matches found.")
                    
                    for result in search_results:
                        payload = result.payload or {}
                        with st.expander(f"**{payload.get('Business Title', 'N/A')}** | Score: {round(result.score * 100, 2)}%"):
                            st.write(f"**Organization:** {payload.get('Agency', 'N/A')}")
                            # Display a snippet of the job description
                            description = payload.get('job_description', 'No description available.')
                            st.write(f"**Description Snippet:** {description[:300]}...")
                
                except Exception as e:
                    st.error(f"Error connecting to or searching Qdrant: {e}")
                    st.info("Is your Qdrant server running? `docker run -p 6333:6333 qdrant/qdrant`")