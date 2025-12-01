import streamlit as st
from sentence_transformers import SentenceTransformer
import argostranslate.package
import argostranslate.translate
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# Load models
# -------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

@st.cache_resource
def load_translator():
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()
    for pkg in available:
        if pkg.from_code == "he" and pkg.to_code == "en":
            pkg.install()
    return True

model = load_embedding_model()
load_translator()

# -------------------------------
# In-memory document store
# -------------------------------
documents = []  # each item: {"filename": ..., "paragraphs": [...], "embeddings": [...]}

# -------------------------------
# Utility functions
# -------------------------------

def split_paragraphs(text):
    # Split on double newlines and strip whitespace
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs

def translate_paragraph(text):
    return argostranslate.translate.translate(text, "he", "en")

def embed_paragraphs(paragraphs):
    return model.encode(paragraphs, convert_to_numpy=True)

def find_most_relevant_paragraph(query):
    if not documents:
        return None, None, None
    query_vec = model.encode([query], convert_to_numpy=True)
    
    best_score = -1
    best_para = ""
    best_doc_name = ""
    
    for doc in documents:
        sims = cosine_similarity(query_vec, doc["embeddings"])[0]
        # Boost paragraphs containing query keywords
        boosted_sims = []
        for p, sim in zip(doc["paragraphs"], sims):
            keyword_boost = 0.2 if any(word.lower() in p.lower() for word in query.lower().split()) else 0
            boosted_sims.append(sim + keyword_boost)
        idx = np.argmax(boosted_sims)
        if boosted_sims[idx] > best_score:
            best_score = boosted_sims[idx]
            best_para = doc["paragraphs"][idx]
            best_doc_name = doc["filename"]
    
    if best_score < 0.3:
        return None, None, None
    
    # Optionally, include surrounding paragraphs
    doc_paras = next(d for d in documents if d["filename"] == best_doc_name)["paragraphs"]
    idx = doc_paras.index(best_para)
    surrounding = doc_paras[max(0, idx-1): idx+2]  # previous + current + next
    return best_doc_name, "\n\n".join(surrounding), best_score

# -------------------------------
# Streamlit UI
# -------------------------------

st.title("📄 AI Hebrew Document Search (English Output)")

st.subheader("Upload Hebrew Documents")
uploaded_files = st.file_uploader(
    "Upload any number of Hebrew .txt files", type=["txt"], accept_multiple_files=True
)

if uploaded_files:
    documents.clear()  # reset for fresh upload
    for file in uploaded_files:
        text = file.read().decode("utf-8")
        paragraphs = split_paragraphs(text)
        english_paragraphs = [translate_paragraph(p) for p in paragraphs]
        embeddings = embed_paragraphs(english_paragraphs)
        documents.append({
            "filename": file.name,
            "paragraphs": english_paragraphs,
            "embeddings": embeddings
        })
    st.success(f"{len(uploaded_files)} document(s) uploaded and indexed!")

st.subheader("Ask a question (in English)")
query = st.text_input("Enter your question:")

if st.button("Search"):
    if not query.strip():
        st.error("Please enter a question.")
    else:
        doc_name, paragraph_text, score = find_most_relevant_paragraph(query)
        if paragraph_text is None:
            st.info("No relevant documents found.")
        else:
            st.subheader(f"Relevant Document: {doc_name} (score {score:.3f})")
            st.text_area("Relevant English Paragraph(s):", paragraph_text, height=400)
