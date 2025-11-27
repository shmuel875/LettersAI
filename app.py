import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import argostranslate.package
import argostranslate.translate
import warnings
import os

# -------------------------------
# SUPPRESS WARNINGS
# -------------------------------
warnings.filterwarnings("ignore")

# -------------------------------
# LOAD EMBEDDING MODEL
# -------------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

@st.cache_resource
def load_translators():
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()

    for pkg in available:
        if pkg.from_code == "he" and pkg.to_code == "en":
            pkg.install()
        if pkg.from_code == "yi" and pkg.to_code == "en":
            pkg.install()
    return True

model = load_embedding_model()
load_translators()

# -------------------------------
# CHROMA CLIENT + PERSISTENT COLLECTION
# -------------------------------
STORAGE_DIR = os.path.join(os.getcwd(), "chroma_db")  # folder for persistence
os.makedirs(STORAGE_DIR, exist_ok=True)

chroma_client = chromadb.Client(Settings(
    persist_directory=STORAGE_DIR
))

try:
    collection = chroma_client.get_collection("docs")
except:
    collection = chroma_client.create_collection("docs")

# -------------------------------
# FUNCTIONS
# -------------------------------
def embed(text):
    return model.encode(text).tolist()

# -------------------------------
# LANGUAGE DETECTION
# -------------------------------
def detect_language(text):
    """
    Simple script-based language detection:
    - If the text contains typical Yiddish letters (ײ, ױ, װ), mark as 'yi'
    - Otherwise, assume Hebrew 'he'
    """
    yiddish_markers = ["ײ", "ױ", "װ"]
    if any(char in text for char in yiddish_markers):
        return "yi"
    return "he"

def translate(text, lang):
    return argostranslate.translate.translate(text, lang, "en")

def index_document(doc_id, text, lang):
    vector = embed(text)
    collection.add(
        ids=[doc_id],
        embeddings=[vector],
        documents=[text],
        metadatas=[{"lang": lang}]
    )
    collection.persist()  # save changes

def search(query):
    qv = embed(query)
    results = collection.query(query_embeddings=[qv], n_results=1)
    if not results["documents"][0]:
        return "No documents found."
    doc = results["documents"][0][0]
    meta = results["metadatas"][0][0]
    lang = meta["lang"]
    translated = translate(doc, lang)
    return translated

def list_documents():
    docs = collection.get()
    result = []
    for i, (doc, meta) in enumerate(zip(docs["documents"], docs["metadatas"])):
        result.append({"id": docs["ids"][i], "lang": meta["lang"], "content_preview": doc[:50]})
    return result

def delete_documents(ids_to_delete):
    for doc_id in ids_to_delete:
        collection.delete(ids=[doc_id])
    collection.persist()

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("📄 AI Document Finder (Unified Upload)")
st.write("Upload Hebrew or Yiddish documents, ask English questions, manage indexed documents.")

# -------------------------------
# Unified uploader
# -------------------------------
st.subheader("Upload Documents")
uploaded_files = st.file_uploader(
    "Upload Hebrew or Yiddish documents (any number)", type=["txt"], accept_multiple_files=True
)

if st.button("Index Documents"):
    if uploaded_files:
        total_files = 0
        current_count = len(collection.get()['documents'])
        for i, file in enumerate(uploaded_files):
            text = file.read().decode("utf-8")
            lang = detect_language(text)
            index_document(f"doc_{current_count+i}", text, lang)
            total_files += 1
        st.success(f"{total_files} document(s) indexed successfully!")
    else:
        st.error("Please upload at least one document.")

# -------------------------------
# Document management
# -------------------------------
st.subheader("Currently Indexed Documents")
docs = list_documents()
if docs:
    doc_labels = [f"{d['id']} | {d['lang']} | {d['content_preview']}..." for d in docs]
    to_delete = st.multiselect("Select documents to delete", doc_labels)

    if st.button("Delete Selected Documents"):
        ids_to_delete = [d['id'] for d, label in zip(docs, doc_labels) if label in to_delete]
        if ids_to_delete:
            delete_documents(ids_to_delete)
            st.success(f"Deleted {len(ids_to_delete)} document(s).")
else:
    st.info("No documents indexed yet.")

# -------------------------------
# Search
# -------------------------------
st.subheader("Ask a question")
query = st.text_input("Your question in English:")

if st.button("Search"):
    if query.strip() == "":
        st.error("Please enter a question.")
    else:
        result = search(query)
        st.subheader("Answer")
        st.write(result)
