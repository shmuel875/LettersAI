import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
import argostranslate.package
import argostranslate.translate
import warnings
import re

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

    # Install Hebrew -> English
    for pkg in available:
        if pkg.from_code == "he" and pkg.to_code == "en":
            pkg.install()
    return True

model = load_embedding_model()
load_translators()

# -------------------------------
# CHROMA CLIENT + COLLECTION
# -------------------------------
chroma_client = chromadb.Client()
try:
    collection = chroma_client.get_collection("docs")
except:
    collection = chroma_client.create_collection("docs")

# -------------------------------
# FUNCTIONS
# -------------------------------
def embed(text):
    return model.encode(text).tolist()

def translate(text):
    return argostranslate.translate.translate(text, "he", "en")

def index_document(doc_id, text):
    vector = embed(text)
    collection.add(
        ids=[doc_id],
        embeddings=[vector],
        documents=[text],
        metadatas=[{"lang": "he"}]
    )
    # no collection.persist() needed

# -------------------------------
# Sentence splitting and snippet extraction
# -------------------------------
def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)

def get_context_snippet(text, query, window=2):
    sentences = split_sentences(text)
    for i, s in enumerate(sentences):
        if query.lower() in s.lower():
            start = max(0, i - window)
            end = min(len(sentences), i + window + 1)
            return " ".join(sentences[start:end])
    # fallback: first 4 sentences
    return " ".join(sentences[:4])

def search(query):
    qv = embed(query)
    results = collection.query(query_embeddings=[qv], n_results=1)
    if not results["documents"][0]:
        return "No documents found."
    doc = results["documents"][0][0]
    
    snippet = get_context_snippet(doc, query, window=2)
    translated = translate(snippet)
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

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("📄 AI Document Finder (Hebrew Only)")
st.write("Upload Hebrew documents, ask English questions, manage indexed documents.")

# -------------------------------
# Upload
# -------------------------------
st.subheader("Upload Documents")
uploaded_files = st.file_uploader(
    "Upload Hebrew documents (any number)", type=["txt"], accept_multiple_files=True
)

if st.button("Index Documents"):
    if uploaded_files:
        total_files = 0
        current_count = len(collection.get()['documents'])
        for i, file in enumerate(uploaded_files):
            text = file.read().decode("utf-8")
            index_document(f"doc_{current_count+i}", text)
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
