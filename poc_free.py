import warnings
warnings.filterwarnings("ignore")  # suppress Stanza warnings

from sentence_transformers import SentenceTransformer
import chromadb
import argostranslate.package
import argostranslate.translate


# ============================================
#             LOAD MODELS
# ============================================
print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# Local vector DB
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="documents_free_poc")


# ============================================
#      DOWNLOAD TRANSLATION MODELS (ONCE)
# ============================================
print("Downloading translation models... (first run only)")

argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()

# Hebrew → English
for pkg in available_packages:
    if pkg.from_code == "he" and pkg.to_code == "en":
        print("Installing Hebrew → English...")
        argostranslate.package.install_from_path(pkg.download())

# Yiddish → English
for pkg in available_packages:
    if pkg.from_code == "yi" and pkg.to_code == "en":
        print("Installing Yiddish → English...")
        argostranslate.package.install_from_path(pkg.download())


# ============================================
#            TRANSLATION FUNCTION
# ============================================
def translate_to_english(text, lang):
    return argostranslate.translate.translate(text, lang, "en")


# ============================================
#            EMBEDDING FUNCTION
# ============================================
def embed(text):
    return model.encode(text).tolist()


# ============================================
#             INDEX DOCUMENTS
# ============================================
def index_document(doc_id, filepath, lang):
    """lang must be 'he' or 'yi'"""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    vector = embed(text)

    collection.add(
        ids=[doc_id],
        embeddings=[vector],
        documents=[text],
        metadatas=[{"filename": filepath, "lang": lang}],
    )

    print(f"Indexed {filepath} as language {lang}")


# ============================================
#               SEARCH FUNCTION
# ============================================
def search(query, n=1):
    query_vec = embed(query)

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=n
    )

    print("\n=== SEARCH RESULTS ===\n")

    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):

        filename = meta["filename"]
        lang = meta["lang"]

        print(f"Matched document: {filename} (Language: {lang})")

        # Translate to English only
        translated = translate_to_english(doc, lang)

        print("\n--- English Translation ---")
        print(translated[:1000], "\n")  # more space for bigger docs


# ============================================
#                   MAIN
# ============================================
if __name__ == "__main__":
    print("\nIndexing documents...\n")

    # IMPORTANT: specify the correct language
    index_document("doc1", "doc1.txt", "he")  # Hebrew document
    index_document("doc2", "doc2.txt", "yi")  # Yiddish document

    print("\nReady! Ask questions.\n")

    while True:
        query = input("Ask a question in English (or 'exit'): ").strip()
        if query.lower() == "exit":
            break
        search(query)
