import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

def build_vectorstore():
    # Définir un chemin absolu fiable pour la persistance
    persist_directory = "/app/db"

    # Crée le dossier s'il n'existe pas
    os.makedirs(persist_directory, exist_ok=True)

    print(f"📁 Persist directory : {persist_directory}")
    print(f"📂 Contenu de /app : {os.listdir('/app')}")

    # Chargement des fichiers dans le dossier 'data'
    loader = DirectoryLoader("/app/data", glob="**/*.txt", show_progress=True)
    documents = loader.load()

    # Split les documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Création ou mise à jour de la base vectorielle
    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    vectordb.persist()
    print("✅ Base vectorielle créée et persistée avec succès.")

if __name__ == "__main__":
    try:
        build_vectorstore()
    except Exception as e:
        print(f"❌ Une erreur est survenue : {e}")
