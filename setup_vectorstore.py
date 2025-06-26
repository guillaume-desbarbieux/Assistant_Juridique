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
    MIN_LEN_NO_SPLIT = 1000

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=120,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    print(f"📄 Nombre de documents chargés : {len(documents)}")

    chunks = []
    for doc in documents:
        if len(doc.page_content) < MIN_LEN_NO_SPLIT:
            chunks.append(doc)
        else:
            chunks.extend(splitter.split_documents([doc]))

    print(f"📚 Nombre de chunks créés : {len(chunks)}")

    # Embeddings
    print("🔍 Création des embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    # Création ou mise à jour de la base vectorielle
    print("🔄 Création ou mise à jour de la base vectorielle...")

    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print("📦 Base vectorielle créée avec succès.")

    # Persist la base vectorielle
    print("💾 Persistance de la base vectorielle en cours...")
    vectordb.persist()
    vectordb = None  # Libération de la mémoire
    print("💾 Base vectorielle persistée avec succès.")

    # Vérification du contenu de la base vectorielle
    print("🔍 Vérification du contenu de la base vectorielle...")
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    print(f"📊 Nombre de documents dans la base vectorielle : {len(vectordb)}")
    print("✅ Base vectorielle prête à l'emploi.")

    # Nettoyage
    print("🧹 Nettoyage de la mémoire en cours...")
    vectordb = None  # Libération de la mémoire
    print("🧹 Nettoyage de la mémoire terminé.")
    print("✅ Processus de création de la base vectorielle terminé avec succès.")

    
if __name__ == "__main__":
    try:
        build_vectorstore()
    except Exception as e:
        print(f"❌ Une erreur est survenue : {e}")
