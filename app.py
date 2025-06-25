import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

st.set_page_config(page_title="Assistant Juridique IA", layout="wide")
st.title("📚 Assistant Juridique avec IA")
st.write("Posez une question juridique en lien avec le droit du travail, la jurisprudence ou les clauses contractuelles.")

# Champ de saisie utilisateur
user_input = st.text_area("✉️ Votre question :", height=200)

if st.button("📤 Envoyer") and user_input.strip():
    with st.spinner("Recherche et génération de la réponse..."):

        # Charger la base vectorielle
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma(persist_directory="./db", embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 5})

       # Récupération des documents pertinents avec score
        docs_and_scores = retriever.vectorstore.similarity_search_with_score(user_input, k=5)

        # Optionnel : seuil de similarité à ajuster si besoin
        SIMILARITY_THRESHOLD = 0.7

        # On garde uniquement les documents avec une similarité suffisante
        filtered_docs = [doc for doc, score in docs_and_scores if score >= SIMILARITY_THRESHOLD]
        context_text = "\n\n".join([doc.page_content for doc in filtered_docs])


               # LLM via Ollama
        model_name = "mistral:latest"
        base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

        # Vérification Ollama accessible
        import requests

        def check_ollama_is_alive():
            try:
                r = requests.get(f"{base_url}/api/generate")
                if r.status_code in [404, 405]:
                    return True
                else:
                    st.error(f"Ollama ne répond pas correctement (code {r.status_code})")
                    st.stop()
            except Exception as e:
                st.error(f"Ollama semble injoignable : {e}")
                st.stop()
        check_ollama_is_alive()

        oai = Ollama(model=model_name, base_url=base_url)
        
           # Création du prompt personnalisé
        prompt_template = """
Tu es un assistant juridique expert. Tu dois répondre en français, de manière claire et précise.
Base ta réponse uniquement sur les documents fournis ci-dessous.
Ne fais pas de suppositions en dehors des documents.

Documents:
{context}

Question:
{question}

Réponse en français :
"""
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template
        )

        # Création de la chaîne LLMChain avec le prompt
        qa_chain = LLMChain(llm=oai, prompt=prompt)

        if not filtered_docs:
            st.warning("❗ Aucun document suffisamment pertinent trouvé pour cette question.")
        else:
            # On lance la chaîne QA avec les documents filtrés
            try:
                result = qa_chain.run({"context": context_text, "question": user_input})
                st.subheader("✅ Réponse générée")
                st.write(result)
            except Exception as e:
                st.error(f"Erreur lors de la génération de la réponse : {e}")
                st.stop()

            st.subheader("📎 Sources utilisées")

            if not docs:
                st.warning("Aucun document pertinent trouvé pour cette question.")

            for doc in filtered_docs:
                st.markdown(f"- **{os.path.basename(doc.metadata.get('source', ''))}**")