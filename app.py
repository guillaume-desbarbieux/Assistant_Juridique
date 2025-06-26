import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

st.set_page_config(page_title="Assistant Juridique IA", layout="wide")
st.title("📚 Assistant Juridique avec IA")

st.sidebar.header("🔧 Paramètres avancés")

max_docs = st.sidebar.slider(
    "Nombre maximal de documents à utiliser",
    min_value=1,
    max_value=20,
    value=5,
    step=1
)

similarity_threshold = st.sidebar.slider(
    "Seuil de pertinence (%)",
    min_value=0,
    max_value=500,
    value=70,
    step=5
)


st.write("Posez une question juridique en lien avec le droit du travail, la jurisprudence ou les clauses contractuelles.")

# Champ de saisie utilisateur
user_input = st.text_area("✉️ Votre question :", height=200)

if st.button("📤 Envoyer") and user_input.strip():
    with st.spinner("Recherche et génération de la réponse..."):

        # Charger la base vectorielle
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma(persist_directory="./db", embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": max_docs})

       # Récupération des documents pertinents avec score
        docs_and_scores = retriever.vectorstore.similarity_search_with_score(user_input, k=max_docs)

        # Optionnel : seuil de similarité à ajuster si besoin
        threshold_value = similarity_threshold / 100
        
        # On garde uniquement les documents avec une similarité suffisante
        filtered_docs = [(doc, score) for doc, score in docs_and_scores if score >= threshold_value]


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
Tu es un assistant juridique expert.
Tu dois faciliter le travail des juristes en présentant les documents qui peuvent leur être utile pour répondre.
Tu dois répondre en français, de manière claire et précise.
Base ta réponse uniquement sur les documents fournis.
Si tu n'as pas assez d'information, dis-le clairement.
Ne fais aucune supposition.

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
            st.info("L'assistant ne peut pas formuler de réponse fiable sans documents de référence.")
        else:
            try:
                context_text = "\n\n".join([
                    f"[Pertinence : {score:.2f}] {doc.page_content}"
                    for doc, score in docs_and_scores if score >= threshold_value
                ])

                result = qa_chain.run({"context": context_text, "question": user_input})
                st.subheader("✅ Réponse générée")
                st.write(result)
            except Exception as e:
                st.error(f"Erreur lors de la génération de la réponse : {e}")
                st.stop()

            st.subheader("📎 Documents utilisés")
            for idx, (doc, score) in enumerate(docs_and_scores, 1):
                if score >= threshold_value:
                    source = os.path.basename(doc.metadata.get('source', 'inconnu'))
                    st.markdown(f"### 📄 Document {idx} — {source} (🔍 Pertinence : {score})")
                    st.markdown(
                        f"""
                        <div style="white-space: pre-wrap; word-wrap: break-word; overflow-x: hidden; background-color: #f9f9f9; padding: 1em; border-radius: 8px; border: 1px solid #ddd;">
                            {doc.page_content}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

