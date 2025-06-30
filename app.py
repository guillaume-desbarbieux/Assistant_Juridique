import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import time

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={"device": "cpu"}
    )

@st.cache_resource(show_spinner=False)
def get_chroma(_embeddings):
    return Chroma(persist_directory="./db", embedding_function=_embeddings)



st.set_page_config(page_title="Assistant Juridique IA", layout="wide")
st.title("📚 Assistant Juridique avec IA")

st.sidebar.markdown("🧠 **Modèle d'embedding :** `paraphrase-multilingual-mpnet-base-v2`")
st.sidebar.markdown("🗂️ **Base vectorielle :** `Chroma`")
st.sidebar.markdown("💬 **Modèle LLM :** `mistral:latest` via Ollama")


st.sidebar.header("🔧 Paramètres avancés")

max_docs = st.sidebar.slider(
    "Nombre maximal de documents à utiliser",
    min_value=1,
    max_value=20,
    value=5,
    step=1
)

# Slider de pertinence exprimé en % (0 = tout passe, 100 = très strict)
similarity_threshold = st.sidebar.slider(
    "Seuil de pertinence (%)",
    min_value=0,
    max_value=100,
    value=80,
    step=1
)


# Choix multi-bases avec checkbox
st.sidebar.markdown("**Bases de documents à interroger :**")
base_options = [
    ("Archive mails", "archives_mails"),
    ("Textes de loi", "textes_loi"),
    ("Jurisprudence", "jurisprudence")
]
selected_bases = [
    key for label, key in base_options if st.sidebar.checkbox(label, value=True)
]

# Vérification qu'au moins une base est sélectionnée
if not selected_bases:
    st.sidebar.warning("⚠️ Veuillez sélectionner au moins une base de documents pour continuer.")
    st.stop()

# Champ de saisie utilisateur
user_input = st.text_area("✉️ Votre question :", height=150)
user_prompt_intro = st.text_area(
    "Début du prompt (modifiable)",
    value="""Tu es un assistant juridique expert.
    Tu dois faciliter le travail des juristes en présentant les documents qui peuvent leur être utile pour répondre.
    Tu dois répondre en français, de manière claire et précise.
    Base ta réponse uniquement sur les CONTEXTES ci-dessous.
    Si tu n'as pas de CONTEXTE, indique-le clairement et refuse de répondre.
    Ne fais aucune supposition et ne génère pas d'information non présente dans les CONTEXTES.""",
    height=150,
    key="prompt_intro"
)

def distance_to_percent(score, max_dist=10.0):
    """
    Convertit une distance en score de pertinence (%) inversée
    en supposant que les distances sont entre 0 et max_dist
    """
    score = max(0, min(score, max_dist))  # clamp
    return round((1 - score / max_dist) * 100)

if st.button("📤 Envoyer") and user_input.strip():
    def get_base_key(meta):
        val = str(meta.get("source", "")).lower()
        if "archives_mails" in val:
            return "archives_mails"
        if "textes_loi" in val:
            return "textes_loi"
        if "jurisprudence" in val:
            return "jurisprudence"
        return os.path.basename(val).replace(".txt", "")

    # Spinner pour chargement des embeddings
    t0 = time.time()
    with st.spinner("Chargement des embeddings HuggingFace..."):
        embeddings = get_embeddings()
    st.success(f"✅ Embeddings chargés ({time.time()-t0:.2f}s)")

    t0 = time.time()
    with st.spinner("Connexion à la base vectorielle Chroma..."):
        db = get_chroma(embeddings)
    st.success(f"✅ Base Chroma connectée ({time.time()-t0:.2f}s)")

    t0 = time.time()
    with st.spinner("Préparation du moteur de recherche sémantique..."):
        retriever = db.as_retriever(search_kwargs={"k": max_docs})
    st.success(f"✅ Moteur de recherche prêt ({time.time()-t0:.2f}s)")

    t0 = time.time()
    with st.spinner("Recherche des documents les plus proches dans la base..."):
        docs_and_scores = retriever.vectorstore.similarity_search_with_score(user_input, k=30)
    st.success(f"✅ Recherche vectorielle terminée ({time.time()-t0:.2f}s)")

    t0 = time.time()
    with st.spinner("Filtrage des documents selon les bases sélectionnées..."):
        docs_and_scores = [
            (doc, score) for doc, score in docs_and_scores
            if get_base_key(doc.metadata) in selected_bases
        ][:max_docs]
    st.success(f"✅ Filtrage par base terminé ({time.time()-t0:.2f}s)")

    t0 = time.time()
    with st.spinner("Calcul des pertinences et sélection des documents pertinents..."):
        docs_scores_pertinences = [
            (doc, score, distance_to_percent(score, max_dist=10.0))
            for doc, score in docs_and_scores
        ]
        max_dist = 10.0
        distance_seuil = max_dist * (1 - similarity_threshold / 100)
        filtered_docs = [
            (doc, score, pertinence)
            for doc, score, pertinence in docs_scores_pertinences
            if pertinence >= similarity_threshold
        ]
    st.success(f"✅ Pertinences calculées et documents sélectionnés ({time.time()-t0:.2f}s)")

    # 2. Affichage des documents pertinents
    st.subheader("📎 Documents pertinents trouvés")
    if not filtered_docs:
        # Calcul de la meilleure pertinence trouvée
        best_pertinence = max((p for _, _, p in docs_scores_pertinences), default=None)
        st.warning("❗ Aucun document suffisamment pertinent trouvé pour cette question.")
        st.info("L'assistant ne peut pas formuler de réponse fiable sans documents de référence.")
        if best_pertinence is not None:
            st.info(f"💡 Astuce : La meilleure pertinence trouvée est {best_pertinence}%. Essayez de baisser le seuil de pertinence dans les paramètres avancés pour augmenter vos chances de trouver des documents pertinents.")
        else:
            st.info("💡 Astuce : Essayez de baisser le seuil de pertinence dans les paramètres avancés pour augmenter vos chances de trouver des documents pertinents.")
        st.stop()
    else:
        for idx, (doc, score, pertinence) in enumerate(filtered_docs, 1):
            source = os.path.basename(doc.metadata.get('source', 'inconnu'))
            with st.expander(f"📄 Document {idx} — {source} (🔍 Pertinence : {pertinence}%)", expanded=False):
                st.markdown(
                    f"""
                    <div style=\"white-space: pre-wrap; word-wrap: break-word; overflow-x: hidden; background-color: #f9f9f9; padding: 1em; border-radius: 8px; border: 1px solid #ddd;\">
                        {doc.page_content}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Affichage debug : tous les documents trouvés avec leur score brut, leur pertinence et leur contenu
        st.subheader("🛠️ Debug : Tous les documents trouvés (pertinents et non pertinents)")
        if docs_scores_pertinences:
            for idx, (doc, score, pertinence) in enumerate(docs_scores_pertinences, 1):
                source = os.path.basename(str(doc.metadata.get('source', 'inconnu')))
                pertinent = "✅" if (doc, score, pertinence) in filtered_docs else "❌"
                with st.expander(f"{pertinent} Document {idx} — {source} | score brut = {score:.4f}, pertinence = {pertinence}%", expanded=False):
                    st.markdown(
                        f"""
                        <div style=\"white-space: pre-wrap; word-wrap: break-word; overflow-x: hidden; background-color: #f9f9f9; padding: 1em; border-radius: 8px; border: 1px solid #ddd;\">
                            {doc.page_content}
                        </div>
                        <hr/>
                        <b>Métadonnées brutes :</b>
                        <pre>{doc.metadata}</pre>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.info("Aucun document trouvé par la recherche, même avant filtrage.")

        # 3. Génération automatique de la réponse
        t0 = time.time()
        with st.spinner("Génération de la réponse..."):
            model_name = "mistral:latest"
            base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
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
            prompt_template = f"""
{{user_prompt_intro}}

CONTEXTES :
{{context}}

QUESTION :
{{question}}

RÉPONSE EN FRANÇAIS :
"""
            prompt = PromptTemplate(
                input_variables=["context", "question", "user_prompt_intro"],
                template=prompt_template
            )
            qa_chain = LLMChain(llm=oai, prompt=prompt)
            context_text = "\n\n".join([
                f"[Pertinence : {pertinence}%] {doc.page_content}"
                for doc, score, pertinence in filtered_docs
            ])
            try:
                result = qa_chain.run({
                    "context": context_text,
                    "question": user_input,
                    "user_prompt_intro": user_prompt_intro.strip()
                })
                st.subheader("✅ Réponse générée")
                st.write(result)
            except Exception as e:
                st.error(f"Erreur lors de la génération de la réponse : {e}")
                st.stop()
        st.success(f"✅ Réponse générée par le LLM ({time.time()-t0:.2f}s)")
