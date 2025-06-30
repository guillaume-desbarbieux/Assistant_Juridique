import os
from dotenv import load_dotenv
load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

import streamlit as st
from langchain_chroma import Chroma
from utils.load_embeddings import get_local_embeddings
import requests
import datetime

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer
import torch
import streamlit as st

import datetime
from fpdf import FPDF


@st.cache_resource
def load_local_model(model_id):
    if model_id == "plguillou/t5-base-fr-sum-cnndm":
        tokenizer = T5Tokenizer.from_pretrained(model_id)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return tokenizer, model

# Pré-chargement des deux modèles
flan_tokenizer, flan_model = load_local_model("google/flan-t5-small")
plg_tokenizer, plg_model = load_local_model("plguillou/t5-base-fr-sum-cnndm")


def generate_response(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Si le tag [RESPONSE] n'est pas généré, on affiche tout
    if "[RESPONSE]" in text:
        text = text.split("[RESPONSE]", 1)[-1].strip()
    # Si la réponse est vide, on affiche la sortie brute
    if not text.strip():
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

st.set_page_config(page_title="Assistant Juridique IA", layout="wide")
st.title("📚 Assistant Juridique avec IA")
st.write("Posez une question juridique.")

# Réorganisation de la sidebar : paramètres avancés en haut
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
    max_value=200,
    value=90,
    step=5
)

# Choix multi-bases avec checkbox
st.sidebar.markdown("**Bases de documents à interroger :**")
base_options = [
    ("Archive mails", "archive_mail", "archives_mails"),
    ("Textes de loi", "textes_loi", "textes_loi"),
    ("Jurisprudence", "jurisprudence", "jurisprudence")
]
selected_bases = [
    key for label, key, _ in base_options if st.sidebar.checkbox(label, value=True)
]

# Vérification qu'au moins une base est sélectionnée
if not selected_bases:
    st.sidebar.warning("⚠️ Veuillez sélectionner au moins une base de documents pour continuer.")
    st.stop()  

# Affichage des modèles utilisés (en bas de la sidebar)
st.sidebar.markdown("---")
st.sidebar.markdown("🧠 **Modèle d'embedding :** `paraphrase-multilingual-mpnet-base-v2`")
st.sidebar.markdown("🗂️ **Base vectorielle :** `Chroma`")
st.sidebar.markdown("💬 **Modèle LLM :** `google/flan-t5-small` (text-generation, multilingue, open source)")

# Saisie de l'utilisateur et personnalisation du prompt en même temps
col1, col2 = st.columns([2, 3])
with col1:
    user_input = st.text_area("✉️ Votre question :", height=200, key="user_question")
with col2:
    user_prompt_intro = st.text_area(
        "Début du prompt (modifiable)",
        value="Vous êtes un assistant juridique spécialisé en droit français.\nVotre tâche est de proposer une réponse synthétique et argumentée à la question suivante, en vous appuyant uniquement sur les extraits de documents fournis, classés par pertinence. Indiquez clairement si la réponse est incertaine ou partielle. Répondez en français.",
        height=120,
        key="prompt_intro"
    )

# Bouton d'envoi de la question
if st.button("📤 Envoyer") and user_input.strip():
    user_input = st.session_state["user_question"]
    user_prompt_intro = st.session_state["prompt_intro"]
    def distance_to_percent(score, max_dist=10.0):
        score = max(0, min(score, max_dist))
        return round((1 - score / max_dist) * 100)

    with st.spinner("Recherche des documents pertinents..."):
        embeddings = get_local_embeddings()
        db_path = os.path.abspath("./db")
        db = Chroma(persist_directory=db_path, embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": max_docs})
        docs_and_scores = [
            (doc, score)
            for doc, score in retriever.vectorstore.similarity_search_with_score(user_input, k=30)
            if doc.metadata.get("source") in selected_bases
        ][:max_docs]
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

    # Affichage des documents pertinents (dropdown fermé par défaut)
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
            titre = os.path.basename(doc.metadata.get("ref", doc.metadata.get("source", "inconnu.txt")))
            with st.expander(f"📄 Document {idx} — {titre} (🔍 Pertinence : {pertinence}%)", expanded=False):
                st.markdown(
                    f"""
                    <div style='white-space: pre-wrap; word-wrap: break-word; overflow-x: hidden; background-color: #f9f9f9; padding: 1em; border-radius: 8px; border: 1px solid #ddd;'>
                        {doc.page_content}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # Préparation du contexte documentaire (doit être défini avant les prompts)
    context_text = "\n\n".join([
        f"<doc pertinence={score:.2f}>\n{doc.page_content.strip()}\n</doc>"
        for doc, score, pertinence in filtered_docs
    ])
    # Définition des mots-clés pour chaque modèle
    flan_keywords = {"question": "Question", "context": "Contexte documentaire", "response": "Réponse"}
    plg_keywords = {"question": "question", "context": "contexte", "response": "résumé"}
    # Construction du prompt à partir de la personnalisation utilisateur
    prompt_flan = f"""{user_prompt_intro}\n\nQuestion : {user_input}\n\nContexte documentaire :\n{context_text}\n"""
    prompt_plg = f"""{user_prompt_intro}\n\nQuestion : {user_input}\n\nContexte documentaire :\n{context_text}\n"""
    # Génération des deux réponses en colonnes, d'abord le modèle le plus rapide (flan-t5-small)
    col1, col2 = st.columns(2)
    output_flan = None
    output_plg = None
    with col1:
        with st.spinner("Génération de la réponse (flan-t5-small)..."):
            try:
                output_flan = generate_response(prompt_flan, flan_tokenizer, flan_model)
            except Exception as e:
                st.error(f"Erreur génération flan-t5-small : {e}")
        st.subheader("Réponse (flan-t5-small)")
        if output_flan:
            st.write(output_flan)
        else:
            st.info("Aucune réponse générée par flan-t5-small.")
    with col2:
        with st.spinner("Génération de la réponse (t5-base-fr-sum-cnndm)..."):
            try:
                output_plg = generate_response(prompt_plg, plg_tokenizer, plg_model)
            except Exception as e:
                st.error(f"Erreur génération t5-base-fr-sum-cnndm : {e}")
        st.subheader("Réponse (t5-base-fr-sum-cnndm)")
        if output_plg:
            st.write(output_plg)
        else:
            st.info("Aucune réponse générée par t5-base-fr-sum-cnndm.")