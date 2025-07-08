import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import time
from src.rag_approaches.baseline_rag import BaselineRAG
from src.rag_approaches.crag import CorrectiveRAG
from src.rag_approaches.graph_rag import GraphRAG
from src.rag_approaches.agentic_rag import AgenticRAG

# --- Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY non trouvée dans le fichier .env")
    st.stop()

# --- Initialisation des modèles ---
@st.cache_resource
def load_models():
    models = {
        "Baseline RAG": BaselineRAG(),
        "Corrective RAG (CRAG)": CorrectiveRAG(),
        "Graph RAG": GraphRAG(),
        "Agentic RAG": AgenticRAG()
    }
    return models

# --- Fonctions utilitaires ---
def get_best_model():
    """Récupère le meilleur modèle d'après les résultats d'évaluation."""
    # Retourne directement le CRAG comme modèle par défaut
    return "Corrective RAG (CRAG)"

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Assistant Juridique - Droit du Travail Ivoirien",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS personnalisé ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4B5563;
        margin-bottom: 2rem;
        text-align: center;
    }
    .stTextInput > div > div > input {
        font-size: 1.2rem;
    }
    .user-bubble {
        background-color: #E5E7EB;
        padding: 1rem;
        border-radius: 15px 15px 15px 0;
        margin-bottom: 1rem;
        max-width: 80%;
        align-self: flex-start;
    }
    .assistant-bubble {
        background-color: #DBEAFE;
        padding: 1rem;
        border-radius: 15px 15px 0 15px;
        margin-bottom: 1rem;
        max-width: 80%;
        align-self: flex-end;
        margin-left: auto;
    }
    .context-box {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
        font-size: 0.9rem;
        border-left: 4px solid #3B82F6;
    }
    .model-info {
        background-color: #FEF3C7;
        padding: 0.5rem;
        border-radius: 5px;
        margin-top: 0.5rem;
        font-size: 0.8rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        font-size: 0.8rem;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

# --- En-tête ---
st.markdown('<h1 class="main-header">Assistant Juridique - Droit du Travail Ivoirien</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Posez vos questions sur le Code du Travail de Côte d\'Ivoire</p>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("Paramètres")
    
    # Sélection du modèle
    best_model = get_best_model()
    model_options = ["Baseline RAG", "Corrective RAG (CRAG)", "Graph RAG", "Agentic RAG"]
    selected_model = st.selectbox(
        "Choisissez un modèle RAG",
        options=model_options,
        index=model_options.index(best_model),
        help="Sélectionnez le modèle RAG à utiliser pour répondre à vos questions."
    )
    
    # Informations sur le modèle sélectionné
    st.markdown("<div class='model-info'>", unsafe_allow_html=True)
    if selected_model == "Baseline RAG":
        st.write("**Baseline RAG**: Modèle de base utilisant une recherche vectorielle simple.")
    elif selected_model == "Corrective RAG (CRAG)":
        st.write("**Corrective RAG**: Modèle amélioré qui vérifie la pertinence des documents récupérés.")
    elif selected_model == "Graph RAG":
        st.write("**Graph RAG**: Utilise un graphe de connaissances pour comprendre les relations entre concepts juridiques.")
    elif selected_model == "Agentic RAG":
        st.write("**Agentic RAG**: Agent avancé qui décompose les questions complexes et utilise plusieurs outils spécialisés.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Affichage des contextes
    show_context = st.checkbox("Afficher les sources", value=False, help="Afficher les passages du Code du Travail utilisés pour générer la réponse.")
    
    st.markdown("---")
    st.markdown("### À propos")
    st.write("Cet assistant utilise le Code du Travail Ivoirien 2021 comme source d'information.")
    st.write("Développé avec LangChain, Gemini et des techniques avancées de RAG (Retrieval-Augmented Generation).")

# --- Initialisation de la session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "contexts" not in st.session_state:
    st.session_state.contexts = {}

# --- Chargement des modèles ---
with st.spinner("Chargement des modèles..."):
    models = load_models()

# --- Affichage des messages ---
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        st.markdown(f"<div class='user-bubble'>{message['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='assistant-bubble'>{message['content']}</div>", unsafe_allow_html=True)
        
        # Affichage du contexte si demandé
        if show_context and message["id"] in st.session_state.contexts:
            with st.expander("Sources utilisées"):
                st.markdown("<div class='context-box'>", unsafe_allow_html=True)
                for j, ctx in enumerate(st.session_state.contexts[message["id"]]):
                    st.markdown(f"**Extrait {j+1}:**\n\n{ctx}")
                st.markdown("</div>", unsafe_allow_html=True)

# --- Zone de saisie ---
question = st.text_input("Posez votre question sur le droit du travail ivoirien:", key="question_input")

# --- Traitement de la question ---
if question:
    # Ajout de la question à l'historique
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Réinitialisation de l'input
    st.text_input("Posez votre question sur le droit du travail ivoirien:", key="question_input_reset", value="")
    
    # Affichage de la question
    st.markdown(f"<div class='user-bubble'>{question}</div>", unsafe_allow_html=True)
    
    # Génération de la réponse
    with st.spinner("Recherche en cours..."):
        try:
            model = models[selected_model]
            
            # Récupération du contexte (si disponible)
            context = []
            if hasattr(model, 'retriever'):
                retrieved_docs = model.retriever.invoke(question)
                context = [doc.page_content for doc in retrieved_docs]
            
            # Génération de la réponse
            start_time = time.time()
            answer = model.ask(question)
            end_time = time.time()
            
            # Génération d'un ID unique pour ce message
            message_id = f"msg_{len(st.session_state.messages)}"
            
            # Stockage du contexte
            if context:
                st.session_state.contexts[message_id] = context
            
            # Ajout de la réponse à l'historique
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "id": message_id
            })
            
            # Affichage de la réponse
            st.markdown(f"<div class='assistant-bubble'>{answer}</div>", unsafe_allow_html=True)
            
            # Affichage du contexte si demandé
            if show_context and context:
                with st.expander("Sources utilisées"):
                    st.markdown("<div class='context-box'>", unsafe_allow_html=True)
                    for j, ctx in enumerate(context):
                        st.markdown(f"**Extrait {j+1}:**\n\n{ctx}")
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # Affichage du temps de réponse
            st.caption(f"Temps de réponse: {end_time - start_time:.2f} secondes")
            
        except Exception as e:
            st.error(f"Une erreur est survenue: {e}")

# --- Pied de page ---
st.markdown("<div class='footer'>© 2023 - Assistant Juridique - Droit du Travail Ivoirien</div>", unsafe_allow_html=True)
st.markdown("<div class='footer'>Ce projet a été realisé par Daniel Grah pour le challenge de Data354 </div>", unsafe_allow_html=True)  