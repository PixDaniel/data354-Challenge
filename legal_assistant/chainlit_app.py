import os
import chainlit as cl
import pandas as pd
from dotenv import load_dotenv
from src.rag_approaches.baseline_rag import BaselineRAG
from src.rag_approaches.crag import CorrectiveRAG
from src.rag_approaches.graph_rag import GraphRAG
from src.rag_approaches.agentic_rag import AgenticRAG

# --- Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY non trouvée dans le fichier .env")

# --- Fonctions utilitaires ---
def get_best_model():
    """Récupère le meilleur modèle d'après les résultats d'évaluation."""
    # Retourne directement le CRAG comme modèle par défaut
    return "corrective_rag"

# --- Initialisation des modèles ---
@cl.cache
def load_models():
    models = {
        "baseline_rag": BaselineRAG(),
        "corrective_rag": CorrectiveRAG(),
        "graph_rag": GraphRAG(),
        "agentic_rag": AgenticRAG()
    }
    return models

# --- Configuration de Chainlit ---
@cl.on_chat_start
async def start_chat():
    # Chargement des modèles
    models = load_models()
    best_model = get_best_model()
    
    # Stockage des modèles dans la session utilisateur
    cl.user_session.set("models", models)
    
    # Création du sélecteur de modèle
    actions = [
        cl.Action(name="baseline_rag", value="baseline_rag", label="Baseline RAG"),
        cl.Action(name="corrective_rag", value="corrective_rag", label="Corrective RAG"),
        cl.Action(name="graph_rag", value="graph_rag", label="Graph RAG"),
        cl.Action(name="agentic_rag", value="agentic_rag", label="Agentic RAG")
    ]
    
    await cl.Action(
        name="select_model",
        value=best_model,
        label="Sélectionner un modèle RAG",
        actions=actions
    ).send()
    
    # Définition du modèle par défaut
    cl.user_session.set("current_model", best_model)
    
    # Message d'accueil
    await cl.Message(
        content=f"""# 👋 Bienvenue dans l'Assistant Juridique
        
Je suis spécialisé dans le droit du travail ivoirien et je peux répondre à vos questions concernant la législation du travail en Côte d'Ivoire.

**Modèle actif**: {best_model.replace('_', ' ').title()}

Posez-moi vos questions sur le Code du Travail Ivoirien!
        """,
        author="Assistant Juridique"
    ).send()

# --- Gestion du changement de modèle ---
@cl.action_callback("select_model")
async def on_model_selection(action):
    # Mise à jour du modèle actif
    cl.user_session.set("current_model", action.value)
    
    # Message de confirmation
    model_names = {
        "baseline_rag": "Baseline RAG - Recherche vectorielle simple",
        "corrective_rag": "Corrective RAG - Vérification de la pertinence des documents",
        "graph_rag": "Graph RAG - Graphe de connaissances juridiques",
        "agentic_rag": "Agentic RAG - Décomposition de questions complexes"
    }
    
    await cl.Message(
        content=f"Modèle RAG changé pour: **{model_names[action.value]}**",
        author="Système"
    ).send()
    
    # Réactivation du sélecteur
    await cl.Action(
        name="select_model",
        value=action.value,
        label="Sélectionner un modèle RAG",
        actions=[
            cl.Action(name="baseline_rag", value="baseline_rag", label="Baseline RAG"),
            cl.Action(name="corrective_rag", value="corrective_rag", label="Corrective RAG"),
            cl.Action(name="graph_rag", value="graph_rag", label="Graph RAG"),
            cl.Action(name="agentic_rag", value="agentic_rag", label="Agentic RAG")
        ]
    ).send()

# --- Gestion des messages ---
@cl.on_message
async def on_message(message: cl.Message):
    # Récupération du modèle actif
    models = cl.user_session.get("models")
    current_model_id = cl.user_session.get("current_model")
    current_model = models[current_model_id]
    
    # Message d'attente
    msg = cl.Message(content="", author="Assistant Juridique")
    await msg.send()
    
    try:
        # Traitement spécial pour l'AgenticRAG (visualisation des étapes)
        if current_model_id == "agentic_rag":
            await msg.stream_token("🔍 Analyse de votre question... ")
            
            # Récupération des sous-questions
            try:
                sub_questions = await current_model.get_steps(message.content)
                
                # Création d'un step pour la décomposition
                step = cl.Step(name="Décomposition de la question", type="decomposeQuestion")
                async with step:
                    sub_questions_text = "\n\n".join([f"- {q}" for q in sub_questions])
                    await cl.Message(
                        content=f"J'ai décomposé votre question en sous-questions :\n\n{sub_questions_text}",
                        author="Agentic RAG"
                    ).send()
                await step.send()
                
                # Pour chaque sous-question, créer un step
                for i, sub_q in enumerate(sub_questions):
                    step = cl.Step(name=f"Recherche pour: {sub_q[:30]}...", type="searchInfo")
                    async with step:
                        await cl.Message(
                            content=f"Je recherche des informations pour répondre à: **{sub_q}**",
                            author="Agentic RAG"
                        ).send()
                    await step.send()
                
                # Step de synthèse
                step = cl.Step(name="Synthèse des informations", type="synthesize")
                async with step:
                    await cl.Message(
                        content="Je synthétise les informations trouvées pour formuler une réponse complète...",
                        author="Agentic RAG"
                    ).send()
                await step.send()
            except Exception as e:
                print(f"Erreur dans la visualisation des étapes: {e}")
            
            # Génération de la réponse finale
            response = current_model.ask(message.content)
            await msg.update(content=response)
            
        else:
            # Pour les autres modèles, traitement standard
            # Récupération du contexte (si disponible)
            context = None
            if hasattr(current_model, 'retriever'):
                docs = current_model.retriever.invoke(message.content)
                if docs:
                    context = [doc.page_content for doc in docs]
            
            # Génération de la réponse
            response = current_model.ask(message.content)
            
            # Affichage des sources si disponibles
            if context:
                elements = []
                for i, ctx in enumerate(context):
                    elements.append(
                        cl.Text(
                            name=f"source_{i+1}",
                            content=ctx,
                            display="side"
                        )
                    )
                await msg.update(content=response, elements=elements)
            else:
                await msg.update(content=response)
    
    except Exception as e:
        await msg.update(content=f"❌ Une erreur est survenue: {str(e)}")

# --- Configuration du thème personnalisé ---
cl.configure_theme(
    accent_color="#1E3A8A",
    neutral_color="#4B5563",
    bg_color="#F9FAFB",
    spacing_size=2,
    font_family="Inter, sans-serif",
    chat_message_opacity=0.9,
    chat_message_avatar_opacity=1,
    chat_message_avatar_border_radius="50%",
    chat_message_border_radius="0.75rem",
)

# --- Configuration de l'application Chainlit ---
cl.configure_llm_widget(
    placeholder="Posez votre question sur le droit du travail ivoirien...",
    prefix_text="Question:"
)

# --- Configuration générale ---
cl.set_chat_profiles([
    cl.ChatProfile(
        name="Assistant Juridique",
        markdown_description="""
        # Assistant Juridique - Droit du Travail Ivoirien
        
        Cet assistant est spécialisé dans le **Code du Travail Ivoirien 2021**. Il peut :
        
        * Répondre à des questions sur la législation du travail
        * Citer les articles pertinents du code
        * Expliquer les droits et obligations des employeurs et salariés
        
        ## Approches RAG utilisées
        
        * **Baseline RAG** : Recherche vectorielle simple
        * **Corrective RAG** : Vérification de la pertinence des documents
        * **Graph RAG** : Graphe de connaissances juridiques
        * **Agentic RAG** : Décomposition de questions complexes
        """,
        default=True
    )
])

# --- Métadonnées de l'application ---
metadata = {
    "title": "Assistant Juridique - Droit du Travail Ivoirien",
    "description": "Un assistant conversationnel spécialisé dans le droit du travail ivoirien.",
    "logo": "https://cdn-icons-png.flaticon.com/512/1995/1995539.png",
    "ui": {
        "show_user_input": True,
        "show_user_input_in_history": True,
        "show_chat_history": True,
        "show_header": True,
    }
}

cl.serve(metadata=metadata) 