import os
from typing import List, Dict, Any, TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from langchain.tools.base import BaseTool
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
import json
import re

# --- Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY non trouvée dans le fichier .env")

# --- Définition de l'état du graphe ---
class AgentState(TypedDict):
    """
    Représente l'état de notre agent.
    """
    question: str
    sub_questions: List[str]
    current_sub_question: str
    sub_answers: Dict[str, str]
    final_answer: str
    done: bool

# --- Outils de recherche ---
class VectorSearchTool(BaseTool):
    """Outil de recherche vectorielle dans le Code du Travail"""
    name = "vector_search"
    description = "Recherche des informations dans le Code du Travail Ivoirien à l'aide d'une recherche vectorielle."
    
    def __init__(self, vector_store_path="src/data/faiss_index"):
        super().__init__()
        # Initialisation des embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
        
        # Chargement de la base vectorielle
        self.vector_store = FAISS.load_local(vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
    
    def _run(self, query: str) -> str:
        """Exécute la recherche vectorielle."""
        documents = self.retriever.invoke(query)
        if not documents:
            return "Aucune information pertinente trouvée."
        
        # Concaténation des contenus des documents
        contexts = [f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(documents)]
        return "\n\n".join(contexts)
    
    def _arun(self, query: str):
        """Version asynchrone (non implémentée)."""
        raise NotImplementedError("La recherche asynchrone n'est pas implémentée.")

class ArticleSearchTool(BaseTool):
    """Outil de recherche d'articles spécifiques dans le Code du Travail"""
    name = "article_search"
    description = "Recherche un article spécifique dans le Code du Travail Ivoirien. Utilisez cette fonction quand vous connaissez le numéro d'article."
    
    def __init__(self, vector_store_path="src/data/faiss_index"):
        super().__init__()
        # Initialisation des embeddings et de la base vectorielle comme dans VectorSearchTool
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
        self.vector_store = FAISS.load_local(vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
    
    def _run(self, article_number: str) -> str:
        """Recherche un article spécifique."""
        # Normalisation du numéro d'article
        article_number = article_number.strip().upper()
        if not article_number.startswith("ARTICLE"):
            article_number = f"ARTICLE {article_number}"
        
        # Recherche dans la base vectorielle
        documents = self.retriever.invoke(article_number)
        
        # Filtrage pour trouver l'article exact
        for doc in documents:
            if article_number in doc.page_content:
                return doc.page_content
        
        return f"L'article {article_number} n'a pas été trouvé dans le Code du Travail."
    
    def _arun(self, article_number: str):
        """Version asynchrone (non implémentée)."""
        raise NotImplementedError("La recherche asynchrone n'est pas implémentée.")

class DefinitionSearchTool(BaseTool):
    """Outil de recherche de définitions juridiques dans le Code du Travail"""
    name = "definition_search"
    description = "Recherche la définition d'un terme juridique dans le Code du Travail Ivoirien."
    
    def __init__(self, vector_store_path="src/data/faiss_index"):
        super().__init__()
        # Initialisation comme les autres outils
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
        self.vector_store = FAISS.load_local(vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
    
    def _run(self, term: str) -> str:
        """Recherche la définition d'un terme."""
        query = f"définition {term} code travail ivoirien"
        documents = self.retriever.invoke(query)
        
        if not documents:
            return f"Aucune définition trouvée pour '{term}'."
        
        # Concaténation des contenus des documents
        contexts = [doc.page_content for doc in documents]
        return "\n\n".join(contexts)
    
    def _arun(self, term: str):
        """Version asynchrone (non implémentée)."""
        raise NotImplementedError("La recherche asynchrone n'est pas implémentée.")

class AgenticRAG:
    """
    Implémentation d'un agent conversationnel AgenticRAG.
    """
    def __init__(self, vector_store_path="src/data/faiss_index"):
        # Initialisation du LLM
        self.llm = GoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=GEMINI_API_KEY, temperature=0.2)
        
        # Outils
        self.tools = [
            VectorSearchTool(vector_store_path),
            ArticleSearchTool(vector_store_path),
            DefinitionSearchTool(vector_store_path)
        ]
        
        # Construction du graphe
        self.workflow = self._build_graph()
    
    def _build_graph(self):
        """Construit le graphe d'exécution de l'agent."""
        
        # --- Nœud : Décomposition de la question ---
        def decompose_question(state):
            """Décompose une question complexe en sous-questions plus simples."""
            print("---DÉCOMPOSITION DE LA QUESTION---")
            question = state["question"]
            
            prompt = ChatPromptTemplate.from_template(
                """Vous êtes un expert en droit du travail ivoirien.
                
                Votre tâche est de décomposer la question complexe suivante en sous-questions plus simples et indépendantes.
                Chaque sous-question doit pouvoir être répondue séparément, et l'ensemble des réponses doit permettre de répondre à la question principale.
                
                Générez entre 1 et 3 sous-questions, selon la complexité de la question principale.
                Si la question est déjà simple, vous pouvez simplement la répéter comme unique sous-question.
                
                Répondez uniquement avec un tableau JSON de sous-questions.
                
                Question principale:
                {question}
                
                Sous-questions (format JSON):"""
            )
            
            chain = prompt | self.llm | JsonOutputParser()
            sub_questions = chain.invoke({"question": question})
            
            # Assurer que le résultat est une liste
            if isinstance(sub_questions, dict) and "sub_questions" in sub_questions:
                sub_questions = sub_questions["sub_questions"]
            elif isinstance(sub_questions, str):
                try:
                    parsed = json.loads(sub_questions)
                    if isinstance(parsed, list):
                        sub_questions = parsed
                    elif isinstance(parsed, dict) and "sub_questions" in parsed:
                        sub_questions = parsed["sub_questions"]
                except:
                    sub_questions = [sub_questions]
            
            if not isinstance(sub_questions, list):
                sub_questions = [question]  # Fallback à la question originale
            
            print(f"Question décomposée en {len(sub_questions)} sous-questions:")
            for i, sq in enumerate(sub_questions):
                print(f"  {i+1}. {sq}")
            
            return {"sub_questions": sub_questions, "current_sub_question": sub_questions[0], "sub_answers": {}, "question": question}
        
        # --- Nœud : Réponse à une sous-question ---
        def answer_sub_question(state):
            """Répond à la sous-question actuelle en utilisant les outils disponibles."""
            print(f"---RÉPONSE À LA SOUS-QUESTION: {state['current_sub_question']}---")
            
            # Création d'un prompt pour l'agent
            prompt = ChatPromptTemplate.from_template(
                """Vous êtes un assistant juridique spécialisé dans le droit du travail ivoirien.
                
                Utilisez les outils à votre disposition pour répondre à la question suivante:
                {question}
                
                {format_instructions}
                
                {agent_scratchpad}"""
            )
            
            # Création de l'agent
            agent = (
                {
                    "question": lambda x: x["question"],
                    "format_instructions": lambda _: "Utilisez les outils disponibles pour rechercher les informations pertinentes.",
                    "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"])
                }
                | prompt
                | self.llm.bind_functions(self.tools)
                | OpenAIFunctionsAgentOutputParser()
            )
            
            # Exécution de l'agent
            agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
            result = agent_executor.invoke({"question": state["current_sub_question"], "intermediate_steps": []})
            
            # Mise à jour des réponses aux sous-questions
            sub_answers = state["sub_answers"].copy()
            sub_answers[state["current_sub_question"]] = result["output"]
            
            return {"sub_answers": sub_answers, "current_sub_question": state["current_sub_question"]}
        
        # --- Nœud : Sélection de la prochaine sous-question ---
        def select_next_question(state):
            """Sélectionne la prochaine sous-question à traiter ou termine si toutes sont traitées."""
            sub_questions = state["sub_questions"]
            current_idx = sub_questions.index(state["current_sub_question"])
            
            if current_idx + 1 < len(sub_questions):
                next_question = sub_questions[current_idx + 1]
                print(f"---PASSAGE À LA SOUS-QUESTION SUIVANTE: {next_question}---")
                return {"current_sub_question": next_question, "done": False}
            else:
                print("---TOUTES LES SOUS-QUESTIONS ONT ÉTÉ TRAITÉES---")
                return {"done": True}
        
        # --- Nœud : Synthèse des réponses ---
        def synthesize_answers(state):
            """Synthétise les réponses aux sous-questions pour produire une réponse finale."""
            print("---SYNTHÈSE DES RÉPONSES---")
            
            question = state["question"]
            sub_questions = state["sub_questions"]
            sub_answers = state["sub_answers"]
            
            # Construction du contexte pour la synthèse
            context_parts = []
            for sq in sub_questions:
                if sq in sub_answers:
                    context_parts.append(f"Question: {sq}\nRéponse: {sub_answers[sq]}")
            
            context = "\n\n".join(context_parts)
            
            prompt = ChatPromptTemplate.from_template(
                """Vous êtes un expert en droit du travail ivoirien.
                
                Votre tâche est de synthétiser les informations suivantes pour répondre à la question principale.
                Basez-vous uniquement sur les informations fournies dans les réponses aux sous-questions.
                
                Question principale:
                {question}
                
                Informations disponibles:
                {context}
                
                Réponse synthétisée:"""
            )
            
            chain = prompt | self.llm | StrOutputParser()
            final_answer = chain.invoke({"question": question, "context": context})
            
            return {"final_answer": final_answer}
        
        # --- Définition du graphe ---
        workflow = StateGraph(AgentState)
        
        # Ajout des nœuds
        workflow.add_node("decompose", decompose_question)
        workflow.add_node("answer", answer_sub_question)
        workflow.add_node("select_next", select_next_question)
        workflow.add_node("synthesize", synthesize_answers)
        
        # Définition du point d'entrée
        workflow.set_entry_point("decompose")
        
        # Connexion des nœuds
        workflow.add_edge("decompose", "answer")
        workflow.add_edge("answer", "select_next")
        
        # Conditionnelle pour soit continuer avec la prochaine question, soit synthétiser
        workflow.add_conditional_edges(
            "select_next",
            lambda x: "synthesize" if x["done"] else "answer",
            {
                "synthesize": "synthesize",
                "answer": "answer"
            }
        )
        
        # Point de sortie
        workflow.add_edge("synthesize", END)
        
        return workflow.compile()
    
    def ask(self, question: str) -> str:
        """Pose une question à l'agent AgenticRAG."""
        inputs = {"question": question, "done": False}
        result = self.workflow.invoke(inputs)
        return result["final_answer"]

if __name__ == '__main__':
    agentic_rag = AgenticRAG()
    
    # Test avec une question complexe
    question = "Quelles sont les obligations de l'employeur et les droits du salarié en cas de licenciement pour motif économique ?"
    print(f"\nQuestion: {question}")
    response = agentic_rag.ask(question)
    print(f"\nRéponse finale:\n{response}") 