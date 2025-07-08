import os
from typing import List
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# --- Définition de l'état du graphe ---
class GraphState(TypedDict):
    """
    Représente l'état de notre graphe.

    Attributs:
        question: La question posée par l'utilisateur.
        documents: La liste des documents récupérés.
        generation: La réponse générée par le LLM.
    """
    question: str
    documents: List[str]
    generation: str

class CorrectiveRAG:
    """
    Implémentation d'un pipeline Corrective-RAG (CRAG) avec LangGraph.
    """
    def __init__(self, vector_store_path="src/data/faiss_index"):
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY non trouvée dans le fichier .env")

        # Modèles
        self.llm = GoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=self.api_key, temperature=0)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=self.api_key)

        # Vectorstore et Retriever
        vector_store = FAISS.load_local(vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
        self.retriever = vector_store.as_retriever()
        
        # Construction du graphe
        self.graph = self._build_graph()

    def _build_graph(self):
        # --- Définition des nœuds du graphe ---
        def retrieve_documents(state):
            print("---RÉCUPÉRATION DES DOCUMENTS---")
            question = state["question"]
            documents = self.retriever.invoke(question)
            return {"documents": documents, "question": question}

        def grade_documents(state):
            print("---ÉVALUATION DE LA PERTINENCE DES DOCUMENTS---")
            question = state["question"]
            documents = state["documents"]

            prompt = ChatPromptTemplate.from_template(
                """Vous êtes un évaluateur expert. Votre rôle est de juger de la pertinence d'un document récupéré par rapport à une question utilisateur.
                Donnez un score binaire 'oui' ou 'non' pour indiquer si le document contient des informations pertinentes pour répondre à la question.
                Répondez uniquement avec un objet JSON au format suivant : {{"score": "oui_ou_non"}}

                Document:
                {document}

                Question:
                {question}
                """
            )
            
            parser = JsonOutputParser()
            
            # Évaluation de chaque document
            relevance_scores = []
            for doc in documents:
                # La chaîne doit être construite et invoquée pour chaque document
                chain = prompt | self.llm | parser
                score = chain.invoke({"question": question, "document": doc.page_content})
                relevance_scores.append(score['score'])

            # Décision basée sur les scores
            if "oui" in relevance_scores:
                print("---DÉCISION : DOCUMENTS PERTINENTS---")
                return "generate"
            else:
                print("---DÉCISION : DOCUMENTS NON PERTINENTS---")
                return "rewrite_query"

        def generate_answer(state):
            print("---GÉNÉRATION DE LA RÉPONSE---")
            question = state["question"]
            documents = state["documents"]

            prompt = ChatPromptTemplate.from_template(
                """Vous êtes un assistant spécialisé dans le droit du travail en Côte d'Ivoire.
                Répondez à la question suivante en vous basant uniquement sur le contexte fourni.
                
                Contexte:
                {context}
                
                Question:
                {question}
                
                Réponse:"""
            )
            
            chain = prompt | self.llm | StrOutputParser()
            generation = chain.invoke({"context": documents, "question": question})
            return {"documents": documents, "question": question, "generation": generation}

        def rewrite_query(state):
            print("---TENTATIVE DE REFORMULATION DE LA QUESTION---")
            question = state["question"]

            prompt = ChatPromptTemplate.from_template(
                """Vous êtes un expert en reformulation de requêtes. Votre tâche est de reformuler la question suivante pour la rendre plus susceptible de trouver des réponses pertinentes dans une base de données juridique sur le droit du travail ivoirien.
                Question originale: {question}
                Nouvelle question:"""
            )
            
            chain = prompt | self.llm | StrOutputParser()
            new_question = chain.invoke({"question": question})
            return {"question": new_question}
        
        def handle_no_answer(state):
            print("---AUCUNE RÉPONSE TROUVÉE---")
            return {"generation": "Je ne trouve pas l'information dans le document après plusieurs tentatives."}

        # --- Construction du graphe ---
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", retrieve_documents)
        workflow.add_node("generate", generate_answer)
        workflow.add_node("rewrite", rewrite_query)
        workflow.add_node("no_answer", handle_no_answer)

        workflow.set_entry_point("retrieve")
        workflow.add_conditional_edges(
            "retrieve",
            grade_documents,
            {
                "generate": "generate",
                "rewrite_query": "rewrite"
            }
        )
        # Après reformulation, on retente la récupération
        workflow.add_edge("rewrite", "retrieve") 
        workflow.add_edge("generate", END)
        workflow.add_edge("no_answer", END) # Point de sortie si aucune réponse n'est trouvée

        # La logique pour gérer le cas "pas de réponse" doit être affinée.
        # Pour l'instant, on ne boucle qu'une fois.
        # Une approche plus robuste utiliserait un compteur dans l'état.
        
        return workflow.compile()

    def ask(self, question: str):
        """Pose une question au pipeline CRAG."""
        inputs = {"question": question}
        result = self.graph.invoke(inputs, {"recursion_limit": 5}) # Limite pour éviter les boucles infinies
        return result["generation"]

if __name__ == '__main__':
    crag_agent = CorrectiveRAG()

    question = "Quelles sont les obligations de l'employeur concernant le repos hebdomadaire ?"
    print(f"Question initiale: {question}\n")
    response = crag_agent.ask(question)
    print(f"\nRéponse finale:\n{response}")
    
    # Test avec une question qui n'est probablement pas dans le texte
    question_2 = "Peut-on utiliser son CPF pour un permis de conduire en Côte d'Ivoire ?"
    print(f"\nQuestion initiale: {question_2}\n")
    response_2 = crag_agent.ask(question_2)
    print(f"\nRéponse finale:\n{response_2}") 