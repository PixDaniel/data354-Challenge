import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI

# Importer nos agents RAG précédents comme outils
from src.rag_approaches.baseline_rag import BaselineRAG
from src.rag_approaches.crag import CorrectiveRAG
from src.rag_approaches.graph_rag import GraphRAG

# --- Configuration initiale ---
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("Veuillez configurer TAVILY_API_KEY dans votre fichier .env")

# --- Définition de l'état de l'agent ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# --- Initialisation des outils ---
# Outil de recherche web
web_search_tool = TavilySearchResults(max_results=3, tavily_api_key=TAVILY_API_KEY)
web_search_tool.name = "recherche_web"
web_search_tool.description = "Utilisez cet outil pour trouver des informations sur le web, pour des sujets d'actualité ou non liés au droit du travail ivoirien."

# Instanciation de nos agents RAG
# Note : Pour une meilleure performance, on pourrait rendre leur initialisation 'lazy'
baseline_rag = BaselineRAG()
crag_agent = CorrectiveRAG()
try:
    graph_rag_agent = GraphRAG()
    graph_tool_description = "Utilisez cet outil pour répondre aux questions sur les relations complexes entre les concepts du droit du travail ivoirien (employeurs, salariés, contrats, licenciements, etc.)."
except Exception:
    graph_rag_agent = None
    graph_tool_description = "Outil de graphe non disponible. Assurez-vous que Neo4j est configuré et que le graphe est construit."


# Création d'outils à partir de nos agents RAG
def create_rag_tool(rag_instance, name, description):
    def rag_func(question: str) -> str:
        return rag_instance.ask(question)
    from langchain_core.tools import Tool
    return Tool(name=name, func=rag_func, description=description)

baseline_rag_tool = create_rag_tool(
    baseline_rag, 
    "droit_travail_simple", 
    "Utilisez cet outil pour les questions factuelles simples sur le droit du travail ivoirien."
)

crag_tool = create_rag_tool(
    crag_agent,
    "droit_travail_robuste",
    "Utilisez cet outil pour les questions ambiguës ou complexes sur le droit du travail ivoirien qui nécessitent une vérification approfondie des sources."
)

# On n'ajoute l'outil de graphe que s'il a pu être initialisé
tools = [web_search_tool, baseline_rag_tool, crag_tool]
if graph_rag_agent:
    graph_rag_tool = create_rag_tool(
        graph_rag_agent,
        "droit_travail_graphe",
        graph_tool_description
    )
    tools.append(graph_rag_tool)

# --- Création de l'agent routeur ---
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Vous êtes un super-agent expert, un routeur intelligent. Votre rôle est de choisir le meilleur outil pour répondre à la question de l'utilisateur.
            Vous avez accès aux outils suivants :
            - recherche_web : Pour les questions sur tout sujet en dehors du droit du travail ivoirien.
            - droit_travail_simple : Pour les questions factuelles et simples sur le droit du travail ivoirien.
            - droit_travail_robuste : Pour les questions complexes ou ambiguës sur le droit du travail ivoirien.
            - droit_travail_graphe : Pour les questions sur les liens et relations entre les concepts juridiques du travail en Côte d'Ivoire.
            """,
        ),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

llm = GoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=os.getenv("GEMINI_API_KEY"))

agent_executor = create_tool_calling_agent(llm, tools, prompt)

# --- Définition du graphe de l'agent ---
agent_graph = StateGraph(AgentState)

def run_agent(state):
    agent_outcome = agent_executor.invoke(state)
    # Le message de l'agent (avec les appels d'outils) est stocké dans 'agent_scratchpad' implicitement.
    # Le message de retour final est ce que nous voulons ajouter à notre état.
    return {"messages": [HumanMessage(content=agent_outcome["output"])]}

agent_graph.add_node("agent", run_agent)
agent_graph.set_entry_point("agent")
agent_graph.add_edge("agent", END)

runnable_agent = agent_graph.compile()

class AgenticRAG:
    """
    Implémentation de l'approche AgenticRAG, un agent routeur.
    """
    def __init__(self):
        self.agent = runnable_agent

    def ask(self, question: str):
        result = self.agent.invoke({"messages": [HumanMessage(content=question)]})
        return result['messages'][-1].content

if __name__ == '__main__':
    print("Initialisation de l'AgenticRAG...")
    agentic_rag = AgenticRAG()

    q1 = "Quelles sont les obligations de l'employeur en matière de congés payés ?"
    print(f"\nQuestion : {q1}")
    # Cette question devrait probablement utiliser l'outil 'droit_travail_robuste' ou 'droit_travail_graphe'
    response1 = agentic_rag.ask(q1)
    print(f"Réponse : {response1}")

    q2 = "Quelle est la météo à Abidjan aujourd'hui ?"
    print(f"\nQuestion : {q2}")
    # Cette question devrait utiliser l'outil 'recherche_web'
    response2 = agentic_rag.ask(q2)
    print(f"Réponse : {response2}")

    q3 = "Quelle est la durée de la période d'essai pour un cadre ?"
    print(f"\nQuestion : {q3}")
    # Cette question pourrait utiliser l'outil 'droit_travail_simple'
    response3 = agentic_rag.ask(q3)
    print(f"Réponse : {response3}") 