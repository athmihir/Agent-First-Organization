import logging

from langgraph.graph import StateGraph, START
from langchain_openai import ChatOpenAI

from agentorg.workers.worker import BaseWorker, register_worker
from agentorg.utils.graph_state import MessageState
from agentorg.workers.tools.RAG.utils import RetrieveEngine, ToolGenerator
from agentorg.utils.model_config import MODEL


logger = logging.getLogger(__name__)


@register_worker
class RAGWorker(BaseWorker):

    description = "Answer the user's questions based on the company's internal documentations (unstructured text data), such as the policies, FAQs, and product information"

    def __init__(self):
        super().__init__()
        self.action_graph = self._create_action_graph()
        self.llm = ChatOpenAI(model=MODEL["model_type_or_path"], timeout=30000)
     
    def _create_action_graph(self):
        workflow = StateGraph(MessageState)
        # Add nodes for each worker
        workflow.add_node("retriever", RetrieveEngine.retrieve)
        workflow.add_node("tool_generator", ToolGenerator.context_generate)
        # Add edges
        workflow.add_edge(START, "retriever")
        workflow.add_edge("retriever", "tool_generator")
        return workflow

    def execute(self, msg_state: MessageState):
        graph = self.action_graph.compile()
        result = graph.invoke(msg_state)
        return result
