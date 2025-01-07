import logging

from langgraph.graph import StateGraph, START
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agentorg.workers.worker import BaseWorker, register_worker
from agentorg.workers.prompts import database_action_prompt
from agentorg.workers.tools.RAG.utils import ToolGenerator
from agentorg.utils.utils import chunk_string
from agentorg.utils.graph_state import MessageState
from agentorg.utils.model_config import MODEL
from agentorg.utils.graph_state import StatusEnum



logger = logging.getLogger(__name__)


@register_worker
class QueueWorker(BaseWorker):

    description = "Helps the user push, pop and read the contents of the queue."

    def __init__(self):
        self.llm = ChatOpenAI(model=MODEL["model_type_or_path"], timeout=30000)
        self.actions = {
            "Push": "Push item onto the queue",
            "Pop": "Pop item out off the queue",
            "List": "Get back all items in the queue as a list",
            "Others": "Other actions not mentioned above"
        }
        self.action_graph = self._create_action_graph()

    def push(self, state: MessageState):
        logger.info("list push function was called:")
        # For now we just append something to the queue
        index = len(state['queue']) + 1
        state['queue'].append(str(index))
        state["status"] = StatusEnum.COMPLETE
        state["message_flow"] = f'''Successfully pushed the element {index} onto the queue''' 
        return state

    def pop(self, state:MessageState):
        logger.info("list pop function was called:")
        if not state['queue']:
            state["status"] = StatusEnum.INCOMPLETE
            state["message_flow"] = f'''Queue is currently empty''' 
            return state
        popped_element = state['queue'].pop()
        state["status"] = StatusEnum.COMPLETE
        state["message_flow"] = f'''Successfully popped element: {popped_element} off the queue''' 
        return state
    
    def list_queue(self, state:MessageState):
        logger.info("list queue function was called:")
        state["status"] = StatusEnum.COMPLETE
        state["message_flow"] = f'''The current queue looks like: {str(state['queue'])}. Display this list as is without adding or removing any elements.''' 
        logger.info(state['queue'])
        return state

    def verify_action(self, msg_state: MessageState):
        logger.info('-------------------------')
        logger.info(msg_state)
        logger.info('-------------------------')
        user_intent = msg_state["orchestrator_message"].attribute.get("task", "")
        actions_info = "\n".join([f"{name}: {description}" for name, description in self.actions.items()])
        actions_name = ", ".join(self.actions.keys())

        prompt = PromptTemplate.from_template(database_action_prompt)
        input_prompt = prompt.invoke({"user_intent": user_intent, "actions_info": actions_info, "actions_name": actions_name})
        chunked_prompt = chunk_string(input_prompt.text, tokenizer=MODEL["tokenizer"], max_length=MODEL["context"])
        logger.info(f"Chunked prompt for deciding choosing Queue action: {chunked_prompt}")
        final_chain = self.llm | StrOutputParser()
        try:
            answer = final_chain.invoke(chunked_prompt)
            for action_name in self.actions.keys():
                if action_name in answer:
                    logger.info(f"Chosen action in the queue worker: {action_name}")
                    return action_name
            logger.info(f"Base action chosen in the queue worker: Others")
            return "Others"
        except Exception as e:
            logger.error(f"Error occurred while choosing action in the database worker: {e}")
            return "Others"

        
    def _create_action_graph(self):
        workflow = StateGraph(MessageState)
        # Add nodes for each worker
        workflow.add_node("Push", self.push)
        workflow.add_node("Pop", self.pop)
        workflow.add_node("List", self.list_queue)
        workflow.add_node("Others", ToolGenerator.generate)
        workflow.add_node("tool_generator", ToolGenerator.context_generate)
        workflow.add_conditional_edges(START, self.verify_action)
        workflow.add_edge("Push", "tool_generator")
        workflow.add_edge("Pop", "tool_generator")
        workflow.add_edge("List", "tool_generator")
        return workflow

    def execute(self, msg_state: MessageState):
        graph = self.action_graph.compile()
        result = graph.invoke(msg_state)
        return result
