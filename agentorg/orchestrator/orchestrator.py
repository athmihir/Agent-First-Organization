import json
import time
from typing import Any, Dict
import logging
import uuid
import os
from dotenv import load_dotenv

from langchain_core.runnables import RunnableLambda
import langsmith as ls
from openai import OpenAI

from agentorg.orchestrator.task_graph import TaskGraph
from agentorg.workers.worker import WORKER_REGISTRY
from agentorg.utils.graph_state import ConvoMessage, OrchestratorMessage
from agentorg.utils.utils import init_logger
from agentorg.orchestrator.NLU.nlu import NLU
from agentorg.utils.graph_state import MessageState, StatusEnum
from agentorg.utils.trace import TraceRunName


load_dotenv()
logger = logging.getLogger(__name__)

class AgentOrg:
    def __init__(self, config, **kwargs):
        self.product_kwargs = json.load(open(config))
        os.environ["AVAILABLE_WORKERS"] = ",".join(self.product_kwargs["workers"])
        self.user_prefix = "USER"
        self.worker_prefix = "ASSISTANT"
        self.__eos_token = "\n"
        self.tools = list(WORKER_REGISTRY.keys())
        self.task_graph = TaskGraph("taskgraph", self.product_kwargs)

    def _format_chat_history(self, chat_history, text):
        '''Includes current user utterance'''
        chat_history_str= ""
        for turn in chat_history:
            chat_history_str += f"{turn['role']}: {turn['content']}{self.__eos_token}"
        chat_history_str += f"{self.user_prefix}: {text}"
        return chat_history_str.strip()

    def get_response(self, inputs: dict) -> Dict[str, Any]:
        text = inputs["text"]
        chat_history = inputs["chat_history"]
        params = inputs["parameters"]
        params["timing"] = {}
        chat_history_str = self._format_chat_history(chat_history, text)
        params["dialog_states"] = params.get("dialog_states", [])
        metadata = params.get("metadata", {})
        metadata["conv_id"] = metadata.get("conv_id", str(uuid.uuid4()))
        metadata["turn_id"] = metadata.get("turn_id", 0) + 1
        params["metadata"] = metadata

        ##### Model safety checking
        # check the response, decide whether to give template response or not
        client = OpenAI()
        text = inputs["text"]
        moderation_response = client.moderations.create(input=text).model_dump()
        is_flagged = moderation_response["results"][0]["flagged"]
        if is_flagged:
            return_response = {
                "answer": self.product_kwargs["safety_response"],
                "parameters": params,
                "has_follow_up": True
            }
            return return_response

        ##### TaskGraph Chain
        taskgraph_inputs = {
            "text": text,
            "chat_history_str": chat_history_str,
            "parameters": params  ## TODO: different params for different components
        }
        dt = time.time()
        taskgraph_chain = RunnableLambda(self.task_graph.get_node) | RunnableLambda(self.task_graph.postprocess_node)
        node_info, params = taskgraph_chain.invoke(taskgraph_inputs)
        params["timing"]["taskgraph"] = time.time() - dt
        logger.info("=============node_info=============")
        logger.info(node_info) # {'name': 'MessageWorker', 'attribute': {'value': 'If you are interested, you can book a calendly meeting https://shorturl.at/crFLP with us. Or, you can tell me your phone number, email address, and name; our expert will reach out to you soon.', 'direct': False, 'slots': {"<name>": {<attributes>}}}}

        with ls.trace(name=TraceRunName.TaskGraph, inputs={"taskgraph_inputs": taskgraph_inputs}) as rt:
            rt.end(
                outputs={
                    "metadata": params.get("metadata"),
                    "timing": params.get("timing", {}),
                    "curr_node": {
                        "id": params.get("curr_node"),
                        "name": node_info.get("name"),
                        "attribute": node_info.get("attribute")
                    },
                    "curr_global_intent": params.get("curr_pred_intent"),
                    "dialog_states": params.get("dialog_states"),
                    "node_status": params.get("node_status")}, 
                metadata={"conv_id": metadata.get("conv_id"), "turn_id": metadata.get("turn_id")}
            )
        #### Worker execution
        user_message = ConvoMessage(history=chat_history_str, message=text)
        orchestrator_message = OrchestratorMessage(message=node_info["attribute"]["value"], attribute=node_info["attribute"])
        sys_instruct = "You are a " + self.product_kwargs["role"] + ". " + self.product_kwargs["user_objective"] + self.product_kwargs["builder_objective"] + self.product_kwargs["intro"]
        # Use the queue in params if it exists
        queue = params['queue'] if 'queue' in params else []
        message_state = MessageState(sys_instruct=sys_instruct, user_message=user_message, orchestrator_message=orchestrator_message, message_flow=params.get("worker_response", {}).get("message_flow", ""), slots=params.get("dialog_states"), queue=queue)
        worker = WORKER_REGISTRY[node_info["name"]]()
        worker_response = worker.execute(message_state)
        # Copy back the message state to persist via params.
        params['queue'] = message_state['queue']

        with ls.trace(name=TraceRunName.ExecutionResult, inputs={"message_state": message_state}) as rt:
            rt.end(
                outputs={"metadata": params.get("metadata"), **worker_response}, 
                metadata={"conv_id": metadata.get("conv_id"), "turn_id": metadata.get("turn_id")}
            )

        params["worker_response"] = worker_response
        return_answer = worker_response.get("response", "")
        # node status
        node_status = params.get("node_status", {})
        current_node = params.get("curr_node")
        node_status[current_node] = worker_response.get("status", StatusEnum.COMPLETE)
        params["node_status"] = node_status
        params["dialog_states"] = worker_response.get("slots", [])

        output = {
            "answer": return_answer,
            "parameters": params
        }

        with ls.trace(name=TraceRunName.OrchestResponse) as rt:
            rt.end(
                outputs={"metadata": params.get("metadata"), **output},
                metadata={"conv_id": metadata.get("conv_id"), "turn_id": metadata.get("turn_id")}
            )

        return output
