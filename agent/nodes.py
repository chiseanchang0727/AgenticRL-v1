from dotenv import load_dotenv
from pydantic import BaseModel
from agent.state import AgentState, JudgeResponse
from agent.base import RunnableNodeBase
from langchain_core.prompts.chat import ChatPromptTemplate
from agent.tools.tool_def import TOOL_DEFINITIONS
from agent.tools.tool_func import get_rooms_and_availability


load_dotenv()

inferenece_server_url = "http://localhost:8000/v1" #  "http://localhost:8000/v1" | None

class PlannerNode(RunnableNodeBase):
    def __init__(self, verl_replacement, prompt: str = "You are a planner. Create a simple plan."):
        super().__init__(
            role="planner",
            prompt=prompt,
            endpoint=None,
            verl_replacement=verl_replacement,
        )

        self.template = ChatPromptTemplate.from_messages([
            ("system", prompt),
            ("human", "User query: {user_input}")
        ])

    def pre_process_input(self, state: AgentState):
        return self.template.invoke({
            "user_input": state.user_input
        })

    async def post_process_response(self, state: AgentState, response):

        history_entry = [{
            "node": "Planner",
            "content": response.content
        }]

        return state.model_copy(update={
            "plan": response.content,
            "history": (state.history or []) + history_entry,
        })

class CandidateRoomFilterNode(RunnableNodeBase):
    def __init__(self, verl_replacement, prompt: str = "You are a scheduling assistant. Call tools to solve the problem"):
        super().__init__(
            role="room_filter",
            prompt=prompt,
            endpoint=None,
            verl_replacement=verl_replacement,
        )

        self.template = ChatPromptTemplate.from_messages([
            ("system", prompt),
            ("human", "User query: {user_input}")
        ])

    def pre_process_input(self, state: AgentState):
        return self.template.invoke({
            "user_input": state.user_input
        })
    
    async def _call_llm(self, state: AgentState, messages):
        llm = self.get_llm()
        response = await llm.bind_tools(TOOL_DEFINITIONS).ainvoke(messages)
        return response

    async def post_process_response(self, state: AgentState, response):
        
        tool_calls = response.additional_kwargs.get("tool_calls", [])
        if tool_calls:
            print("[DEBUG] Tool calls detected:", tool_calls)

            # Usually you want to parse the args JSON
            first_call = tool_calls[0]
            function_name = first_call["function"]["name"]
            function_args = first_call["function"]["arguments"]  # this is JSON string

            # Convert args string → Python dict
            import json
            args_dict = json.loads(function_args)
            if function_name == "get_rooms_and_availability":
                tool_output = get_rooms_and_availability(
                    date=args_dict["date"],
                    time_str=args_dict["time"],
                    duration_min=args_dict["duration_min"],
                )
            else:
                tool_output = {"error": f"Unknown tool: {function_name}"}

            # Save into state
            history_entry = [{
                "node": "RoomSelector",
                "content": {
                    "tool_name": function_name,
                    "args": args_dict,
                    "result": tool_output,
                }
            }]

            print(f"[CandidateRoomFilte]: {tool_output}")

            return state.model_copy(update={
                "tool_call": {
                    "name": function_name,
                    "args": args_dict,
                },
                "tool_result": tool_output,
                "history": (state.history or []) + history_entry,
            })


        # history_entry = [{
        #     "node": "CandidateRooms",
        #     "content": response.content
        # }]

        # return state.model_copy(update={
        #     "tool_result": response.content,
        #     "history": (state.history or []) + history_entry,
        # })


class RoomSelectorNode(RunnableNodeBase):
    def __init__(self, verl_replacement):
        super().__init__(
            verl_replacement=verl_replacement,
            role="room_selector",
            prompt="You evaluate the quality of the result."
        )
        
        self.template = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                    Based on the provided tool result, selector the room matches user's requirement.
                    If there is no such thing, return "No Room
                """                
            ),
            ("human", "User demand:{user_demand}"),
            ("human", "Tool Result:\n{tool_result}"),
        ])
    
    def pre_process_input(self, state: AgentState):
        return self.template.invoke({
            "user_demand": state.user_input,
            "tool_result": state.tool_result,
        })
        
    async def post_process_response(self, state: AgentState, response: str):

        history_entry = [{
            "node": "RoomSelector",
            "content": response.content
        }]
        
        print(f"[RoomSelector]:{response.content}")

        return state.model_copy(update={
            "final_answer": response.content,
            "history": (state.history or []) + history_entry,
        })


class JudgeNode(RunnableNodeBase):
    def __init__(self, verl_replacement):
        super().__init__(
            verl_replacement=verl_replacement,
            role="judge",
            prompt="You evaluate the quality of the result."
        )

        self.template = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a strict grader of exact room choice.
                    Task output:
                    {final_message}

                    Task expected answer:
                    {expected_choice}

                    Score the match on a 0–1 scale. Be critical.
                    A partially correct match should be scored between 0 and 1.

                    Respond ONLY in JSON:
                    {
                    "score": <float>,
                    "feedback": "<short critique>"
                    }
                """
            ),
            ("human", "Task output again (for attention): {final_message}"),
            ("human", "Expected answer again: {expected_choice}")
        ])


    def _build_structured_output_schema(self):
        self._output_schema = JudgeResponse

    def pre_process_input(self, state: AgentState):
        return self.template.invoke({
            "final_message": state.tool_result,
            "expected_choice": state.plan,
        })

    async def post_process_response(self, state: AgentState, response: JudgeResponse):

        history_entry = [{
            "node": "Judge",
            "content": response.model_dump()
        }]

        return state.model_copy(update={
            "judge_output": response.model_dump(),
            "history": (state.history or []) + history_entry,
        })