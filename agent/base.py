import os
from typing import Optional, Union, Dict, Any
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.runnables import Runnable
from agent.state import AgentState
from langchain_openai import ChatOpenAI


class RunnableNodeBase(Runnable):
    def __init__(
            self, 
            role: str, 
            prompt: str, 
            endpoint: str = None,
            verl_replacement: Dict[str, Any] | None = None,
    ):
        self.role = role
        self.prompt = prompt
        self.endpoint=endpoint
        self.verl_replacement = verl_replacement
        
        self._output_schema: Optional[type[BaseModel]] = None
        if hasattr(self, "_build_structured_output_schema"):
            self._build_structured_output_schema()

    def get_llm(self):

        if self.verl_replacement is not None:
            assert self.endpoint is not None
            model: str = self.verl_replacement['model']
            llm = ChatOpenAI(
                model=model,
                openai_api_base=self.endpoint,
                openai_api_key=os.environ.get("OPENAI_API_KEY", "test12345"),
                temperature=self.verl_replacement["temperature"],
                max_tokens=self.verl_replacement["max_tokens"],
            )
        else:
            model: str = os.environ.get("MODEL", "gpt-4.1-mini")
            llm = ChatOpenAI(
                model=model,
                temperature=0.8,
                max_tokens=None,
                max_retries=2,
                api_key=os.environ.get("OPENAI_API_KEY")
            )

        return llm

    async def _call_llm(self, state: AgentState, messages: list[BaseMessage]) -> Union[str, BaseModel, AIMessage]:

        llm = self.get_llm()

        if self._output_schema:
            llm = llm.with_structured_output(self._output_schema)

        response = await llm.ainvoke(messages)
        
        return response 

    def invoke(self, state: AgentState, config: Optional[dict] = None) -> AgentState:
        raise NotImplementedError("Only async flow supported. Use ainvoke.")
    
    def pre_process_input(self, state: AgentState) -> list[BaseMessage]:
        raise NotImplementedError("Each node must define how to compose messages.")

    async def post_process_response(self, state: AgentState, response: AIMessage) -> AgentState:
        return state.model_copy(update={
            "history": state.history + [response.content],
        })
    
    async def ainvoke(self, state: AgentState, config: Optional[dict]=None, **kwargs) -> AgentState:
        messages = self.pre_process_input(state)
        response = await self._call_llm(state, messages) 
        return await self.post_process_response(state, response)
    
    async def __call__(self, state, config: Optional[dict]=None, **kwargs) -> AgentState:
        return await self.ainvoke(state, config, **kwargs)