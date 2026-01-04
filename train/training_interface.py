import asyncio
from typing import Dict, Any, Optional, cast
from agent.build_agent import build_agent
from agent.state import AgentState
import agentlightning as agl


async def run_graph(app, question):
    return await app.ainvoke({"user_input": question})

def run_async_compatible(coro):
    try:
        loop = asyncio.get_running_loop()
        future = asyncio.ensure_future(coro)
        loop.run_until_complete(future)
        return future.result()
    except RuntimeError:
        # No loop running → safe to use asyncio.run()
        return asyncio.run(coro)


class TrainableLitAgent(agl.LitAgent):

    def __init__(
            self, 
            trained_agents: Optional[str],
            verl_replacement: Dict[str, Any] = None,
            max_turns: int = 3,
            ) -> None:
        super().__init__(trained_agents=trained_agents)
        self.verl_replacement = verl_replacement
        self.max_turns = max_turns
        
    def rollout(self, task: Dict[str, Any], resources: agl.NamedResources, rollout: agl.Rollout) -> float | None:
        question = task.get("question")

        llm: agl.LLM = cast(agl.LLM, resources["main_llm"])

        # Build graph
        app = build_agent(state=AgentState, verl_replacement=self.verl_replacement)

        async def coro():
            return await app.ainvoke({"user_input": question})

        result = run_async_compatible(coro())

        final_answer = result.get("final_answer")
        return final_answer
    

def train(config, active_agent) -> None:

    agent = TrainableLitAgent()
    algorithm = agl.VERL(config)