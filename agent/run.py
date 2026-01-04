from agent.state import AgentState
from agent.build_agent import build_agent


if __name__ == "__main__":

    import asyncio

    app = build_agent(state=AgentState, verl_replacement=None)

    async def run():
        result = await app.ainvoke({"user_input": """
            task_input": {"date": "2025-10-13", "time": "14:30", "duration_min": 30, "attendees": 12, "needs": ["whiteboard", "confphone"], "accessible_required": true}
        """} )
        
        # print(f"[RoomSelector]:{result}")
        # print(f"[Judge]:{result['judge_output']}")

    asyncio.run(run())
