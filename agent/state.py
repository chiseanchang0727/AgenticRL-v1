from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class JudgeResponse(BaseModel):
    reason: str = Field(description="The reason for the score. No more than 100 characters.")
    score: float = Field(description="The score for the match on a 0-1 scale. Be critical.")

class AgentState(BaseModel):
    user_input: Optional[str] = None
    history: Optional[List[Dict[str, Any]]]  = None
    plan: Optional[str] = None
    tool_call: Optional[Dict[str, Any]] = None  
    tool_result: Optional[Dict[str, Any]] = None
    # model_response: Optional[str] = None
    final_answer: Optional[str] = None
    judge_output: Optional[JudgeResponse] = None
