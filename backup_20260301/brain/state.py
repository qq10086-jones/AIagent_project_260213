from typing import Annotated, TypedDict, List, Dict, NotRequired
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[List[Dict], add_messages]
    facts: List[Dict]
    candidates: List[Dict] # List of symbols found during discovery
    narrative: str
    next_step: str
    run_id: str
    symbol: str
    mode: str # 'analysis' for specific ticker, 'discovery' for market scan
    model_preference: str # 'local_small', 'local_large', 'api'
    tool_name: NotRequired[str]
    tool_payload: NotRequired[Dict]
    qwen_model: NotRequired[str]
    local_model: NotRequired[str]
    report_markdown: NotRequired[str]
    report_html_object_key: NotRequired[str]
