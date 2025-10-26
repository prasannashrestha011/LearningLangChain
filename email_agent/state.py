from typing  import TypedDict,Literal

class EmailClassification(TypedDict):
    intent:Literal["question","bug","billing","feature","complex"]
    urgency:Literal["low","mid","high"]
    topic:str 
    summary:str

class EmailAgentState(TypedDict):
    email_content:str 
    sender_email:str 
    email_id:str 
    classification:EmailClassification | None 
    search_results:list[str] | None 
    customer_history:list[str] | None
