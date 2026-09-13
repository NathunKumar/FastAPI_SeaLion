from typing import Any, Dict
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    customerQuery: str = Field(min_length=1, max_length=4000)

class QueryResponse(BaseModel):
    language: str
    intent: str
    parameters: Dict[str, Any]

class ResponseRequest(BaseModel):
    language: str
    customerQuery: str
    databaseResponse: Any

class ResponseResponse(BaseModel):
    response: str
