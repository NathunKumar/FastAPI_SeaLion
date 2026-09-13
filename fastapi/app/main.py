from fastapi import FastAPI, HTTPException
from .model_service import ModelService
from .prompts import QUERY_SYSTEM_PROMPT, RESPONSE_SYSTEM_PROMPT
from .schemas import QueryRequest, QueryResponse, ResponseRequest, ResponseResponse

app = FastAPI(title="SEA-LION Audit AI", version="1.0.0")
model_service = None
ALLOWED = {"TOP_FAILED_LOGINS","FAILED_LOGIN_COUNT","TOP_ERROR_MODULE","TOP_SLOW_USERS","USER_ACTIVITY_COUNT","MODULE_AVG_RESPONSE_TIME","TOP_ERROR_USERS","GENERAL"}

@app.on_event("startup")
def startup():
    global model_service
    model_service = ModelService()

@app.get("/api/ai/health")
def health():
    return {"status": "UP", "model_loaded": model_service is not None}

@app.post("/api/ai/query", response_model=QueryResponse)
def classify_query(request: QueryRequest):
    try:
        raw = model_service.generate([
            {"role":"system", "content":QUERY_SYSTEM_PROMPT},
            {"role":"user", "content":request.customerQuery}
        ])
        data = model_service.extract_json(raw)
        if data.get("intent") not in ALLOWED:
            raise ValueError("Unsupported intent")
        return QueryResponse(language=data.get("language","en"), intent=data["intent"], parameters=data.get("parameters",{}))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/api/ai/response", response_model=ResponseResponse)
def format_response(request: ResponseRequest):
    try:
        prompt = f"Requested language: {request.language}\nCustomer question: {request.customerQuery}\nDatabase result: {request.databaseResponse}"
        raw = model_service.generate([
            {"role":"system", "content":RESPONSE_SYSTEM_PROMPT},
            {"role":"user", "content":prompt}
        ], max_new_tokens=300)
        return ResponseResponse(response=raw)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
