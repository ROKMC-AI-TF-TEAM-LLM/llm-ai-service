from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.router import api_router
from api.dependencies import get_rag_chain, get_rag_agent_chain
from core.exceptions import AppHTTPException
from dotenv import load_dotenv

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_rag_chain()
    get_rag_agent_chain()
    yield


app = FastAPI(
    title="RAG API",
    description="""
로컬 Ollama LLM을 활용한 하이브리드 RAG REST API 서버입니다.

## 엔드포인트 목록

| 경로 | 설명 |
|---|---|
| `POST /api/rag` | 하이브리드 RAG 질의응답 |
| `POST /api/rag/stream` | 하이브리드 RAG 질의응답 (스트리밍) |
| `POST /api/rag/agent` | 하이브리드 RAG 질의응답 + 다중 턴 대화 |
| `POST /api/rag/agent/stream` | 하이브리드 RAG 대화 (스트리밍) |
""",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.exception_handler(AppHTTPException)
async def app_http_exception_handler(request: Request, exc: AppHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error_code": exc.error_code, "detail": exc.detail},
    )


app.include_router(api_router)


@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse("/docs")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
