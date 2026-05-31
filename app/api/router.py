from fastapi import APIRouter
from api.routes import health, rag, rag_agent

api_router = APIRouter(prefix="/api")
api_router.include_router(health.router)
api_router.include_router(rag.router)
api_router.include_router(rag_agent.router)
