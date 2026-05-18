from fastapi import APIRouter
from api.routes import healt, rag, rag_agent

api_router = APIRouter(prefix="/api")
api_router.include_router(healt.router)
api_router.include_router(rag.router)
api_router.include_router(rag_agent.router)
