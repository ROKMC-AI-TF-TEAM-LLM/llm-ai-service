from pathlib import Path
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from chains.base import BaseChain
from chains.utils import format_docs
from core.logger import get_logger
from core.prompt_loader import load_chat_prompt
from core.settings import settings

logger = get_logger(__name__)


class RagChain(BaseChain):

    def __init__(self, retriever, **kwargs):
        super().__init__(**kwargs)
        self.retriever = retriever

    def setup(self):
        prompt_file = settings.llm.prompt_file
        logger.info(f"RAG 체인 구성 중 (모델: {self.model}, 프롬프트: {prompt_file}, temperature: {self.temperature})")

        base_dir = Path(__file__).parent.parent
        prompt = load_chat_prompt(base_dir / "prompts" / prompt_file)
        llm = ChatOllama(model=self.model, temperature=self.temperature)

        chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain
