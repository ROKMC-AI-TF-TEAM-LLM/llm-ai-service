from datetime import datetime
from typing import List, Literal
from pydantic import BaseModel, Field

#Rag 

class RagInput(BaseModel):
    question: str = Field(..., description="질문", examples=["이 문서에서 중요한 내용은 무엇인가요?"])

class RagOutput(BaseModel):
    output: str = Field(..., description="LLM 응답")


#Agent

class MessageItem(BaseModel):
    role: Literal["human", "ai", "system"] = Field(..., description="메시지 역할")
    content: str = Field(..., description="메시지 내용")


class RagAgentInput(BaseModel):
    question: str = Field(..., description="현재 질문", examples=["이 문서에서 중요한 내용은 무엇인가요?"])
    messages: List[MessageItem] = Field(
        default=[],
        description="이전 대화 기록",
        examples=[[{"role": "human", "content": "안녕하세요!"}, {"role": "ai", "content": "안녕하세요! 무엇을 도와드릴까요?"}]],
    )


class SourceItem(BaseModel):
    name: str = Field(..., description="파일명")
    page: str | None = Field(None, description="페이지 번호")


class RagAgentOutput(BaseModel):
    content: str = Field(..., description="LLM 답변")
    sources: List[SourceItem] = Field(default=[], description="검색된 문서 출처 목록")


# Documents

class DocumentItem(BaseModel):
    name: str = Field(..., description="문서 파일명")
    type: str = Field(..., description="문서 형태 (예: PDF)")
    applied_at: datetime = Field(..., description="적용 일시")


class DocumentListOutput(BaseModel):
    documents: List[DocumentItem] = Field(..., description="현재 페이지 문서 목록")
    total: int = Field(..., description="총 문서 수")
    offset: int = Field(..., description="현재 오프셋")
    limit: int = Field(..., description="페이지 크기")
    has_more: bool = Field(..., description="다음 페이지 존재 여부")
