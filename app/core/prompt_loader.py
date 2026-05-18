import yaml
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def load_system_message(path: str | Path, encoding: str = "utf-8") -> str:
    """YAML 프롬프트 파일에서 system 메시지 content만 추출합니다."""
    with open(path, encoding=encoding) as f:
        config = yaml.safe_load(f)
    for msg in config["messages"]:
        if msg["_type"] == "system":
            return msg["content"]
    raise ValueError(f"system 메시지를 찾을 수 없습니다: {path}")


def load_format_reminder(path: str | Path, encoding: str = "utf-8") -> str | None:
    """YAML 프롬프트 파일에서 format_reminder 메시지 content를 추출합니다. 없으면 None."""
    with open(path, encoding=encoding) as f:
        config = yaml.safe_load(f)
    for msg in config["messages"]:
        if msg["_type"] == "format_reminder":
            return msg["content"]
    return None


def load_chat_prompt(path: str | Path, encoding: str = "utf-8") -> ChatPromptTemplate:
    """messages 배열 형식의 YAML 파일에서 ChatPromptTemplate을 로드합니다."""
    with open(path, encoding=encoding) as f:
        config = yaml.safe_load(f)

    messages = []
    for msg in config["messages"]:
        msg_type = msg["_type"]
        if msg_type == "placeholder":
            messages.append(
                MessagesPlaceholder(
                    variable_name=msg["variable_name"],
                    optional=msg.get("optional", False),
                )
            )
        else:
            role = {"system": "system", "human": "human", "ai": "ai"}[msg_type]
            messages.append((role, msg["content"]))

    return ChatPromptTemplate.from_messages(messages)
