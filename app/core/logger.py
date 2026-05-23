import logging

_level: int = logging.INFO
_loggers: list[logging.Logger] = []
_formatter = logging.Formatter(
    fmt="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def configure_logging(level: str = "INFO") -> None:
    """main.py의 lifespan에서 settings 로드 후 1회 호출한다.
    이후 생성된 로거도 동일 레벨을 사용한다."""
    global _level
    _level = getattr(logging, level.upper(), logging.INFO)
    for lgr in _loggers:
        lgr.setLevel(_level)


def get_logger(name: str) -> logging.Logger:
    lgr = logging.getLogger(name)
    if not lgr.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_formatter)
        lgr.addHandler(handler)
        lgr.propagate = False
        lgr.setLevel(_level)
        _loggers.append(lgr)
    return lgr