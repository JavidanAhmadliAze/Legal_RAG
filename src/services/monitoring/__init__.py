from .factory import get_monitoring_client


def init(
    project_name: str | None = None,
    port: int | None = None,
    *,
    instrument_langchain: bool | None = None,
    instrument_openai: bool | None = None,
) -> None:
    get_monitoring_client().init(
        project_name=project_name,
        port=port,
        instrument_langchain=instrument_langchain,
        instrument_openai=instrument_openai,
    )


def get_tracer(name: str | None = None):
    return get_monitoring_client().get_tracer(name)
