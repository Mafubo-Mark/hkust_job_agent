from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class DialogueState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    dataset_id: str
    displayed_job_ids: list[str]
    selected_job_id: str | None
    selected_company: str | None
    tasks: list[dict]
    agent: str
    scratch: list[dict]
    calls: list[dict]
    tool_events: list[dict]
    steps: int
    answer: str
    notes: list[str]
    turn_count: int
    end_session: bool


class PreparationState(TypedDict, total=False):
    resume_path: str
    resume: dict
    mode: str
    filters: dict | None
    force: bool
    count: int
