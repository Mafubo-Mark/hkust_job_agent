from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from config import packed


class Task(BaseModel):
    model_config = ConfigDict(extra='forbid')
    agent: Literal['job_agent', 'resume_agent', 'company_search_agent']
    request: str = Field(min_length=1, max_length=12000)


class Plan(BaseModel):
    model_config = ConfigDict(extra='forbid')
    tasks: list[Task] = Field(max_length=3)
    reply: str = ''


def conversation(state):
    """Provide recent user/assistant turns and explicit referents to each decision-making model."""
    return {'history': [{'role': 'user' if m.type == 'human' else 'assistant', 'content': m.content}
                        for m in state['messages'][-12:]],
            'displayed_job_ids': state.get('displayed_job_ids', []),
            'selected_job_id': state.get('selected_job_id'),
            'selected_company': state.get('selected_company')}


def supervisor(ctx, state):
    """Let DeepSeek plan specialist handoffs using conversation context instead of keywords."""
    instruction = (ctx.root / 'prompts' / 'supervisor.txt').read_text(encoding='utf-8')
    plan = Plan.model_validate(ctx.llm.json(instruction, conversation(state)))
    if not plan.tasks and not plan.reply.strip():
        raise ValueError('The supervisor returned neither a task nor a reply.')
    return {'tasks': [t.model_dump() for t in plan.tasks], 'answer': plan.reply if not plan.tasks else ''}
