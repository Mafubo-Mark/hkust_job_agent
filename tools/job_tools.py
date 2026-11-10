"""Validated dispatch for the five job tools and the session-ending bye tool."""
from pydantic import BaseModel, ConfigDict, Field
from tools.tool_summarize import tool_summarize
from tools.tool_return_jobs import tool_return_jobs
from tools.tool_job_resume_question import tool_job_resume_question
from tools.tool_find_required_jobs import tool_find_required_jobs
from tools.tool_search import tool_search
from tools.bye_tool import bye_tool


class Arguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Range(Arguments):
    begin_index: int = Field(default=1, ge=1)
    finish_index: int = Field(default=10, ge=1)


class Question(Arguments):
    user_input_job_name: str = Field(min_length=1, max_length=1000)
    user_query: str = Field(min_length=1, max_length=12000)


class Requirements(Arguments):
    User_requests: str = Field(min_length=1, max_length=12000)


class Search(Arguments):
    User_question: str = Field(min_length=1, max_length=12000)


class Summary(Arguments):
    text: str = Field(min_length=1, max_length=100000)


TOOLS = {
    'tool_return_jobs': (Range, tool_return_jobs, 'Return saved job rankings: 1-based inclusive range. Top 5 = 1 to 5. Scores measure fit, not guaranteed eligibility.'),
    'tool_job_resume_question': (Question, tool_job_resume_question, 'Explain job fit, resume changes, requirements, or interview preparation. Pass a job ID/Ref No./name or exactly "the second one"/"it" for context.'),
    'tool_find_required_jobs': (Requirements, tool_find_required_jobs, 'Find all jobs mentioning required skills. Use the full-job RAG index and return every grounded company name and title.'),
    'tool_search': (Search, tool_search, 'Search the web for company and career information with Baidu sources. Resolve company pronouns using context.'),
    'tool_summarize': (Summary, tool_summarize, 'Summarize supplied resume text to <=500 units without changing the prepared resume or rankings during chat.'),
    'bye_tool': (Arguments, bye_tool, 'End this conversation and clear all runtime outputs when the user says goodbye or is finished.')}


def schemas(names):
    """Expose only the permitted specialist tools through native function-calling schemas."""
    return [{'type': 'function', 'function': {'name': n, 'description': TOOLS[n][2],
             'parameters': TOOLS[n][0].model_json_schema()}} for n in names]


def execute(ctx, name, arguments, state):
    """Validate model arguments before dispatch and keep resume/state arguments under application control."""
    if name not in TOOLS:
        raise ValueError('Unknown tool.')
    model, function, _ = TOOLS[name]
    values = model.model_validate(arguments).model_dump()
    if name == 'tool_job_resume_question':
        values['state'] = state
    if name == 'tool_summarize':
        values['save'] = False
    return function(ctx, **values)
