from agents.specialist import specialist


def resume_agent(ctx, state):
    """Let the resume specialist choose grounded fit/advice tools or text summarization."""
    return specialist(ctx, state, 'resume_agent', ['tool_job_resume_question', 'tool_summarize', 'bye_tool'])
