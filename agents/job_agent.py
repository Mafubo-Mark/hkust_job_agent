from agents.specialist import specialist


def job_agent(ctx, state):
    """Let the job specialist choose rankings, full-job requirement search, or job explanations."""
    return specialist(ctx, state, 'job_agent',
                      ['tool_return_jobs', 'tool_find_required_jobs', 'tool_job_resume_question', 'bye_tool'])
