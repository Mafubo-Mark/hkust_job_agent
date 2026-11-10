from agents.specialist import specialist


def company_search_agent(ctx, state):
    """Let the company specialist choose cited web search using remembered employer context."""
    return specialist(ctx, state, 'company_search_agent', ['tool_search', 'bye_tool'])
