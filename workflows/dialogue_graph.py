import json
from contextlib import contextmanager
from functools import partial
from langchain_core.messages import AIMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from agents.supervisor import supervisor
from agents.job_agent import job_agent
from agents.resume_agent import resume_agent
from agents.company_search_agent import company_search_agent
from config import packed
from tools.job_tools import execute
from tools.bye_tool import is_goodbye
from workflows.state import DialogueState


def begin(ctx, state):
    """Reset per-turn work and invalidate referents when the prepared dataset changes."""
    version = ctx.ready()
    result = {'dataset_id': version, 'tasks': [], 'scratch': [], 'calls': [],
              'tool_events': [], 'steps': 0, 'answer': '', 'notes': [],
              'turn_count': state.get('turn_count', 0) + 1, 'end_session': False}
    if state.get('dataset_id') != version:
        result.update(displayed_job_ids=[], selected_job_id=None, selected_company=None,
                      messages=[RemoveMessage(id=REMOVE_ALL_MESSAGES), state['messages'][-1]])
    return result


def run_tools(ctx, state):
    """Execute real tool calls sequentially so later calls can use updated job references."""
    scratch, events, current, updates = list(state['scratch']), list(state['tool_events']), dict(state), {}
    for call in state['calls']:
        name = call['function']['name']
        try:
            result = execute(ctx, name, json.loads(call['function']['arguments']), current)
        except Exception as error:
            result = {'status': 'error', 'answer': f'{name} failed ({type(error).__name__}). '
                      'Check saved inputs, service credentials, or connectivity and retry. No result was fabricated.'}
        events.append({'tool': name, **result})
        if result.get('end_session'):
            updates['end_session'] = True
        for field in ('displayed_job_ids', 'selected_job_id', 'selected_company'):
            if field in result:
                current[field] = updates[field] = result[field]
        if 'displayed_job_ids' in result:
            current['selected_job_id'] = updates['selected_job_id'] = None
            current['selected_company'] = updates['selected_company'] = None
        scratch.append({'role': 'tool', 'tool_call_id': call['id'], 'content': packed(result)})
        if result.get('end_session'):
            break
    return {**updates, 'scratch': scratch, 'tool_events': events, 'calls': []}


def next_task(state):
    """Finish a specialist handoff and prepare the next specialist's independent context."""
    notes = list(state.get('notes', []))
    if state.get('answer') and not any(m['role'] == 'tool' for m in state.get('scratch', [])):
        notes.append(state['answer'])
    return {'tasks': state['tasks'][1:], 'scratch': [], 'calls': [], 'notes': notes}


def route_first(state):
    """Send a farewell straight to the bye tool before model routing."""
    return 'bye' if is_goodbye(state['messages'][-1].content) else 'supervisor'


def bye(ctx, state):
    """Record an actual bye-tool call; file cleanup waits until the graph has closed."""
    return {'tool_events': [{'tool': 'bye_tool', **execute(ctx, 'bye_tool', {}, state)}],
            'end_session': True}


def return_to_agent(state):
    """Return ordinary results to their specialist and end after the bye tool."""
    return 'finish' if state.get('end_session') else state['agent']


def route_task(state):
    """Route only validated supervisor tasks to known LangGraph nodes."""
    return state['tasks'][0]['agent'] if state.get('tasks') else 'finish'


def after_agent(state):
    """Continue to actual tools when requested, otherwise complete this specialist task."""
    return 'tools' if state.get('calls') else 'next_task'


def finish(state):
    """Render actual tool outputs and tool names, preserving exact rankings and source links."""
    events = state.get('tool_events', [])
    answer = '\n\n'.join(f"Tool used: {e['tool']}\n{e['answer']}" for e in events)
    if state.get('notes'):
        answer = '\n\n'.join(filter(None, [answer, *state['notes']]))
    answer = answer or state.get('answer') or 'Please specify a job or career question.'
    if state.get('steps', 0) >= 8:
        answer += '\n\nTool-call limit reached; narrow any unanswered part of your request.'
    finished = state.get('end_session', False) or state.get('turn_count', 0) >= 30
    if state.get('turn_count', 0) >= 30 and not state.get('end_session'):
        answer += '\n\n30 dialogue rounds reached. Closing this session and clearing its generated files.'
    return {'answer': answer, 'messages': [AIMessage(content=answer)], 'end_session': finished}


def build_dialogue_graph(ctx, checkpointer):
    """Compile the supervisor, specialists, farewell route, and bounded in-memory dialogue."""
    graph = StateGraph(DialogueState)
    for name, function in {'begin': begin, 'supervisor': supervisor, 'job_agent': job_agent,
                           'resume_agent': resume_agent, 'company_search_agent': company_search_agent,
                           'tools': run_tools}.items():
        graph.add_node(name, partial(function, ctx))
    graph.add_node('next_task', next_task)
    graph.add_node('bye', partial(bye, ctx))
    graph.add_node('finish', finish)
    graph.add_edge(START, 'begin')
    graph.add_conditional_edges('begin', route_first, ['bye', 'supervisor'])
    graph.add_edge('bye', 'finish')
    choices = ['job_agent', 'resume_agent', 'company_search_agent', 'finish']
    graph.add_conditional_edges('supervisor', route_task, choices)
    graph.add_conditional_edges('next_task', route_task, choices)
    for name in choices[:-1]:
        graph.add_conditional_edges(name, after_agent, ['tools', 'next_task'])
    graph.add_conditional_edges('tools', return_to_agent, choices)
    graph.add_edge('finish', END)
    return graph.compile(checkpointer=checkpointer)


@contextmanager
def dialogue_session(ctx):
    """Keep conversation state only in memory for this program run."""
    yield build_dialogue_graph(ctx, InMemorySaver())
