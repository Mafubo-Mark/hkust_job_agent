from agents.llm import BOUNDARY
from agents.supervisor import conversation
from config import packed
from tools.job_tools import schemas


def specialist(ctx, state, name, allowed):
    """Run a specialist's native tool-choice loop with its own prompt and permitted tools."""
    scratch = state.get('scratch', [])
    if not scratch:
        prompt = (ctx.root / 'prompts' / (name + '.txt')).read_text(encoding='utf-8')
        scratch = [{'role': 'system', 'content': BOUNDARY + '\n' + prompt},
                   {'role': 'user', 'content': packed({**conversation(state),
                    'task': state['tasks'][0]['request'], 'earlier_tool_results': state.get('tool_events', [])})}]
    if state.get('steps', 0) >= 8:
        return {'calls': [], 'scratch': scratch,
                'answer': 'The tool-call limit was reached. Please narrow the remaining request.'}
    response = ctx.llm.chat(scratch, tools=schemas(allowed))
    calls = response.get('tool_calls') or []
    if len(calls) > 5 or any(c.get('function', {}).get('name') not in allowed for c in calls):
        raise ValueError('The specialist requested an unavailable tool or too many calls.')
    return {'scratch': scratch + [response], 'calls': calls, 'agent': name,
            'steps': state.get('steps', 0) + 1,
            'answer': response.get('content') or state.get('answer', '')}
