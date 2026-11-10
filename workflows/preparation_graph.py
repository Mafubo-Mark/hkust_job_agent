import copy
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from functools import partial
from langgraph.graph import StateGraph, START, END
from pydantic import TypeAdapter
import job_matcher_tool as matcher
from tools.tool_summarize import tool_summarize
from rag.build_index import mark_ready
from workflows.state import PreparationState


def extract(ctx, state):
    """Read searchable PDF pages using pypdf; reject scanned or encrypted resumes before scraping."""
    return {'resume': matcher.read_searchable_resume(state['resume_path'])}


def summarize(ctx, state):
    """Store the short resume used for dialogue while retaining full PDF text for scoring."""
    tool_summarize(ctx, state['resume']['text'])
    return {}


def match(ctx, state):
    """Reuse original Selenium filtering and two-line AI scoring, then write the agent's three JSON files."""
    checkpoint = getattr(ctx, '_progress_path', ctx.data / 'pending_match.json')
    pending = matcher.read_json(checkpoint) if state['mode'] == 'resume_scoring' else None
    if pending is not None:
        jobs = TypeAdapter(list[matcher.JobRecord]).validate_python(pending['jobs'])
    elif state['mode'] == 'score_saved':
        jobs = TypeAdapter(list[matcher.JobRecord]).validate_python(matcher.read_json(ctx.data / 'jobs.json'))
    else:
        driver = getattr(ctx, 'driver', None) or matcher.create_driver()
        try:
            jobs = matcher.collect_jobs(driver, ctx.settings, state.get('filters'),
                                        already_logged_in=getattr(ctx, 'driver', None) is not None)
        finally:
            if getattr(ctx, 'driver', None) is None:
                driver.quit()
    if len({j.job_id for j in jobs}) != len(jobs):
        raise ValueError('Duplicate collected job IDs.')
    previous_path = ctx.data / 'result.json'
    previous = matcher.read_json(previous_path) if previous_path.exists() and not state.get('force') else []
    if pending is not None:
        previous = pending['results']
    snapshot = {'jobs': [j.model_dump(mode='json') for j in jobs], 'results': previous}
    matcher.write_json(checkpoint, snapshot)

    def save_progress(rows):
        """Atomically save collected jobs and each validated score for a later scoring-only retry."""
        matcher.write_json(checkpoint, {**snapshot, 'results': rows})

    result = matcher.legacy_scores(jobs, matcher.ResumeProfile.model_validate(state['resume']),
                                ctx.settings, previous, state.get('force', False), on_progress=save_progress)
    matcher.write_json(ctx.data / 'jobs.json', [j.model_dump(mode='json') for j in jobs])
    matcher.write_json(ctx.data / 'jobs_nodes.json', [{'company': j.company, 'title': j.title} for j in jobs])
    matcher.write_json(ctx.data / 'result.json', result)
    return {'count': len(jobs)}


def index(ctx, state):
    """Create the two semantic indexes and a readiness fingerprint after successful scoring."""
    mark_ready(ctx)
    return {}


def build_preparation_graph(ctx):
    """Connect extraction, optional AI summarization, matching, and indexing with LangGraph."""
    graph = StateGraph(PreparationState)
    previous = START
    for name, function in [('extract', extract), ('summarize', summarize), ('match', match), ('index', index)]:
        graph.add_node(name, partial(function, ctx))
        graph.add_edge(previous, name)
        previous = name
    graph.add_edge(previous, END)
    return graph.compile()


def prepare(ctx, resume_path, mode='full', filters=None, force=False):
    """Prepare in a temporary directory, preserving previous data if extraction/scoring/indexing fails."""
    if mode not in ('full', 'score_saved', 'resume_scoring'):
        raise ValueError('Unsupported preparation mode.')
    limits = {name: limit for name, _, limit in matcher.FILTERS}
    if filters is not None:
        if not isinstance(filters, dict) or set(filters) - limits.keys():
            raise ValueError('Unknown filter names.')
        for name, values in filters.items():
            if (not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values)
                    or len(values) > limits[name] or len(set(values)) != len(values)):
                raise ValueError(f'Invalid filter values for {name}.')
    resume_path = str(Path(resume_path).expanduser().resolve())
    with TemporaryDirectory(prefix='.preparing-', dir=ctx.data) as temporary:
        staged = copy.copy(ctx)
        staged.data = Path(temporary)
        staged._progress_path = ctx.data / 'pending_match.json'
        staged._embedder = ctx.embedder
        for name in ('jobs.json', 'result.json'):
            if (ctx.data / name).exists():
                shutil.copy2(ctx.data / name, staged.data / name)
        try:
            result = build_preparation_graph(staged).invoke(
                dict(resume_path=resume_path, mode=mode, filters=filters, force=force))
        finally:
            ctx._llm = staged._llm
        (ctx.data / 'prepared.json').unlink(missing_ok=True)
        for name in ('resume.txt', 'jobs.json', 'jobs_nodes.json', 'result.json'):
            (staged.data / name).replace(ctx.data / name)
        (ctx.data / 'vector_db').mkdir(exist_ok=True)
        (staged.data / 'vector_db' / 'jobs.sqlite').replace(ctx.data / 'vector_db' / 'jobs.sqlite')
        (staged.data / 'prepared.json').replace(ctx.data / 'prepared.json')
        staged._progress_path.unlink(missing_ok=True)
    return {'count': result['count'], 'data_dir': str(ctx.data)}
