import re
from rag.document_loader import source_version
from rag.vector_store import VectorStore


def retrieve(ctx, query, kind='jobs', limit=8):
    """Search either semantic index; reject stale indexes instead of returning old jobs."""
    store = VectorStore(ctx.data / 'vector_db' / 'jobs.sqlite')
    if store.version() != source_version(ctx):
        raise ValueError('Job index is stale; start a fresh run with your PDF.')
    if not ctx.jobs():
        return []
    return store.query(kind, ctx.embedder.encode([query])[0], limit)


def resolve_job(ctx, name, state):
    """Resolve exact IDs, displayed ordinals, remembered pronouns, or ambiguous semantic names."""
    jobs = {j['job_id']: j for j in ctx.jobs()}
    value = name.strip().casefold()
    shown = state.get('displayed_job_ids', [])
    words = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
             '第一': 1, '第二': 2, '第三': 3, '第四': 4, '第五': 5}
    ordinal = re.search(r'\b(\d+)(?:st|nd|rd|th)?\b|第\s*(\d+)', value)
    number = int(next(x for x in ordinal.groups() if x)) if ordinal and re.fullmatch(
        r'(?:the |job |number |#|第)?\s*\d+(?:st|nd|rd|th)?(?: one| job|个|份|个职位)?', value) else None
    for word, n in words.items():
        if value in (word, f'the {word}', f'{word} one', f'the {word} one', f'{word} job', f'the {word} job', word + '个'):
            number = n
    if number is not None:
        if 1 <= number <= len(shown) and shown[number - 1] in jobs:
            return {'status': 'ok', 'job': jobs[shown[number - 1]]}
        return {'status': 'not_found', 'answer': 'That position is not in the last displayed list. Ask for a ranking first.'}
    if value in ('it', 'this job', 'that job', 'this position', 'that position', '它', '这个职位', '这份工作'):
        selected = state.get('selected_job_id')
        return ({'status': 'ok', 'job': jobs[selected]} if selected in jobs else
                {'status': 'ambiguous', 'answer': 'Which job? Please give its title, Ref No., or position in the last list.'})
    exact = [j for j in jobs.values() if value in (j['job_id'].casefold(),
             (j.get('ref_no') or '\0').casefold(), (j['company'] + ' ' + j['title']).casefold())]
    if len(exact) == 1:
        return {'status': 'ok', 'job': exact[0]}
    candidates = exact or [jobs[r['job_id']] for r in retrieve(ctx, name, 'names', 8)]
    if not candidates:
        return {'status': 'not_found', 'answer': 'No saved jobs match that request.'}
    result = ctx.llm.json('Resolve the user job name against candidates. Return {"job_id": string|null, '
        '"ambiguous": boolean}. Select only when company/title identify one clear position; '
        'similarity alone is not proof. Multiple plausible roles require clarification.',
        {'query': name, 'candidates': [{k: j.get(k) for k in ('job_id', 'company', 'title', 'ref_no')} for j in candidates]})
    allowed = {j['job_id'] for j in candidates}
    if result.get('ambiguous') is False and result.get('job_id') in allowed:
        return {'status': 'ok', 'job': jobs[result['job_id']]}
    return {'status': 'ambiguous', 'answer': 'Please choose a job by Ref No. or full title.',
            'candidates': [{k: j.get(k) for k in ('job_id', 'company', 'title', 'ref_no')} for j in candidates]}
