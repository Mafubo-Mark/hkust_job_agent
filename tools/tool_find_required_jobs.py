"""Find every job listing that mentions requested skills using the full-job RAG index."""
import re
from config import packed
from rag.retriever import retrieve

SKILLS = ('python', 'sql', 'java', 'javascript', 'typescript', 'c++', 'c#', 'rust',
          'pytorch', 'tensorflow', 'excel', 'react', 'docker', 'kubernetes',
          'machine learning', 'data analysis')


def requested_skills(query):
    """Recognize explicit technology names without asking a model to invent search criteria."""
    query = query.casefold()
    return [skill for skill in SKILLS if re.search(r'(?<!\w)' + re.escape(skill) + r'(?!\w)', query)]


def source_quotes(job, skills):
    """Collect exact listing excerpts for every requested skill, or reject the job if one is absent."""
    quotes = []
    for skill in skills:
        pattern = re.compile(r'(?<!\w)' + re.escape(skill) + r'(?!\w)', re.I)
        requirement = next((r for r in job.get('requirements', []) if pattern.search(r)), None)
        line = requirement or next((line for line in job['description'].splitlines() if pattern.search(line)), None)
        if not line:
            return []
        quotes.append(' '.join(line.split())[:250])
    return quotes


def tool_find_required_jobs(ctx, User_requests):
    """Use full-job vectors to cover every saved job, then ground matches in exact job text."""
    jobs = {j['job_id']: j for j in ctx.jobs()}
    ordered = [jobs[row['job_id']] for row in retrieve(ctx, User_requests, 'jobs', None)]
    if len(ordered) != len(jobs):
        raise ValueError('The job index is incomplete; prepare the jobs again.')
    skills = requested_skills(User_requests)
    matched, uncertain = [], []
    for job in ordered:
        if skills:
            quotes = source_quotes(job, skills)
            decision = {'decision': 'match' if quotes else 'no_match', 'quotes': quotes,
                        'reason': 'Listing mentions ' + ', '.join(skills) + '.' if quotes else ''}
        else:
            try:
                decision = ctx.llm.json('Decide if this job satisfies the user criteria. Return '
                    '{"decision":"match"|"no_match"|"unknown", "reason":string, "quotes":[string]}. '
                    'For a match, quote exact contiguous text from the supplied job. '
                    'Missing information is unknown, not a match.', {'request': User_requests, 'job': job})
                quotes = decision.get('quotes')
                if (decision.get('decision') not in ('match', 'no_match', 'unknown') or
                    not isinstance(decision.get('reason'), str) or not isinstance(quotes, list) or
                    any(not isinstance(q, str) or q not in packed(job) for q in quotes) or
                    (decision['decision'] == 'match' and not quotes)):
                    raise ValueError('Unsupported model evidence')
            except (ValueError, TypeError, KeyError, IndexError):
                decision = {'decision': 'unknown', 'reason': 'Could not verify the criteria from this listing.', 'quotes': []}
        item = {**job, **decision}
        if decision['decision'] == 'match':
            matched.append(item)
        elif decision['decision'] == 'unknown':
            uncertain.append(item)
    names = [{k: job.get(k) for k in ('job_id', 'company', 'title', 'ref_no')} for job in matched]
    lines = [f'Checked all {len(jobs)} saved jobs. Found {len(matched)} listings mentioning the requested skill(s).'
             if skills else f'Checked all {len(jobs)} saved jobs: {len(matched)} matches, {len(uncertain)} uncertain.']
    for index, job in enumerate(matched, 1):
        lines.append(f"{index}. Company Name: {job['company']}\n   Job Title: {job['title']}\n"
                     f"   Ref No.: {job.get('ref_no') or 'Not provided'}\n"
                     f"   Listing text: {'; '.join(job['quotes'])}")
    if uncertain:
        lines.append(f'{len(uncertain)} job(s) could not be verified from the saved listing.')
    if skills and matched:
        lines.append('A mention can be a preferred skill or one of several options; check the listing before applying.')
    return {'status': 'ok' if not uncertain else 'partial', 'answer': '\n\n'.join(lines),
            'jobs': names, 'uncertain': [{k: job.get(k) for k in ('job_id', 'company', 'title')} for job in uncertain],
            'checked': len(jobs), 'displayed_job_ids': [job['job_id'] for job in matched]}
