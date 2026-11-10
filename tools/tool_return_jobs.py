def job_line(job, number):
    """Render identifying fields and the saved score without asking an LLM to rewrite them."""
    return (f"{number}. Company Name: {job['company']}\n   Job Title: {job['title']}\n"
            f"   Ref No.: {job.get('ref_no') or 'Not provided'}\n   Score: {job['score']:g}/100\n"
            f"   Eligibility: {job.get('eligibility', 'unknown')}")


def tool_return_jobs(ctx, begin_index=1, finish_index=10):
    """Return a 1-based inclusive range from result.json; record its exact displayed order."""
    if type(begin_index) is not int or type(finish_index) is not int or not 1 <= begin_index <= finish_index:
        raise ValueError('Use 1-based indices with 1 <= begin_index <= finish_index.')
    rows = ctx.rankings()
    fields = ('job_id', 'company', 'title', 'ref_no', 'score', 'rank', 'eligibility')
    selected = [{k: j.get(k) for k in fields} for j in rows[begin_index - 1:finish_index]]
    answer = '\n\n'.join(job_line(j, i + 1) for i, j in enumerate(selected))
    if selected:
        answer = f'Saved ranking positions {begin_index}–{min(finish_index, len(rows))} of {len(rows)}.\n\n' + answer
    else:
        answer = f'No jobs in that range; there are {len(rows)} saved jobs.'
    return {'status': 'ok', 'answer': answer, 'jobs': selected,
            'displayed_job_ids': [j['job_id'] for j in selected], 'total': len(rows)}
