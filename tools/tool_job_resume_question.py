from rag.retriever import resolve_job


def tool_job_resume_question(ctx, user_input_job_name, user_query, resume_txt=None, state=None):
    """Resolve a job and ground fit, requirements, interview, or resume advice in saved evidence."""
    resolution = resolve_job(ctx, user_input_job_name, state or {})
    if resolution['status'] != 'ok':
        if resolution.get('candidates'):
            resolution['answer'] += '\n' + '\n'.join(
                f"- {j['company']} | {j['title']} | Ref No.: {j.get('ref_no') or 'Not provided'}"
                for j in resolution['candidates'])
        return resolution
    resume = ctx.resume()
    if resume_txt is not None and resume_txt != resume:
        raise ValueError('Use the saved resume.txt, not replacement resume evidence.')
    job = resolution['job']
    assessment = next(r for r in ctx.rankings() if r['job_id'] == job['job_id'])
    answer = ctx.llm.text('Answer the question using the job, saved assessment, and resume. '
        'For fit: explain supported strengths, gaps, and unknowns; a score is not hiring probability. '
        'For resume edits: propose truthful wording, mark missing experience as a learning task, '
        'never invent achievements, skills, or numbers. For interviews: suggest practice topics '
        'based on the listing; do not claim these are actual company interview questions. '
        'For requirements: cover all supplied requirements and relevant description constraints. '
        'Cite Ref No. and relevant short source quotes. A compressed resume may omit facts; '
        'The saved two-line score is an AI assessment, not proof of any qualification. '
        'Only use original-PDF evidence quotes if the saved assessment actually contains them; '
        'otherwise ground explanations in the resume summary and job text and acknowledge missing facts. '
        'If the sources do not answer the question, say so.',
        {'question': user_query, 'job': job, 'resume_summary': resume, 'assessment': assessment})
    heading = f"{job['company']} — {job['title']} (Ref No.: {job.get('ref_no') or 'Not provided'})"
    return {'status': 'ok', 'answer': heading + '\n\n' + answer,
            'selected_job_id': job['job_id'], 'selected_company': job['company']}
