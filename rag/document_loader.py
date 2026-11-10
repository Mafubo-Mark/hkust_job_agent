import hashlib
from config import packed
from job_matcher_tool import read_json


def source_version(ctx):
    """Fingerprint both JSON sources plus the embedding model and chunking version."""
    return hashlib.sha256((packed(ctx.jobs()) + packed(read_json(ctx.data / 'jobs_nodes.json')) +
                           ctx.embedder.name + ':chunks-350-overlap60-v1').encode()).hexdigest()


def documents(ctx):
    """Join title-only rows by validated source order and chunk full jobs with stable IDs."""
    jobs, names = ctx.jobs(), read_json(ctx.data / 'jobs_nodes.json')
    if names != [{'company': j['company'], 'title': j['title']} for j in jobs]:
        raise ValueError('jobs_nodes.json is out of sync with jobs.json; rerun preparation.')
    rows = []
    for index, (job, name) in enumerate(zip(jobs, names)):
        heading = name['company'] + ' | ' + name['title']
        rows.append(('names', job['job_id'], index, heading))
        text = packed(job)
        rows.extend(('jobs', job['job_id'], index, heading + '\n' + text[start:start + 350])
                    for start in range(0, len(text), 290))
    return rows
