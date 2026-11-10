import hashlib
from job_matcher_tool import read_json, write_json
from rag.document_loader import documents, source_version
from rag.vector_store import VectorStore


def build_index(ctx):
    """Build both semantic collections and reuse an unchanged index without re-embedding."""
    store = VectorStore(ctx.data / 'vector_db' / 'jobs.sqlite')
    version = source_version(ctx)
    if store.version() != version:
        rows = documents(ctx)
        vectors = []
        for start in range(0, len(rows), 64):
            vectors.extend(ctx.embedder.encode([r[3] for r in rows[start:start + 64]]))
        store.replace(rows, vectors, version)
    return {'jobs': len(ctx.jobs()), 'index_version': version}


def mark_ready(ctx):
    """Publish a completion marker only when scores, resume, and both indexes are ready."""
    resume_hash = hashlib.sha256((ctx.data / 'resume.txt').read_bytes()).hexdigest()
    marker = ctx.data / 'prepared.json'
    if marker.exists() and read_json(marker).get('resume_hash', resume_hash) != resume_hash:
        raise ValueError('Resume text changed; start a fresh run with the updated PDF.')
    ctx.rankings()
    ctx.resume()
    result = build_index(ctx)
    write_json(ctx.data / 'prepared.json', {**result, 'dataset_id': ctx.dataset_id(), 'resume_hash': resume_hash})
    return result
