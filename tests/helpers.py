import hashlib
import shutil
import tempfile
from pathlib import Path
from config import Context
from job_matcher_tool import write_json
from rag.build_index import mark_ready


class FakeEmbeddings:
    name = 'test-character-vectors-v1'

    def __init__(self):
        """Track embedding calls so index reuse can be tested without downloading a model."""
        self.calls = 0

    def encode(self, texts):
        """Produce reproducible nonzero test vectors; these are not production semantic embeddings."""
        self.calls += 1
        result = []
        for text in texts:
            vector = [0.01] * 32
            for word in text.lower().split():
                vector[int(hashlib.sha256(word.encode()).hexdigest(), 16) % 32] += 1
            result.append(vector)
        return result


def fixture(test, llm=None, count=3):
    """Create isolated synthetic job data, scores, prompts, and real SQLite vector indexes."""
    temporary = tempfile.TemporaryDirectory()
    test.addCleanup(temporary.cleanup)
    root = Path(temporary.name)
    shutil.copytree(Path(__file__).resolve().parents[1] / 'prompts', root / 'prompts')
    ctx = Context(root, llm=llm, embedder=FakeEmbeddings())
    jobs = [{'job_id': f'job-{i}', 'company': 'Huawei' if i < 2 else 'Example',
             'title': ['Data Engineer', 'AI Researcher', 'Developer'][i % 3], 'ref_no': f'REF-{i}',
             'description': 'Python and machine learning. Work location: Hong Kong.',
             'requirements': ['Python'], 'url': f'https://example.com/job/{i}'} for i in range(count)]
    write_json(ctx.data / 'jobs.json', jobs)
    write_json(ctx.data / 'jobs_nodes.json', [{'company': j['company'], 'title': j['title']} for j in jobs])
    write_json(ctx.data / 'result.json', [{**j, 'score': 95 - i, 'rank': i + 1,
                'eligibility': 'unknown'} for i, j in enumerate(jobs)])
    (ctx.data / 'resume.txt').write_text('Built a Python machine learning project.', encoding='utf-8')
    mark_ready(ctx)
    return ctx


class ScriptedLLM:
    def __init__(self, plans=(), calls=(), answers=(), objects=()):
        """Queue expected provider responses to test orchestration independently of model behavior."""
        self.plans, self.calls = list(plans), list(calls)
        self.answers, self.objects, self.seen = list(answers), list(objects), []

    def json(self, instruction, data):
        """Return a queued supervisor plan or tool JSON decision and retain its input for assertions."""
        self.seen.append(('json', instruction, data))
        return self.plans.pop(0) if 'coordinate an HKUST' in instruction else self.objects.pop(0)

    def chat(self, messages, tools=None, json_mode=False):
        """Alternate a queued tool call with a completion, emulating native function calling."""
        self.seen.append(('chat', messages, tools))
        if messages[-1]['role'] == 'tool':
            return {'role': 'assistant', 'content': 'Done.'}
        import json
        name, arguments = self.calls.pop(0)
        return {'role': 'assistant', 'tool_calls': [{'id': f'call-{len(self.seen)}', 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(arguments)}}]}

    def text(self, instruction, data):
        """Return queued grounded text and record the exact job/resume context used."""
        self.seen.append(('text', instruction, data))
        return self.answers.pop(0)


def plan(agent, request='answer the latest question'):
    """Construct a valid supervisor handoff response for graph tests."""
    return {'tasks': [{'agent': agent, 'request': request}], 'reply': ''}
