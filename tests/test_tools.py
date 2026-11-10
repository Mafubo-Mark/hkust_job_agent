import unittest
from unittest.mock import patch, Mock
from pydantic import ValidationError
from tests.helpers import fixture, ScriptedLLM
from tools.job_tools import execute
from tools.tool_summarize import units, tool_summarize
from tools.tool_return_jobs import tool_return_jobs
from tools.tool_job_resume_question import tool_job_resume_question
from tools.tool_find_required_jobs import tool_find_required_jobs
from tools.tool_search import tool_search, baidu_search


class ToolsTests(unittest.TestCase):
    def test_rank_range_and_exact_fields(self):
        """Ranges are 1-based inclusive, clamped to availability, and show true saved fields."""
        ctx = fixture(self)
        result = tool_return_jobs(ctx, 2, 20)
        self.assertEqual(result['displayed_job_ids'], ['job-1', 'job-2'])
        self.assertIn('AI Researcher', result['answer'])
        self.assertIn('REF-1', result['answer'])
        self.assertIn('94/100', result['answer'])
        for begin, finish in [(0, 2), (3, 2), (True, 3)]:
            with self.assertRaises(ValueError):
                tool_return_jobs(ctx, begin, finish)

    def test_argument_validation(self):
        """Model tool calls cannot inject unrecognized arguments or coerce strings into indices."""
        ctx = fixture(self)
        for values in ({'finish_index': '5'}, {'path': '/tmp/private'}, {'finish_index': True}):
            with self.assertRaises(ValidationError):
                execute(ctx, 'tool_return_jobs', values, {})

    def test_short_and_long_summary(self):
        """Short resumes bypass AI; long resumes are summarized and counted across languages."""
        llm = ScriptedLLM(answers=['Python developer. 机器学习。'])
        ctx = fixture(self, llm)
        self.assertEqual(tool_summarize(ctx, 'Python 开发')['units'], 3)
        self.assertEqual(len(llm.seen), 0)
        result = tool_summarize(ctx, 'experience ' * 501)
        self.assertLessEqual(result['units'], 500)
        self.assertEqual(ctx.resume(), result['answer'])
        before = ctx.resume()
        execute(ctx, 'tool_summarize', {'text': 'Other text'}, {})
        self.assertEqual(ctx.resume(), before)

    def test_oversized_summary_preserves_file(self):
        """Repeated model failures to obey the limit cannot replace the existing resume."""
        ctx = fixture(self, ScriptedLLM(answers=['word ' * 600] * 3))
        before = ctx.resume()
        with self.assertRaises(ValueError):
            tool_summarize(ctx, 'word ' * 501)
        self.assertEqual(ctx.resume(), before)

    def test_job_advice_uses_saved_evidence(self):
        """The advice model receives the selected full job, actual score, and saved resume."""
        llm = ScriptedLLM(answers=['Emphasize your Python project; do not invent experience.'])
        ctx = fixture(self, llm)
        result = tool_job_resume_question(ctx, 'it', 'How should I edit my resume?', state={'selected_job_id': 'job-1'})
        self.assertEqual(result['selected_job_id'], 'job-1')
        self.assertEqual(llm.seen[-1][2]['job']['ref_no'], 'REF-1')
        self.assertEqual(llm.seen[-1][2]['resume_summary'], ctx.resume())
        with self.assertRaises(ValueError):
            tool_job_resume_question(ctx, 'REF-1', 'why', resume_txt='invented credentials')

    def test_find_checks_every_job(self):
        """All 12 jobs are examined, including results beyond a usual top-eight RAG cutoff."""
        llm = ScriptedLLM(objects=[{'decision': 'match', 'reason': 'Python required', 'quotes': ['Python']} for _ in range(12)])
        result = tool_find_required_jobs(fixture(self, llm, 12), 'Python')
        self.assertEqual(result['checked'], 12)
        self.assertEqual(len(result['jobs']), 12)
        self.assertEqual(len(llm.seen), 0)

    def test_unsupported_requirement_quote_is_uncertain(self):
        """A fabricated quote cannot make a matching job or fail the whole search."""
        ctx = fixture(self, ScriptedLLM(objects=[{'decision': 'match', 'reason': 'high pay', 'quotes': ['$1 million']}]))
        result = tool_find_required_jobs(ctx, 'high pay')
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['jobs'], [])
        self.assertEqual(len(result['uncertain']), 3)

    def test_search_sources_and_scope(self):
        """Web answers contain actual source URLs and unrelated questions never invoke search."""
        llm = ScriptedLLM(objects=[{'in_scope': True, 'query': 'Huawei careers'}, {'in_scope': False, 'query': ''}],
                          answers=['Huawei has a careers page. [1]'])
        ctx = fixture(self, llm)
        ctx.search = Mock(return_value=[{'title': 'Careers', 'url': 'https://example.com/careers', 'content': 'Jobs'},
                                       {'title': 'bad', 'url': 'javascript:alert(1)'}])
        result = tool_search(ctx, 'Tell me about Huawei careers')
        self.assertEqual(len(result['sources']), 1)
        self.assertIn('https://example.com/careers', result['answer'])
        self.assertEqual(tool_search(ctx, 'Cook pasta')['status'], 'refused')
        ctx.search.assert_called_once()

    def test_qianfan_request_contract(self):
        """The provider adapter sends the documented endpoint, authentication, and request shape."""
        ctx = fixture(self)
        ctx.env['QIANFAN_API_KEY'] = 'synthetic-test-token'
        response = Mock()
        response.json.return_value = {'references': [{'title': 'A', 'url': 'https://example.com'}]}
        with patch('tools.tool_search.httpx.post', return_value=response) as post:
            self.assertEqual(len(baidu_search(ctx, 'Huawei careers')), 1)
        self.assertTrue(post.call_args.args[0].endswith('/v2/ai_search/web_search'))
        self.assertEqual(post.call_args.kwargs['json']['search_source'], 'baidu_search_v2')
        self.assertEqual(post.call_args.kwargs['headers'], {'Authorization': 'Bearer synthetic-test-token'})
