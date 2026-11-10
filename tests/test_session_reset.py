import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage
import main as entry
from tests.helpers import ScriptedLLM, fixture, plan
from tools.bye_tool import cleanup_generated, is_goodbye
from tools.tool_find_required_jobs import tool_find_required_jobs
from workflows.dialogue_graph import dialogue_session


class SessionResetTests(unittest.TestCase):
    def test_goodbye_routes_to_real_tool_and_preserves_input(self):
        """A farewell invokes bye_tool without an LLM and removes only known runtime outputs."""
        llm = ScriptedLLM()
        ctx = fixture(self, llm)
        (ctx.data / 'resume.pdf').write_bytes(b'%PDF-input')
        (ctx.data / 'conversations.sqlite').write_bytes(b'old chat')
        (ctx.data / 'conversations.sqlite-wal').write_bytes(b'old log')
        (ctx.data / 'model_cache').mkdir()
        (ctx.data / 'model_cache' / 'weights').write_bytes(b'cached model')
        self.assertTrue(is_goodbye("That's all, thanks!"))
        self.assertFalse(is_goodbye('Tell me about a goodbye email for a job'))
        with dialogue_session(ctx) as app:
            state = app.invoke({'messages': [HumanMessage(content="That's all, thanks!")]},
                               {'configurable': {'thread_id': 'current-run'}})
        self.assertEqual(state['turn_count'], 1)
        self.assertTrue(state['end_session'])
        self.assertEqual(state['tool_events'][0]['tool'], 'bye_tool')
        self.assertIn('Tool used: bye_tool', state['answer'])
        self.assertEqual(llm.seen, [])
        cleanup_generated(ctx)
        self.assertEqual([p.name for p in ctx.data.iterdir() if p.is_file()], ['resume.pdf'])
        self.assertTrue((ctx.data / 'model_cache' / 'weights').exists())
        self.assertFalse((ctx.data / 'vector_db').exists())

    def test_thirtieth_round_ends_session(self):
        """Rounds 1–29 continue; the 30th response closes the in-memory chat."""
        llm = ScriptedLLM(plans=[{'tasks': [], 'reply': 'I can help with careers.'} for _ in range(30)])
        ctx = fixture(self, llm)
        config = {'configurable': {'thread_id': 'current-run'}}
        with dialogue_session(ctx) as app:
            for number in range(1, 31):
                state = app.invoke({'messages': [HumanMessage(content='Hello')]}, config)
                self.assertEqual(state['turn_count'], number)
                self.assertEqual(state['end_session'], number == 30)
        self.assertIn('30 dialogue rounds reached', state['answer'])
        self.assertEqual(len(state['messages']), 60)

    def test_main_prepares_every_run_and_cleans_on_goodbye(self):
        """Two independent launches each request a full fresh preparation and discard outputs."""
        ctx = fixture(self)
        (ctx.data / 'resume.pdf').write_bytes(b'%PDF-input')
        names = ('resume.txt', 'jobs.json', 'jobs_nodes.json', 'result.json', 'prepared.json')
        content = {name: (ctx.data / name).read_bytes() for name in names}
        prepared = Mock()

        def refresh(context, pdf, mode, filters, force):
            """Recreate synthetic outputs after main removes the previous session's files."""
            self.assertEqual(Path(pdf), ctx.data / 'resume.pdf')
            self.assertEqual((mode, filters, force), ('full', None, True))
            for name, data in content.items():
                (ctx.data / name).write_bytes(data)
            return {'count': 3, 'data_dir': str(ctx.data)}

        prepared.side_effect = refresh
        with patch.object(entry, 'Context', return_value=ctx), patch.object(entry, 'prepare', prepared), \
             patch('builtins.input', return_value='goodbye'), redirect_stdout(io.StringIO()):
            self.assertEqual(entry.main([]), 0)
            self.assertEqual(entry.main([]), 0)
        self.assertEqual(prepared.call_count, 2)
        self.assertFalse((ctx.data / 'result.json').exists())
        self.assertTrue((ctx.data / 'resume.pdf').exists())

    def test_startup_failure_keeps_diagnostic_but_clears_generated_jobs(self):
        """A failed scrape leaves its error report readable after session cleanup."""
        ctx = fixture(self)
        (ctx.data / 'resume.pdf').write_bytes(b'%PDF-input')
        with patch.object(entry, 'Context', return_value=ctx), \
             patch.object(entry, 'prepare', side_effect=ValueError('Synthetic scrape failure')), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(entry.main([]), 1)
        self.assertIn('Synthetic scrape failure', (ctx.data / 'last_error.txt').read_text())
        self.assertFalse((ctx.data / 'jobs.json').exists())

    def test_skill_search_handles_source_variants_without_model_calls(self):
        """Vector search covers all records; skill matches return company/title without quote errors."""
        llm = ScriptedLLM()
        ctx = fixture(self, llm, count=12)
        result = tool_find_required_jobs(ctx, 'Find all jobs that require Python skill')
        self.assertEqual(result['checked'], 12)
        self.assertEqual(len(result['jobs']), 12)
        self.assertEqual(result['status'], 'ok')
        self.assertIn('Company Name:', result['answer'])
        self.assertIn('Job Title:', result['answer'])
        self.assertEqual(llm.seen, [])

    def test_python_question_uses_rag_tool_in_dialogue(self):
        """A full user turn returns grounded company/title matches instead of a tool failure."""
        llm = ScriptedLLM(plans=[plan('job_agent')],
                          calls=[('tool_find_required_jobs', {'User_requests': 'all jobs that require Python skill'})])
        ctx = fixture(self, llm, count=12)
        with dialogue_session(ctx) as app:
            state = app.invoke({'messages': [HumanMessage(content='Find all jobs that require Python skill')]},
                               {'configurable': {'thread_id': 'current-run'}})
        self.assertEqual(state['tool_events'][0]['status'], 'ok')
        self.assertEqual(state['tool_events'][0]['checked'], 12)
        self.assertIn('Tool used: tool_find_required_jobs', state['answer'])
        self.assertIn('Company Name:', state['answer'])
        self.assertIn('Job Title:', state['answer'])
