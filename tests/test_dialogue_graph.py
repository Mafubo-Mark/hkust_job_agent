import unittest
import json
import subprocess
import sys
from unittest.mock import Mock
from langchain_core.messages import HumanMessage
from workflows.dialogue_graph import dialogue_session
from tests.helpers import fixture, ScriptedLLM, plan
from job_matcher_tool import read_json, write_json
from rag.build_index import mark_ready


class DialogueTests(unittest.TestCase):
    def test_multiturn_is_in_memory_only(self):
        """Resolve follow-ups within a session and forget them after its checkpointer closes."""
        llm = ScriptedLLM(plans=[plan('job_agent'), plan('resume_agent'), plan('resume_agent'),
                                plan('company_search_agent'), plan('resume_agent')],
            calls=[('tool_return_jobs', {'finish_index': 10}),
                   ('tool_job_resume_question', {'user_input_job_name': 'the second one', 'user_query': 'Why do I fit?'}),
                   ('tool_job_resume_question', {'user_input_job_name': 'it', 'user_query': 'Improve my resume'}),
                   ('tool_search', {'User_question': 'Tell me about Huawei'}),
                   ('tool_job_resume_question', {'user_input_job_name': 'it', 'user_query': 'Why do I fit?'})],
            answers=['Your Python project supports the Python requirement.', 'Emphasize your real Python project.',
                     'Huawei company information [1].'], objects=[{'in_scope': True, 'query': 'Huawei company careers'}])
        ctx = fixture(self, llm)
        ctx.search = lambda *_: [{'title': 'Company', 'url': 'https://example.com/company', 'content': 'Huawei info'}]
        config = {'configurable': {'thread_id': 'alice'}, 'recursion_limit': 64}
        with dialogue_session(ctx) as app:
            first = app.invoke({'messages': [HumanMessage(content='Show my top 10 jobs')]}, config)
            self.assertEqual(first['displayed_job_ids'], ['job-0', 'job-1', 'job-2'])
            second = app.invoke({'messages': [HumanMessage(content='Why do I fit the second one?')]}, config)
            self.assertEqual(second['selected_job_id'], 'job-1')
            self.assertIn('REF-1', second['answer'])
            self.assertEqual(len(second['messages']), 4)
            third = app.invoke({'messages': [HumanMessage(content='How should I modify my resume for it?')]}, config)
            self.assertEqual(third['selected_job_id'], 'job-1')
            self.assertEqual(len(third['messages']), 6)
            fourth = app.invoke({'messages': [HumanMessage(content='Tell me about that company')]}, config)
            self.assertIn('Tool used: tool_search', fourth['answer'])
            self.assertIn('https://example.com/company', fourth['answer'])
        child = subprocess.run([sys.executable, '-c',
            'import sys,json; from config import Context; from workflows.dialogue_graph import dialogue_session; '
            'ctx=Context(sys.argv[1]); session=dialogue_session(ctx); app=session.__enter__(); '
            'print(json.dumps(app.get_state({"configurable":{"thread_id":"alice"}}).values)); '
            'session.__exit__(None,None,None)', str(ctx.root)], capture_output=True, text=True)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(json.loads(child.stdout), {})
        self.assertFalse((ctx.data / 'conversations.sqlite').exists())
        with dialogue_session(ctx) as app:
            other = app.invoke({'messages': [HumanMessage(content='Why do I fit it?')]}, config)
            self.assertIsNone(other.get('selected_job_id'))
            self.assertIn('Which job?', other['answer'])

    def test_changed_data_clears_referents(self):
        """A new resume invalidates previous selected jobs and history on the same thread."""
        llm = ScriptedLLM(plans=[plan('job_agent'), plan('resume_agent')], calls=[
            ('tool_return_jobs', {'finish_index': 2}),
            ('tool_job_resume_question', {'user_input_job_name': 'the second one', 'user_query': 'why'})])
        ctx = fixture(self, llm)
        config = {'configurable': {'thread_id': 'same'}}
        with dialogue_session(ctx) as app:
            app.invoke({'messages': [HumanMessage(content='top 2')]}, config)
            (ctx.data / 'resume.txt').write_text('New resume with Java projects.')
            (ctx.data / 'prepared.json').unlink()
            mark_ready(ctx)
            result = app.invoke({'messages': [HumanMessage(content='why second?')]}, config)
            self.assertEqual(result['displayed_job_ids'], [])
            self.assertEqual(len(result['messages']), 2)
            self.assertIn('not in the last displayed list', result['answer'])

    def test_refusal_and_tools_are_real(self):
        """An out-of-scope reply requires no tool and cannot show a fabricated tool-use label."""
        llm = ScriptedLLM(plans=[{'tasks': [], 'reply': 'I can help with jobs and careers.'}])
        ctx = fixture(self, llm)
        with dialogue_session(ctx) as app:
            result = app.invoke({'messages': [HumanMessage(content='Write a cake recipe')]},
                                {'configurable': {'thread_id': 'scope'}})
        self.assertEqual(result['tool_events'], [])
        self.assertNotIn('Tool used:', result['answer'])

    def test_multiple_specialists(self):
        """One supervisor plan can hand off to two independent specialists in order."""
        llm = ScriptedLLM(plans=[{'tasks': [plan('job_agent')['tasks'][0], plan('company_search_agent')['tasks'][0]], 'reply': ''}],
            calls=[('tool_return_jobs', {'finish_index': 1}), ('tool_search', {'User_question': 'Huawei careers'})],
            objects=[{'in_scope': True, 'query': 'Huawei careers'}], answers=['Careers page [1]'])
        ctx = fixture(self, llm)
        ctx.search = lambda *_: [{'title': 'Careers', 'url': 'https://example.com', 'content': 'Careers'}]
        with dialogue_session(ctx) as app:
            result = app.invoke({'messages': [HumanMessage(content='Top job and Huawei company information')]},
                                {'configurable': {'thread_id': 'combined'}})
        self.assertEqual([e['tool'] for e in result['tool_events']], ['tool_return_jobs', 'tool_search'])

    def test_bounded_repeated_tool_calls(self):
        """A model that keeps calling tools stops after eight decisions rather than looping forever."""
        llm = ScriptedLLM(plans=[plan('job_agent')])
        llm.chat = Mock(return_value={'role': 'assistant', 'tool_calls': [
            {'id': 'repeat', 'type': 'function', 'function': {'name': 'tool_return_jobs', 'arguments': '{"finish_index":1}'}}]})
        ctx = fixture(self, llm)
        with dialogue_session(ctx) as app:
            result = app.invoke({'messages': [HumanMessage(content='top job')]},
                {'configurable': {'thread_id': 'bounded'}, 'recursion_limit': 64})
        self.assertEqual(llm.chat.call_count, 8)
        self.assertIn('limit reached', result['answer'])

    def test_invalid_tool_arguments_report_failure(self):
        """A malformed model call yields an honest error, with no ranking or invented success."""
        llm = ScriptedLLM(plans=[plan('job_agent')], calls=[('tool_return_jobs', {'finish_index': 'ten'})])
        ctx = fixture(self, llm)
        with dialogue_session(ctx) as app:
            result = app.invoke({'messages': [HumanMessage(content='top ten')]},
                                {'configurable': {'thread_id': 'bad-args'}})
        self.assertEqual(result['tool_events'][0]['status'], 'error')
        self.assertEqual(result['displayed_job_ids'], [])
        self.assertIn('No result was fabricated', result['answer'])

    def test_specialist_tool_allowlist(self):
        """A resume specialist cannot invoke an unapproved search tool."""
        llm = ScriptedLLM(plans=[plan('resume_agent')], calls=[('tool_search', {'User_question': 'anything'})])
        ctx = fixture(self, llm)
        with dialogue_session(ctx) as app:
            with self.assertRaises(ValueError):
                app.invoke({'messages': [HumanMessage(content='help')]}, {'configurable': {'thread_id': 'allowlist'}})
