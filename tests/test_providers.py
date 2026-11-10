import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from agents.llm import DeepSeek
from job_matcher_tool import Settings


class ProviderTests(unittest.TestCase):
    def test_deepseek_native_calls_and_json_contract(self):
        """Native tool schemas, JSON mode, and disabled thinking reach the real SDK adapter."""
        client, message = Mock(), Mock()
        message.model_dump.return_value = {'role': 'assistant', 'content': '{"ok":true}'}
        client.chat.completions.create.return_value = SimpleNamespace(choices=[
            SimpleNamespace(finish_reason='stop', message=message)])
        with patch('agents.llm.OpenAI', return_value=client):
            llm = DeepSeek(Settings(api_key='synthetic-test-token', model='mock'))
        llm.chat([{'role': 'user', 'content': 'top jobs'}], tools=[{'type': 'function', 'function': {'name': 'tool_return_jobs'}}])
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs['tool_choice'], 'auto')
        self.assertEqual(kwargs['extra_body'], {'thinking': {'type': 'disabled'}})
        self.assertEqual(llm.json('Return an object', {}), {'ok': True})
        self.assertEqual(client.chat.completions.create.call_args.kwargs['response_format'], {'type': 'json_object'})
        llm.close()
        client.close.assert_called_once()

    def test_truncated_response_rejected(self):
        """Incomplete provider output cannot become a successful agent answer."""
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(finish_reason='length')])
        with patch('agents.llm.OpenAI', return_value=client):
            llm = DeepSeek(Settings(api_key='synthetic-test-token', model='mock'))
        with self.assertRaises(ValueError):
            llm.chat([])
