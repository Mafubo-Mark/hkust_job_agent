import unittest
from types import SimpleNamespace
from pydantic import BaseModel, ValidationError
from diagnostics import describe_error


class DiagnosticsTests(unittest.TestCase):
    def test_reason_and_cause_with_secret_redaction(self):
        """Show the actionable cause but never an API key or authorization token."""
        try:
            try:
                raise ValueError('Missing requirement; synthetic-secret; Bearer private-token')
            except ValueError as cause:
                raise ValueError('Invalid assessment for job-1') from cause
        except ValueError as error:
            text = describe_error(error, SimpleNamespace(env={'DEEPSEEK_API_KEY': 'synthetic-secret'}))
        self.assertIn('Missing requirement', text)
        self.assertIn('Invalid assessment', text)
        self.assertNotIn('synthetic-secret', text)
        self.assertNotIn('private-token', text)

    def test_validation_inputs_are_not_printed(self):
        """Pydantic diagnostics identify failed fields without echoing private source content."""
        class Example(BaseModel):
            score: int
        try:
            Example(score='private resume text')
        except ValidationError as error:
            text = describe_error(error)
        self.assertIn('score: int_parsing', text)
        self.assertNotIn('private resume text', text)
