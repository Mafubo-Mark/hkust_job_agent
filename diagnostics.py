"""Actionable error reports without API keys, provider response bodies, or resume inputs."""
import re
from pathlib import Path
from pydantic import ValidationError


def describe_error(error, ctx=None):
    """Explain local validation failures and their causes while masking configured secrets."""
    lines, seen = [], set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ValidationError):
            detail = '; '.join('.'.join(map(str, e['loc'])) + ': ' + e['type']
                               for e in current.errors(include_input=False, include_context=False))
        elif type(current) in (ValueError, TypeError, FileNotFoundError):
            detail = str(current)
        else:
            status = getattr(current, 'status_code', None)
            detail = f'HTTP status {status}; check provider access and connectivity.' if status else 'See the failure location below.'
        lines.append(f'{type(current).__name__}: {detail}')
        frames = current.__traceback__
        while frames and frames.tb_next:
            frames = frames.tb_next
        if frames:
            lines.append(f'Location: {Path(frames.tb_frame.f_code.co_filename).name}:{frames.tb_lineno} ({frames.tb_frame.f_code.co_name})')
        current = current.__cause__ or current.__context__
    text = '\n'.join(lines)
    if ctx is not None:
        for name, value in ctx.env.items():
            if value and any(word in name.upper() for word in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD')):
                text = text.replace(str(value), '[REDACTED]')
    return re.sub(r'(?i)Bearer\s+\S+', 'Bearer [REDACTED]', text)


def report_error(error, ctx=None):
    """Print a safe diagnostic and save it locally so failed runs can be investigated."""
    text = describe_error(error, ctx)
    print(text)
    if ctx is not None:
        try:
            (ctx.data / 'last_error.txt').write_text(text + '\n', encoding='utf-8')
            print('Diagnostic saved in data/last_error.txt.')
        except OSError:
            pass
