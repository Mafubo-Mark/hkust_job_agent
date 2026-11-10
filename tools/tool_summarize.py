import re
from pathlib import Path


def units(text):
    """Count each Chinese character and each other word as one summary unit."""
    return len(re.findall(r'[\u3400-\u9fff]|[^\W_]+(?:[+.#-][^\W_]+)*',
                          re.sub(r'([\u3400-\u9fff])', r' \1 ', text), re.UNICODE))


def tool_summarize(ctx, text, save=True):
    """Save a factual resume summary of at most 1,000 units."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Resume text must not be empty.')

    answer = text.strip()
    for attempt in range(4):
        count = units(answer)
        if count <= 1000:
            break

        answer = ctx.llm.text(
            f'This text is {count} units long. Shorten it to at most 800 units. '
            'Count each Chinese character and each English word as one unit. '
            'Preserve key skills, qualifications, experience, dates, and achievements. '
            'Do not invent facts. Return only the summary.',
            {'text_to_summarize': answer, 'attempt': attempt + 1}
        ).strip()

    if not answer or units(answer) > 1000:
        raise ValueError('DeepSeek could not produce a summary within 1,000 units.')

    if save:
        path = ctx.data / 'resume.txt'
        temporary = path.with_suffix('.txt.tmp')
        try:
            temporary.write_text(answer, encoding='utf-8')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    return {'status': 'ok', 'answer': answer, 'units': units(answer)}