"""Prepare fresh HKUST jobs and start a 30-round LangGraph conversation."""
import argparse
import json
from pathlib import Path
from langchain_core.messages import HumanMessage
from config import Context
from workflows.preparation_graph import prepare
from workflows.dialogue_graph import dialogue_session
from diagnostics import report_error
from tools.bye_tool import cleanup_generated


def main(argv=None):
    """Scan and score from scratch on every run, then clear generated files on exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', help='Searchable PDF; supplying it prepares jobs before chat')
    parser.add_argument('--filters-json', help='Existing HKUST dropdown values; {} skips filter menus')
    args = parser.parse_args(argv)
    ctx, failed = None, False
    try:
        ctx = Context(Path(__file__).resolve().parent)
        resume = Path(args.resume or ctx.data / 'resume.pdf').expanduser().resolve()
        if not resume.is_file():
            raise FileNotFoundError(f'Put a searchable PDF at {resume} or pass --resume.')
        filters = json.loads(args.filters_json) if args.filters_json is not None else None
        cleanup_generated(ctx)
        print('Preparing resume, fresh job scores, and semantic indexes…')
        result = prepare(ctx, str(resume), 'full', filters, True)
        print(f"Ready: {result['count']} jobs in {result['data_dir']}")
        ctx.ready()
        config = {'configurable': {'thread_id': 'current-run'}, 'recursion_limit': 64}
        print('Chat ready: up to 30 rounds. Say goodbye or /exit to finish and clear generated files.')
        with dialogue_session(ctx) as app:
            while True:
                question = input('You: ').strip()
                if question.casefold() in ('/exit', '/quit'):
                    question = 'goodbye'
                if question:
                    try:
                        state = app.invoke({'messages': [HumanMessage(content=question)]}, config)
                        print('\nAssistant: ' + state['answer'] + '\n')
                        if state.get('end_session'):
                            break
                    except Exception as error:
                        print('Could not finish this turn:')
                        report_error(error, ctx)
        return 0
    except (KeyboardInterrupt, EOFError):
        print('\nStopped. Clearing generated session files.')
        return 0
    except Exception as error:
        failed = True
        print('Startup failed:')
        report_error(error, ctx)
        return 1
    finally:
        if ctx is not None:
            if ctx._llm is not None and hasattr(ctx._llm, 'close'):
                ctx._llm.close()
            cleanup_generated(ctx, preserve_diagnostic=failed)


if __name__ == '__main__':
    raise SystemExit(main())
