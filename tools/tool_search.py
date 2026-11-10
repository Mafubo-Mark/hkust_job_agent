from urllib.parse import urlsplit
import httpx


def baidu_search(ctx, query):
    """Call Baidu Qianfan's real web-search endpoint and return source records, not model guesses."""
    key = ctx.env.get('QIANFAN_API_KEY')
    if not key:
        raise ValueError('Set QIANFAN_API_KEY in .env to enable web search.')
    header = ctx.env.get('QIANFAN_AUTH_HEADER') or 'Authorization'
    if header not in ('Authorization', 'X-Appbuilder-Authorization'):
        raise ValueError('Unsupported Qianfan authentication header.')
    response = httpx.post('https://qianfan.baidubce.com/v2/ai_search/web_search',
        headers={header: 'Bearer ' + key}, timeout=ctx.settings.request_timeout,
        json={'messages': [{'role': 'user', 'content': query}], 'search_source': 'baidu_search_v2',
              'resource_type_filter': [{'type': 'web', 'top_k': 10}]})
    response.raise_for_status()
    body = response.json()
    if body.get('code') not in (None, 0, '0'):
        raise ValueError('Qianfan rejected the search request; check credentials and service access.')
    if not isinstance(body.get('references', []), list):
        raise ValueError('Unexpected Qianfan search response.')
    return body.get('references', [])


def tool_search(ctx, User_question):
    """Search career/company topics and summarize snippets with verified source links."""
    plan = ctx.llm.json('Is this question about jobs, careers, employers, interviews, or HKUST career '
        'services? Return {"in_scope":boolean,"query":string}. Rewrite a concise web query of '
        'at most 72 units (ASCII character=1, other character=2). Do not include private resume data.',
        {'question': User_question})
    if plan.get('in_scope') is not True:
        return {'status': 'refused', 'answer': 'I can help with jobs, resumes, employers, and career services.'}
    query = plan.get('query', '')
    if not isinstance(query, str) or not query.strip() or sum(1 if ord(c) < 128 else 2 for c in query) > 72:
        raise ValueError('The search query must be concise (maximum 72 weighted characters).')
    sources = (ctx.search or baidu_search)(ctx, query)
    sources = [dict(title=s.get('title') or 'Source', url=s['url'], content=s.get('content') or '',
                    date=s.get('date') or '') for s in sources
               if isinstance(s, dict) and isinstance(s.get('url'), str) and
               urlsplit(s['url']).scheme in ('https', 'http') and urlsplit(s['url']).hostname]
    if not sources:
        return {'status': 'not_found', 'answer': 'The web search returned no usable sources.'}
    answer = ctx.llm.text('Answer using only the supplied search snippets. Use [1], [2], etc. for '
        'source references, never invent URLs. Clearly distinguish suggested interview preparation '
        'from verified company information. If snippets are insufficient, say so.',
        {'question': User_question, 'sources': sources})
    links = '\n'.join(f"[{i}] {s['title']}: {s['url']}" for i, s in enumerate(sources, 1))
    return {'status': 'ok', 'answer': answer + '\n\nSources:\n' + links, 'sources': sources}
