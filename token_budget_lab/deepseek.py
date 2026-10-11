"""Chat usage normalization and redirect policy."""
import urllib.request

ENDPOINT = "https://api.deepseek.com/chat/completions"

def normalize_usage(usage, protocol):
    """Anthropic input_tokens excludes cache reads/writes; Chat totals include hits."""
    usage = dict(usage)
    if protocol == 'anthropic':
        fields = ('input_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens', 'output_tokens')
        values = {f: usage.get(f, 0 if f.startswith('cache_') else None) for f in fields}
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in values.values()):
            raise RuntimeError('Response missing valid usage; token totals unavailable')
        usage.update(prompt_tokens=sum(values[f] for f in fields[:-1]),
                     completion_tokens=values['output_tokens'],
                     prompt_cache_hit_tokens=values['cache_read_input_tokens'],
                     prompt_cache_miss_tokens=values['input_tokens']+values['cache_creation_input_tokens'])
    for field in ('prompt_tokens', 'completion_tokens'):
        if not isinstance(usage.get(field), int) or isinstance(usage[field], bool) or usage[field] < 0:
            raise RuntimeError('Response missing valid usage; token totals unavailable')
    return usage


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("API redirect rejected; check the official endpoint")

