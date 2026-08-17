"""
One interface, three swappable backends, with automatic failover.

Groq is the fastest inference available (confirmed directly: 390-650ms full
answer vs. ~2-2.3s for Claude Haiku on the same prompts) -- but its free tier
caps at 8000 tokens/minute, and a RAG call burns 400-800 tokens (prompt +
hidden reasoning + answer). Measured directly: a rapid 40-query loop blew
through that budget after ~10-15 queries, and Groq's SDK started retrying
internally for several seconds per call instead of failing fast -- median
latency went from ~500ms to 4.6s, and grounded-answer rate dropped from ~95%
to 55%, entirely from truncated/degraded responses during those internal
retries. That's the exact failure mode an automated eval loop firing 10+
queries back-to-back would trigger.

The fix: disable each SDK's own internal retry (max_retries=0) so a failure
surfaces immediately instead of silently eating seconds, and fail over to
LLM_FALLBACK_PROVIDER the instant the primary provider's *first* token fails
to arrive. Falling back only before any content has been yielded (never
mid-stream) means the answer is never a garbled mix of two providers' output.

Clients are created once per process and reused (module-level cache) instead
of per-call -- recreating an SDK client every request means a fresh TLS
handshake every request, which was costing 1-2.5s alone on top of actual
inference time.
"""

from backend.config import settings


class LLMError(Exception):
    pass


_clients: dict = {}

# SDK defaults are generous (often 10min, plus their own internal retries) --
# fine for a human waiting, bad for an automated eval loop, where a struggling
# provider should fail fast so our own fallback logic can take over instead
# of silently eating several seconds per call inside the SDK.
_CLIENT_TIMEOUT_S = 20.0


def _get_client(provider: str):
    if provider in _clients:
        return _clients[provider]

    if provider == "groq":
        from groq import Groq
        client = Groq(api_key=settings.GROQ_API_KEY, timeout=_CLIENT_TIMEOUT_S, max_retries=0)
    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY, timeout=_CLIENT_TIMEOUT_S, max_retries=0)
    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY, timeout=_CLIENT_TIMEOUT_S, max_retries=0)
    else:
        raise LLMError(f"Unknown LLM_PROVIDER: {provider}")

    _clients[provider] = client
    return client


def _groq_extra_kwargs() -> dict:
    """
    Groq's `gpt-oss` models are reasoning models: by default they spend the
    entire max_tokens budget on a hidden chain-of-thought (exposed as a
    separate `message.reasoning` field, not `content`) and return an EMPTY
    answer once the budget runs out -- confirmed directly against the API
    (default settings: 76 of 86 tokens went to reasoning for a two-word
    reply; on our actual RAG prompt it used 100% of a 250-token budget on
    reasoning and returned nothing). `reasoning_effort="low"` caps that,
    which is what makes gpt-oss usable at RAG-answer length/latency at all.
    """
    if "gpt-oss" in settings.GROQ_MODEL:
        return {"reasoning_effort": "low"}
    return {}


def _stream_groq(prompt, system, max_tokens):
    client = _get_client("groq")
    stream = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
        stream=True,
        **_groq_extra_kwargs(),
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _stream_openai(prompt, system, max_tokens):
    client = _get_client("openai")
    stream = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _stream_anthropic(prompt, system, max_tokens):
    client = _get_client("anthropic")
    with client.messages.stream(
        model=settings.ANTHROPIC_MODEL,
        system=system,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text


def _generate_groq(prompt, system, max_tokens):
    client = _get_client("groq")
    resp = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
        **_groq_extra_kwargs(),
    )
    return resp.choices[0].message.content.strip()


def _generate_openai(prompt, system, max_tokens):
    client = _get_client("openai")
    resp = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def _generate_anthropic(prompt, system, max_tokens):
    client = _get_client("anthropic")
    resp = client.messages.create(
        model=settings.ANTHROPIC_MODEL,
        system=system,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


_STREAM_FNS = {"groq": _stream_groq, "openai": _stream_openai, "anthropic": _stream_anthropic}
_GENERATE_FNS = {"groq": _generate_groq, "openai": _generate_openai, "anthropic": _generate_anthropic}


def _provider_configured(provider: str) -> bool:
    key = {"groq": settings.GROQ_API_KEY, "openai": settings.OPENAI_API_KEY,
           "anthropic": settings.ANTHROPIC_API_KEY}.get(provider, "")
    return bool(key)


def generate(prompt: str, system: str = "", max_tokens: int = 300) -> str:
    provider = settings.LLM_PROVIDER
    fn = _GENERATE_FNS.get(provider)
    if fn is None:
        raise LLMError(f"Unknown LLM_PROVIDER: {provider}")
    try:
        return fn(prompt, system, max_tokens)
    except Exception:
        fallback = settings.LLM_FALLBACK_PROVIDER
        if fallback and fallback != provider and _provider_configured(fallback):
            return _GENERATE_FNS[fallback](prompt, system, max_tokens)
        raise


def generate_stream(prompt: str, system: str = "", max_tokens: int = 300):
    """
    Yields text chunks as they arrive instead of blocking for the full
    completion. The point isn't a nicer UX -- it's that "time to first
    token" (how long until the model starts responding) is the only
    LLM-latency number that can honestly approach 200ms. Full-completion
    time scales with answer length no matter how fast the provider is, so
    reporting only that number and calling it "under 200ms" would be
    misleading; TTFT is the real, defensible fast number.

    Falls over to LLM_FALLBACK_PROVIDER if the primary provider fails before
    yielding any content (pulling the first chunk is what actually forces the
    API call to execute) -- never mid-stream, so an answer is never a
    garbled mix of two providers' output.
    """
    provider = settings.LLM_PROVIDER
    fn = _STREAM_FNS.get(provider)
    if fn is None:
        raise LLMError(f"Unknown LLM_PROVIDER: {provider}")

    gen = fn(prompt, system, max_tokens)
    try:
        first = next(gen)
    except StopIteration:
        return
    except Exception:
        fallback = settings.LLM_FALLBACK_PROVIDER
        if fallback and fallback != provider and _provider_configured(fallback):
            yield from _STREAM_FNS[fallback](prompt, system, max_tokens)
            return
        raise
    yield first
    yield from gen
