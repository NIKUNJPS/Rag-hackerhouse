"""
One interface, three swappable backends. Groq is the fastest inference we can
get through an API, which matters a lot when part of the pipeline is budgeted
at 200ms; Claude/OpenAI are kept as drop-in fallbacks -- useful if Groq
rate-limits during the demo, or if you want to compare answer quality vs.
speed for the write-up.

Clients are created once per process and reused (module-level cache) instead
of per-call -- recreating an SDK client every request means a fresh TLS
handshake every request, which was costing 1-2.5s alone on top of actual
inference time.
"""

from backend.config import settings


class LLMError(Exception):
    pass


_clients: dict = {}


def _get_client(provider: str):
    if provider in _clients:
        return _clients[provider]

    if provider == "groq":
        from groq import Groq
        client = Groq(api_key=settings.GROQ_API_KEY)
    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    else:
        raise LLMError(f"Unknown LLM_PROVIDER: {provider}")

    _clients[provider] = client
    return client


def generate(prompt: str, system: str = "", max_tokens: int = 300) -> str:
    provider = settings.LLM_PROVIDER
    if provider == "groq":
        return _generate_groq(prompt, system, max_tokens)
    elif provider == "openai":
        return _generate_openai(prompt, system, max_tokens)
    elif provider == "anthropic":
        return _generate_anthropic(prompt, system, max_tokens)
    else:
        raise LLMError(f"Unknown LLM_PROVIDER: {provider}")


def _generate_groq(prompt, system, max_tokens):
    client = _get_client("groq")
    resp = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
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
