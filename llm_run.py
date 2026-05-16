"""Minimal async OpenAI chat client with retry/backoff.

This file is missing from Riedl's upstream repo (referenced in experiment.py
and persona_experiment.py as `from llm_run import chat`, but no llm_run.py
exists in the GitHub release). This is our reconstruction of the expected
interface based on usage at the call sites.

Behavior matches Riedl's README description: "API client with retry logic"
and "Exponential backoff with jitter".
"""

import asyncio
import os
import random
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError

# Load OPENAI_API_KEY from .env at import time.
load_dotenv()

# Module-level client; OpenAI SDK is safe to reuse across coroutines.
_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Place it in AI-MACS/.env as: "
                "OPENAI_API_KEY=sk-..."
            )
        _client = AsyncOpenAI(api_key=api_key)
    return _client


async def chat(
    model: str,
    prompt: str,
    temperature: float = 1.0,
    max_tokens: int = 200,
    max_retries: int = 5,
    base_delay: float = 1.0,
) -> str:
    """Send a single-turn user prompt and return the response text.

    Retries on RateLimitError / APITimeoutError / generic APIError with
    exponential backoff + jitter. Bubbles up other exceptions.
    """
    client = _get_client()
    last_err: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            return content if content is not None else ""
        except (RateLimitError, APITimeoutError, APIError) as e:
            last_err = e
            if attempt == max_retries - 1:
                break
            # Exponential backoff with jitter
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(delay)

    # Exhausted retries
    raise RuntimeError(
        f"chat() failed after {max_retries} attempts. Last error: {last_err!r}"
    )
