"""
OpenAI-compatible client wrapper for local MLX LLM server.

Handles requests to the local mlx_lm.server (defaulting to localhost:8080),
implements response caching, and sets custom headers/template options.
"""

import logging
import hashlib
import json
import sqlite3
import time
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Iterator
from openai import OpenAI

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "cache"
LLM_CACHE_DB = CACHE_DIR / "llm_cache.db"

from llm.server_config import BASE_URL, PORT

class LLMClient:
    """Wrapper for local OpenAI-compatible LLM server (LM Studio / MLX) with caching."""

    def __init__(self, base_url: str = None, api_key: str = "EMPTY"):
        url = base_url or BASE_URL
        self.client = OpenAI(base_url=url, api_key=api_key)
        self.model_name = "mlx-community/Qwen3.5-9B-MLX-4bit" # default fallback
        self._init_cache()
        self._detect_active_model()

    def _detect_active_model(self):
        """Retrieve the active loaded model from the local server to avoid name conflicts."""
        try:
            models = self.client.models.list()
            if models.data:
                # Use the first loaded model name (e.g. from LM Studio)
                self.model_name = models.data[0].id
                logger.info(f"Dynamically detected active model: '{self.model_name}'")
        except Exception as e:
            logger.warning(f"Could not dynamically query active models: {e}. Using default fallback '{self.model_name}'.")


    def _init_cache(self):
        """Initialize the SQLite database for LLM query caching."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(LLM_CACHE_DB) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS response_cache (
                    hash TEXT PRIMARY KEY,
                    messages TEXT NOT NULL,
                    response TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            conn.commit()

    def _compute_hash(self, messages: List[Dict[str, str]], kwargs: Dict[str, Any]) -> str:
        """Compute MD5 hash of messages and arguments for caching."""
        data = {
            "messages": messages,
            "kwargs": {k: v for k, v in kwargs.items() if k != "stream"}
        }
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.md5(serialized.encode("utf-8")).hexdigest()

    def _get_cached_response(self, cache_hash: str) -> Optional[str]:
        """Retrieve a response from cache if exists."""
        try:
            with sqlite3.connect(LLM_CACHE_DB) as conn:
                row = conn.execute(
                    "SELECT response FROM response_cache WHERE hash = ?", (cache_hash,)
                ).fetchone()
            if row:
                logger.info("Retrieved cached response for identical query.")
                return row[0]
        except Exception as e:
            logger.error(f"Error reading LLM cache: {e}")
        return None

    def _save_to_cache(self, cache_hash: str, messages: List[Dict[str, str]], response: str):
        """Save a query response to the cache."""
        try:
            with sqlite3.connect(LLM_CACHE_DB) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO response_cache (hash, messages, response, timestamp) VALUES (?, ?, ?, ?)",
                    (cache_hash, json.dumps(messages), response, time.time())
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving to LLM cache: {e}")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
        enable_thinking: bool = False,
        bypass_cache: bool = False,
        stream: bool = False,
    ) -> Union[str, Iterator[str]]:
        """
        Generate completion from the MLX server.

        Args:
            messages: List of message dicts (role, content)
            temperature: LLM temperature
            max_tokens: Maximum tokens to generate
            enable_thinking: Toggle thinking model features if supported by server template kwargs
            bypass_cache: Force fresh LLM generation
            stream: Stream the response back chunk by chunk
        """
        # Extra arguments for Qwen thinking-mode toggle
        extra_body = {
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking
            }
        }
        
        # Build API kwargs
        api_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": extra_body,
        }

        # Handle caching (only for non-streaming calls)
        cache_hash = ""
        if not stream and not bypass_cache:
            cache_hash = self._compute_hash(messages, api_kwargs)
            cached = self._get_cached_response(cache_hash)
            if cached is not None:
                return cached

        # Execute API call
        try:
            if stream:
                response = self.client.chat.completions.create(
                    stream=True,
                    **api_kwargs
                )
                def generator():
                    full_resp = []
                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            full_resp.append(content)
                            yield content
                    
                    # Cache the full response after streaming finishes if needed (not cached by default, but we print/log)
                    logger.debug("Finished streaming LLM response.")
                return generator()
            else:
                response = self.client.chat.completions.create(
                    stream=False,
                    **api_kwargs
                )
                content = response.choices[0].message.content or ""
                
                # Strip thinking tags if any leaked in non-thinking mode (e.g. empty <think></think> tags)
                if not enable_thinking:
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                
                if not bypass_cache:
                    self._save_to_cache(cache_hash, messages, content)
                return content
                
        except Exception as e:
            logger.error(f"Error calling local MLX server: {e}")
            raise ConnectionError(
                "Could not connect to the local MLX server. Is it running on http://localhost:8080? "
                "Ensure you run start.sh or start the server via 'python -m mlx_lm.server'."
            ) from e
            

