import os
import time
import json
from typing import Optional, Dict, Any, List, Union
from openai import OpenAI, APIError, RateLimitError, APITimeoutError, APIConnectionError

try:
    import weave

    HAS_WEAVE = True
except ImportError:
    weave = None
    HAS_WEAVE = False


if HAS_WEAVE:
    @weave.op()
    def _weave_create_chat_completion(client, kwargs: Dict[str, Any]) -> Any:
        return client.chat.completions.create(**kwargs)
else:
    def _weave_create_chat_completion(client, kwargs: Dict[str, Any]) -> Any:
        return client.chat.completions.create(**kwargs)

class Agent:
    """
    OpenAI-compatible LLM Agent wrapper.
    
    Features:
    - Automatic retry logic for Rate Limits (429) and Server Errors (5xx).
    - Supports JSON mode enforcement (useful for structured pruning decisions).
    - Exponential backoff for retries.
    """

    DEFAULT_MODELS = {
        "openai": "gpt-4o",
        "deepseek": "deepseek-v4-flash",
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.5,  # Paper uses 0.5
        max_output_tokens: int = 4096,
        max_retries: int = 3,
        base_delay_s: int = 2,
        api_key: Optional[str] = None,
        seed: Optional[int] = None,
        provider: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ):
        self.provider = (provider or os.getenv("AGENT_PROVIDER") or "openai").lower()
        if self.provider not in self.DEFAULT_MODELS:
            raise ValueError(
                f"unsupported agent provider {self.provider!r}; expected openai or deepseek"
            )
        self.model_name = model_name or self.DEFAULT_MODELS[self.provider]
        self.system_instruction = system_instruction
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.base_delay_s = base_delay_s
        self.seed = seed
        self.reasoning_effort = reasoning_effort or os.getenv("AGENT_REASONING_EFFORT")
        self.api_key_env = (
            "DEEPSEEK_API_KEY" if self.provider == "deepseek" else "OPENAI_API_KEY"
        )
        self.api_key = api_key or os.getenv(self.api_key_env)
        self.last_response_metadata = {}
        
        self._init_client()

    def _init_client(self):
        """Initializes the selected OpenAI-compatible client."""
        if not self.api_key:
            raise ValueError(
                f"{self.api_key_env} is not set. Set it as an environment variable "
                "or pass api_key to the constructor."
            )

        client_kwargs = {"api_key": self.api_key}
        if self.provider == "deepseek":
            client_kwargs["base_url"] = "https://api.deepseek.com"
        self._client = OpenAI(**client_kwargs)

    def send(self, message: str, json_mode: bool = False) -> str:
        """
        Sends a message to the LLM and retrieves the response.
        """
        messages = []
        
        # Prepare System Prompt
        sys_prompt = self.system_instruction or "You are a helpful assistant."
        
        # FIX 2: Safety check for JSON mode
        # OpenAI API requires the word "json" to appear in the prompt if response_format is json_object
        if json_mode and "json" not in sys_prompt.lower() and "json" not in message.lower():
            sys_prompt += " Please respond in valid JSON format."

        messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": message})

        # Prepare arguments
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }
        if self.seed is not None and self.provider == "openai":
            kwargs["seed"] = self.seed

        if self.provider == "deepseek":
            kwargs["max_tokens"] = self.max_output_tokens
            if self.reasoning_effort in {"low", "high", "max"}:
                kwargs["reasoning_effort"] = self.reasoning_effort
                kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            else:
                kwargs["temperature"] = self.temperature
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        elif self.model_name.startswith("gpt-5"):
            kwargs["max_completion_tokens"] = self.max_output_tokens
        else:
            kwargs["max_tokens"] = self.max_output_tokens
            kwargs["temperature"] = self.temperature

        # Enable JSON mode if requested
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # Retry Loop
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = _weave_create_chat_completion(self._client, kwargs)
                usage = getattr(response, "usage", None)
                self.last_response_metadata = {
                    "usage": usage.model_dump() if hasattr(usage, "model_dump") else usage if isinstance(usage, dict) else None,
                    "model": getattr(response, "model", self.model_name),
                    "finish_reason": getattr(response.choices[0], "finish_reason", None),
                }
                content = response.choices[0].message.content
                
                if content is None:
                    return ""
                
                return content

            except Exception as e:
                last_exception = e
                
                # Check if we should retry
                if attempt < self.max_retries and self._is_retryable(e):
                    # Exponential Backoff: base * 2^attempt
                    wait_time = self.base_delay_s * (2 ** attempt)
                    print(f"API Warning: {e}. Retrying {attempt + 1}/{self.max_retries} in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                # If it's the last attempt or a non-retryable error
                print(f"API Failed: {e}")
                raise last_exception

        return ""

    def _is_retryable(self, e: Exception) -> bool:
        """Determines if the exception is transient and worth retrying."""
        # Network/Timeout/RateLimit errors are always retryable
        if isinstance(e, (RateLimitError, APITimeoutError, APIConnectionError)):
            return True
        
        # FIX 3: Correct attribute for Status Code
        if isinstance(e, APIError):
            # Check .status_code (HTTP status), not .code (Error string)
            if e.status_code and isinstance(e.status_code, int) and e.status_code >= 500:
                return True
                
        return False

# ============================================================
# Usage Example
# ============================================================
if __name__ == "__main__":
    # Ensure you set your key before running: export OPENAI_API_KEY=<your key>
    try:
        agent = Agent(
            system_instruction="You are a helpful coding assistant.",
            temperature=0.7
        )
        
        print("--- Sending Standard Message ---")
        response = agent.send("Say hello!")
        print(f"Response: {response}\n")

        print("--- Sending JSON Mode Message ---")
        # Even if we forget to say "JSON" in the prompt, the fix code above handles it
        json_response = agent.send("List 3 colors.", json_mode=True)
        print(f"JSON Response: {json_response}")

    except Exception as e:
        print(f"Test failed with error: {e}")
