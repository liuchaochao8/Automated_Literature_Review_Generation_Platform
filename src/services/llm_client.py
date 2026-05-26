from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import AsyncOpenAI
from src.config.settings import settings


class LLMClient:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.provider = provider or settings.llm_provider
        self.model = model or settings.llm_model
        self._client = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
            )
        return self._client

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        if not self.provider:
            raise ValueError("LLM provider not configured")

        if self.provider == "deepseek":
            return await self._call_deepseek(system_prompt, user_prompt, max_tokens, temperature)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def _call_deepseek(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""
