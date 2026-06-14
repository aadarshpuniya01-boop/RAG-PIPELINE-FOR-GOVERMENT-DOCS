import httpx

from legal_rag.config import get_settings


class LLMClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def complete(self, prompt: str) -> str:
        if not (self.settings.llm_base_url and self.settings.llm_api_key and self.settings.llm_model):
            return self._extractive_fallback(prompt)

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.settings.llm_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {self.settings.llm_api_key}"},
                json={
                    "model": self.settings.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _extractive_fallback(prompt: str) -> str:
        lines = [line.strip() for line in prompt.splitlines() if line.strip()]
        evidence = [line for line in lines if line.startswith("[") or "Circular" in line or "Notification" in line]
        if not evidence:
            return "I could not find enough retrieved context to answer confidently."
        return "\n".join(["### Extractive Answer", *[f"- {line[:500]}" for line in evidence[:8]]])
