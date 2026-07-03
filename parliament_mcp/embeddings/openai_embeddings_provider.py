import logging
from itertools import batched

import httpx
from openai import AsyncAzureOpenAI
from tenacity import retry, stop_after_attempt

from parliament_mcp.embeddings.embeddings_provider import EmbeddingsProvider
from parliament_mcp.settings import OpenAIParliamentMCPSettings

logger = logging.getLogger(__name__)


class OpenAIEmbeddingsProvider(EmbeddingsProvider):
    def __init__(self, settings: OpenAIParliamentMCPSettings):
        super().__init__(settings=settings)
        self.model = settings.AZURE_OPENAI_EMBEDDING_MODEL
        self.dimensions = settings.EMBEDDING_DIMENSIONS

    def _get_client(self, settings: OpenAIParliamentMCPSettings) -> AsyncAzureOpenAI:
        """Get an async Azure OpenAI client."""
        return AsyncAzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            http_client=httpx.AsyncClient(timeout=30.0),
        )

    async def embed_single(self, text: str) -> list[float]:
        """Generate a single embedding for a text using Azure OpenAI."""
        response = await self.client.embeddings.create(
            input=text,
            model=self.model,
            dimensions=self.dimensions,
        )
        return response.data[0].embedding

    @retry(stop=stop_after_attempt(3))
    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 100,
    ) -> list[list[float]]:
        """Generate embeddings for a list of texts using Azure OpenAI.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each API call

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        for i, batch in enumerate(batched(texts, batch_size)):
            try:
                response = await self.client.embeddings.create(
                    input=batch,
                    model=self.model,
                    dimensions=self.dimensions,
                )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            except Exception:
                logger.exception("Error generating embeddings for batch %d", i // batch_size + 1)
                raise

        return all_embeddings
