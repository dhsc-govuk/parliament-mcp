from abc import ABC, abstractmethod

from langchain_aws import BedrockEmbeddings
from openai import AsyncAzureOpenAI

from parliament_mcp.settings import ParliamentMCPSettings


class EmbeddingsProvider(ABC):
    def __init__(self, settings: ParliamentMCPSettings):
        self.client = self._get_client(settings)

    @abstractmethod
    def _get_client(self, settings: ParliamentMCPSettings) -> AsyncAzureOpenAI | BedrockEmbeddings:
        """Getting the client."""

    @abstractmethod
    def embed_single(self) -> list[float]:
        "Embed a single text chunk."

    @abstractmethod
    def embed_batch(self) -> list[list[float]]:
        "Embed a batch of text chunks"
