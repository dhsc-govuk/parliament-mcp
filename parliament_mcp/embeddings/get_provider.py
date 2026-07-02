import logging

from parliament_mcp.embeddings.bedrock_embeddings_provider import BedrockEmbeddingsProvider
from parliament_mcp.embeddings.embeddings_provider import EmbeddingsProvider
from parliament_mcp.embeddings.openai_embeddings_provider import OpenAIEmbeddingsProvider
from parliament_mcp.settings import BedrockParliamentMCPSettings, OpenAIParliamentMCPSettings, ParliamentMCPSettings

logger = logging.getLogger(__name__)


class EmbeddingProviderError(Exception):
    pass


def get_provider(settings: ParliamentMCPSettings) -> EmbeddingsProvider:
    if isinstance(settings, OpenAIParliamentMCPSettings):
        provider = OpenAIEmbeddingsProvider(settings)
    elif isinstance(settings, BedrockParliamentMCPSettings):
        provider = BedrockEmbeddingsProvider(settings)
    else:
        raise EmbeddingProviderError

    return provider
