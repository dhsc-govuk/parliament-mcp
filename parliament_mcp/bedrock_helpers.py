import logging
from itertools import batched

import boto3
from botocore.config import Config
from langchain_aws import BedrockEmbeddings
from tenacity import retry, stop_after_attempt

from parliament_mcp.settings import ParliamentMCPSettings

logger = logging.getLogger(__name__)


def get_bedrock_client(settings: ParliamentMCPSettings) -> BedrockEmbeddings:
    """Get an async Bedrock Embeddings client."""
    config = Config(connect_timeout=30, read_timeout=30, retries={"max_attempts": 3})

    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name=settings.AWS_REGION,
        config=config,
    )

    return BedrockEmbeddings(
        client=bedrock_client,
        # i.e. "amazon.nova-2-multimodal-embeddings-v1:0"
        # "amazon.titan-embed-text-v1"
        model_id=settings.BEDROCK_EMBEDDINGS_MODEL,
        dimensions=settings.EMBEDDING_DIMENSIONS,
    )


async def embed_single(
    client: BedrockEmbeddings,
    text: str,
) -> list[float]:
    """Generate a single embedding for a text using Bedrock Embeddings.

    Args:
        client: BedrockEmbeddings client
        text: List of texts to embed

    Returns:
        Embedding vector.
    """
    return await client.aembed_query(text=text)


@retry(stop=stop_after_attempt(3))
async def embed_batch(
    client: BedrockEmbeddings,
    texts: list[str],
    batch_size: int = 100,
) -> list[list[float]]:
    """Generate embeddings for a list of texts using Bedrock Embeddings.

    Args:
        client: BedrockEmbeddings client
        texts: List of texts to embed
        batch_size: Number of texts to process in each API call

    Returns:
        List of embedding vectors
    """
    all_embeddings = []

    for i, batch in enumerate(batched(texts, batch_size)):
        try:
            response = await client.aembed_documents(texts=batch)
            all_embeddings.extend(response)

        except Exception:
            logger.exception("Error generating embeddings for batch %d", i // batch_size + 1)
            raise

    return all_embeddings
