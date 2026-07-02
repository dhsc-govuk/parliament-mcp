import logging
import os
from functools import lru_cache

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class InvalidLLMProviderError(Exception):
    pass


@lru_cache
def get_ssm_parameter(parameter_name: str, region: str = "eu-west-2") -> str:
    """Fetch a parameter from AWS Systems Manager Parameter Store."""
    try:
        ssm = boto3.client("ssm", region_name=region)
        response = ssm.get_parameter(Name=parameter_name, WithDecryption=True)
        return response["Parameter"]["Value"]
    except (ClientError, BotoCoreError) as e:
        logger.warning("Could not fetch SSM parameter %s: %s", parameter_name, e)
        return ""


def get_environment_or_ssm(env_var_name: str, ssm_path: str | None = None, default: str = "") -> str:
    """Get value from environment variable or fall back to SSM parameter."""
    env_value = os.environ.get(env_var_name)
    if env_value:
        return env_value

    # Only use SSM if not in local environment
    environment = os.environ.get("ENVIRONMENT", "local")
    if ssm_path and os.environ.get("AWS_REGION") and environment != "local":
        return get_ssm_parameter(ssm_path, os.environ.get("AWS_REGION"))

    return default


class ParliamentMCPSettings(BaseSettings):
    """Configuration settings for Parliament MCP application with environment-based loading."""

    AWS_ACCOUNT_ID: str | None = None
    AWS_REGION: str = "eu-west-2"
    ENVIRONMENT: str = "local"

    def _get_project_name(self) -> str:
        """Get the project name from environment or use default."""
        return os.environ.get("PROJECT_NAME", "i-dot-ai-dev-parliament-mcp")

    # Use SSM for sensitive parameters in AWS environments
    @property
    def SENTRY_DSN(self) -> str | None:
        return get_environment_or_ssm("SENTRY_DSN", f"/{self._get_project_name()}/env_secrets/SENTRY_DSN")

    # Qdrant connection settings
    @property
    def QDRANT_URL(self) -> str | None:
        return get_environment_or_ssm("QDRANT_URL", f"/{self._get_project_name()}/env_secrets/QDRANT_URL")

    @property
    def QDRANT_API_KEY(self) -> str | None:
        return get_environment_or_ssm("QDRANT_API_KEY", f"/{self._get_project_name()}/env_secrets/QDRANT_API_KEY")

    AUTH_PROVIDER_PUBLIC_KEY: str | None = None
    DISABLE_AUTH_SIGNATURE_VERIFICATION: bool = ENVIRONMENT == "local"

    # Qdrant collection names
    QDRANT_COLLECTION_PREFIX: str = "parliament_mcp_"

    # Sparse text embedding model
    SPARSE_TEXT_EMBEDDING_MODEL: str = "Qdrant/bm25"

    # Chunking settings
    # See https://www.elastic.co/search-labs/blog/elasticsearch-chunking-inference-api-endpoints
    CHUNK_SIZE: int = 300
    SENTENCE_OVERLAP: int = 1
    CHUNK_STRATEGY: str = "sentence"

    PARLIAMENTARY_QUESTIONS_COLLECTION: str = "parliament_mcp_parliamentary_questions"
    HANSARD_CONTRIBUTIONS_COLLECTION: str = "parliament_mcp_hansard_contributions"

    # MCP settings
    MCP_HOST: str = "0.0.0.0"  # nosec B104 - Binding to all interfaces is intentional for containerized deployment
    MCP_PORT: int = 8080

    # The MCP server can be accessed at /{MCP_ROOT_PATH}/mcp
    MCP_ROOT_PATH: str = "/"

    # Allowed hosts for MCP transport security (comma-separated)
    # Used to prevent DNS rebinding attacks
    @property
    def MCP_ALLOWED_HOSTS(self) -> str | None:
        return get_environment_or_ssm(
            "MCP_ALLOWED_HOSTS",
            f"/{self._get_project_name()}/env_secrets/MCP_ALLOWED_HOSTS",
            default="localhost,127.0.0.1",
        )

    @property
    def MODEL_PROVIDER(self) -> str | None:
        return get_environment_or_ssm(
            "MODEL_PROVIDER",
            f"/{self._get_project_name()}/env_secrets/MODEL_PROVIDER",
            default="azure_openai",
        )

    # Rate limiting settings for parliament.uk API.
    HTTP_MAX_RATE_PER_SECOND: float = 10

    # Load environment variables from .env file in local environment
    # from pydantic_settings import SettingsConfigDict
    if ENVIRONMENT == "local":
        model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class OpenAIParliamentMCPSettings(ParliamentMCPSettings):
    """Configuration settings for Parliament MCP application with Azure OpenAI backend."""

    @property
    def AZURE_OPENAI_API_KEY(self) -> str:
        return get_environment_or_ssm(
            "AZURE_OPENAI_API_KEY",
            f"/{self._get_project_name()}/env_secrets/AZURE_OPENAI_API_KEY",
        )

    @property
    def AZURE_OPENAI_ENDPOINT(self) -> str:
        return get_environment_or_ssm(
            "AZURE_OPENAI_ENDPOINT",
            f"/{self._get_project_name()}/env_secrets/AZURE_OPENAI_ENDPOINT",
        )

    @property
    def AZURE_OPENAI_EMBEDDING_MODEL(self) -> str:
        return get_environment_or_ssm(
            "AZURE_OPENAI_EMBEDDING_MODEL",
            f"/{self._get_project_name()}/env_secrets/AZURE_OPENAI_EMBEDDING_MODEL",
        )

    @property
    def AZURE_OPENAI_API_VERSION(self) -> str:
        return get_environment_or_ssm(
            "AZURE_OPENAI_API_VERSION",
            f"/{self._get_project_name()}/env_secrets/AZURE_OPENAI_API_VERSION",
            "preview",
        )

    EMBEDDING_DIMENSIONS: int = 1024


class BedrockParliamentMCPSettings(ParliamentMCPSettings):
    """Configuration settings for Parliament MCP application with AWS Bedrock backend."""

    @property
    def BEDROCK_EMBEDDINGS_MODEL(self) -> str:
        return get_environment_or_ssm(
            "BEDROCK_EMBEDDINGS_MODEL",
            f"/{self._get_project_name()}/env_secrets/BEDROCK_EMBEDDINGS_MODEL",
        )

    EMBEDDING_DIMENSIONS: int = 1024


def get_mcp_settings():
    settings_lookup = {"azure_openai": OpenAIParliamentMCPSettings, "aws_bedrock": BedrockParliamentMCPSettings}

    default_settings = ParliamentMCPSettings()
    provider = default_settings.MODEL_PROVIDER.lower()
    if provider not in settings_lookup:
        err_msg = f"MODEL_PROVIDER must be one of {list(settings_lookup.keys())}."
        raise InvalidLLMProviderError(err_msg)

    return settings_lookup[provider]()


settings = get_mcp_settings()
