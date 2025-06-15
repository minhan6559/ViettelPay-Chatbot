"""
Configuration utilities for ViettelPay Agent
Supports both Streamlit secrets.toml and environment variables
"""

import os
from typing import Optional, Any


def get_secret(
    key: str, section: str = "api_keys", default: Optional[str] = None
) -> Optional[str]:
    """
    Get secret from Streamlit secrets.toml or environment variables

    Args:
        key: The secret key name
        section: Section in secrets.toml (default: "api_keys")
        default: Default value if secret not found

    Returns:
        Secret value or default
    """
    try:
        # Try to get from Streamlit secrets first
        import streamlit as st

        if hasattr(st, "secrets") and section in st.secrets:
            return st.secrets[section].get(key, default)
    except Exception as e:
        print(e)
        pass

    # Fallback to environment variables
    return os.getenv(key, default)


def get_config(
    key: str, section: str = "models", default: Optional[str] = None
) -> Optional[str]:
    """
    Get configuration from Streamlit secrets.toml or environment variables

    Args:
        key: The config key name
        section: Section in secrets.toml (default: "models")
        default: Default value if config not found

    Returns:
        Config value or default
    """
    try:
        # Try to get from Streamlit secrets first
        import streamlit as st

        if hasattr(st, "secrets") and section in st.secrets:
            return st.secrets[section].get(key, default)
    except Exception as e:
        print(e)
        pass

    # Fallback to environment variables
    return os.getenv(key, default)


def get_path(
    key: str, section: str = "paths", default: Optional[str] = None
) -> Optional[str]:
    """
    Get path configuration from Streamlit secrets.toml or environment variables

    Args:
        key: The path key name
        section: Section in secrets.toml (default: "paths")
        default: Default value if path not found

    Returns:
        Path value or default
    """
    try:
        # Try to get from Streamlit secrets first
        import streamlit as st

        if hasattr(st, "secrets") and section in st.secrets:
            return st.secrets[section].get(key, default)
    except Exception as e:
        print(e)
        pass

    # Fallback to environment variables
    return os.getenv(key, default)


def is_streamlit_environment() -> bool:
    """
    Check if running in Streamlit environment

    Returns:
        True if running in Streamlit, False otherwise
    """
    try:
        import streamlit as st

        return hasattr(st, "secrets")
    except ImportError:
        return False


# Common API keys
def get_google_api_key() -> Optional[str]:
    """Get Google API key from secrets or environment"""
    return get_secret("GOOGLE_API_KEY")


def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from secrets or environment"""
    return get_secret("OPENAI_API_KEY")


def get_cohere_api_key() -> Optional[str]:
    """Get Cohere API key from secrets or environment"""
    return get_secret("COHERE_API_KEY")


# Common configurations
def get_embedding_model() -> str:
    """Get embedding model name"""
    return get_config(
        "EMBEDDING_MODEL", default="dangvantuan/vietnamese-document-embedding"
    )


def get_llm_provider() -> str:
    """Get LLM provider"""
    return get_config("LLM_PROVIDER", default="gemini")


def get_knowledge_base_path() -> str:
    """Get knowledge base path"""
    return get_path("KNOWLEDGE_BASE_PATH", default="./knowledge_base")


def get_documents_folder() -> str:
    """Get documents folder path"""
    return get_path("DOCUMENTS_FOLDER", default="./viettelpay_docs")
