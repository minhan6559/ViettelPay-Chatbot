"""
LLM Client Abstraction Layer
Supports multiple LLM providers without hardcoding
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from prompt"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if LLM service is available"""
        pass


class GeminiClient(BaseLLMClient):
    """Google Gemini client implementation"""

    def __init__(
        self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash-lite"
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError("Gemini API key not provided")

        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)
            print(f"✅ Gemini client initialized with model: {self.model}")
        except ImportError:
            raise ImportError("google-generativeai package not installed")

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Gemini"""
        try:
            # Set default temperature to 0.1 for consistency
            generation_config = {
                "temperature": kwargs.get("temperature", 0.1),
                "top_p": kwargs.get("top_p", 0.8),
                "top_k": kwargs.get("top_k", 40),
                "max_output_tokens": kwargs.get("max_output_tokens", 2048),
            }

            response = self.client.generate_content(
                prompt, generation_config=generation_config
            )
            return response.text
        except Exception as e:
            print(f"❌ Gemini generation error: {e}")
            raise

    def is_available(self) -> bool:
        """Check Gemini availability"""
        try:
            test_response = self.client.generate_content("Hello")
            return bool(test_response.text)
        except:
            return False


class OpenAIClient(BaseLLMClient):
    """OpenAI client implementation"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        try:
            import openai

            self.client = openai.OpenAI(api_key=self.api_key)
            print(f"✅ OpenAI client initialized with model: {self.model}")
        except ImportError:
            raise ImportError("openai package not installed")

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI"""
        try:
            # Set default temperature to 0.1 for consistency
            openai_kwargs = {
                "temperature": kwargs.get("temperature", 0.1),
                "top_p": kwargs.get("top_p", 1.0),
                "max_tokens": kwargs.get("max_tokens", 2048),
            }
            # Remove any Gemini-specific parameters
            openai_kwargs.update(
                {
                    k: v
                    for k, v in kwargs.items()
                    if k
                    in [
                        "temperature",
                        "top_p",
                        "max_tokens",
                        "frequency_penalty",
                        "presence_penalty",
                    ]
                }
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **openai_kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ OpenAI generation error: {e}")
            raise

    def is_available(self) -> bool:
        """Check OpenAI availability"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
            )
            return bool(response.choices[0].message.content)
        except:
            return False


class LLMClientFactory:
    """Factory for creating LLM clients"""

    SUPPORTED_PROVIDERS = {
        "gemini": GeminiClient,
        "openai": OpenAIClient,
    }

    @classmethod
    def create_client(self, provider: str = "gemini", **kwargs) -> BaseLLMClient:
        """Create LLM client by provider name"""

        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider: {provider}. Supported: {list(self.SUPPORTED_PROVIDERS.keys())}"
            )

        client_class = self.SUPPORTED_PROVIDERS[provider]
        return client_class(**kwargs)

    @classmethod
    def get_available_providers(cls) -> list:
        """Get list of available providers"""
        return list(cls.SUPPORTED_PROVIDERS.keys())


# Usage example
if __name__ == "__main__":
    # Test Gemini client
    try:
        client = LLMClientFactory.create_client("gemini")
        if client.is_available():
            response = client.generate("Xin chào, bạn có khỏe không?")
            print(f"Response: {response}")
        else:
            print("Gemini not available")
    except Exception as e:
        print(f"Error: {e}")
