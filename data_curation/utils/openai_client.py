"""Shared OpenAI client factory for data curation pipeline."""
import os

def get_openai_client():
    """Create OpenAI client from environment variables.
    Supports standard OpenAI (OPENAI_API_KEY) and Azure OpenAI (AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_KEY).
    """
    if os.environ.get("AZURE_OPENAI_ENDPOINT"):
        from openai import AzureOpenAI
        return AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_KEY"],
            api_version="2024-02-15-preview",
        )
    else:
        from openai import OpenAI
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
