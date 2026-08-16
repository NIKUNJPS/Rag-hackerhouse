import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # STT
    STT_PROVIDER = os.getenv("STT_PROVIDER", "elevenlabs")  # elevenlabs | sarvam
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
    SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")

    # LLM (answer generation)
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # groq | openai | anthropic
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    # Embeddings (local, no network hop -> keeps retrieval fast)
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Retrieval
    VECTOR_INDEX_DIR = os.getenv("VECTOR_INDEX_DIR", "backend/data/index")
    TOP_K = int(os.getenv("TOP_K", "5"))

    # Guardrails
    OFFTOPIC_SIM_THRESHOLD = float(os.getenv("OFFTOPIC_SIM_THRESHOLD", "0.28"))
    GROUNDING_OVERLAP_THRESHOLD = float(os.getenv("GROUNDING_OVERLAP_THRESHOLD", "0.15"))

    # Dataset
    DATASET_NAME = os.getenv("DATASET_NAME", "ai4bharat/IndicMSMARCO")
    DATASET_CONFIG = os.getenv("DATASET_CONFIG", "hi")
    DATASET_SAMPLE_SIZE = int(os.getenv("DATASET_SAMPLE_SIZE", "5000"))

settings = Settings()
