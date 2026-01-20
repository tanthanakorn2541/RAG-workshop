from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    GOOGLE_API_KEY: str  # from .env
    PROJECT_NAME: str = "Free Insight RAG"

    # Storage
    CHROMA_DB_DIR: str = "./chroma_db_free"
    METADATA_PATH: str = "./metadata.json"   # ✅ ADD THIS

    # API
    API_V1_STR: str = "/api/v1"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
