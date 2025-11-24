"""
Simplified configuration for RAG Course QA Project
Works without Pydantic BaseSettings complexity
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ProjectPaths:
    """Project directory paths"""
    ROOT = Path(__file__).parent.parent
    SRC = ROOT / "src"
    DATA = ROOT / "data"
    RAW_DATA = DATA / "raw"
    PROCESSED_DATA = DATA / "processed"
    EMBEDDINGS = DATA / "embeddings"
    CHROMA_DATA = DATA / "chroma"
    MODELS = ROOT / "models"
    LOGS = ROOT / "logs"
    CONFIGS = ROOT / "configs"
    
    @classmethod
    def create_all_dirs(cls):
        """Create all necessary directories"""
        for attr_name in dir(cls):
            if attr_name.isupper():
                path = getattr(cls, attr_name)
                if isinstance(path, Path):
                    path.mkdir(parents=True, exist_ok=True)

class APIConfig:
    """API Keys Configuration"""
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "course-qa-index")
        self.chroma_persist_dir = Path(os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma"))
        self.chroma_collection = os.getenv("CHROMA_COLLECTION_NAME", "course_materials")
    
    def validate_keys(self) -> Dict[str, bool]:
        """Check which API keys are configured"""
        return {
            "openai": bool(self.openai_api_key),
            "pinecone": bool(self.pinecone_api_key),
            "huggingface": bool(self.huggingface_api_key),
            "cohere": bool(self.cohere_api_key),
            "chromadb": True,  # Always available locally
        }

class ModelConfig:
    """Model Configuration"""
    def __init__(self):
        self.max_tokens = int(os.getenv("MAX_TOKENS", "2000"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.top_k_retrieval = int(os.getenv("TOP_K_RETRIEVAL", "5"))
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        self.default_llm = os.getenv("DEFAULT_LLM", "gpt-3.5-turbo")

class AppConfig:
    """Application Configuration"""
    def __init__(self):
        self.app_env = os.getenv("APP_ENV", "development")
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.debug = os.getenv("DEBUG", "true").lower() == "true"
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8000"))
        self.api_reload = os.getenv("API_RELOAD", "true").lower() == "true"
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///./data/app.db")

class ConfigurationManager:
    """Central configuration manager"""
    def __init__(self):
        self.api = APIConfig()
        self.model = ModelConfig()
        self.app = AppConfig()
        self.paths = ProjectPaths()
        
        # Create necessary directories
        self.paths.create_all_dirs()
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration"""
        api_status = self.api.validate_keys()
        if not api_status["openai"]:
            print("⚠️ OpenAI API key not configured")
    
    def __repr__(self):
        return f"ConfigurationManager(env={self.app.app_env})"

# Create global instances
config = ConfigurationManager()
api_config = config.api
model_config = config.model
app_config = config.app
paths = config.paths

# For backward compatibility
def validate_environment() -> bool:
    """Check if minimum requirements are met"""
    return bool(config.api.openai_api_key)

def get_api_key(service: str) -> Optional[str]:
    """Get API key for a service"""
    keys = {
        "openai": config.api.openai_api_key,
        "pinecone": config.api.pinecone_api_key,
        "huggingface": config.api.huggingface_api_key,
        "cohere": config.api.cohere_api_key,
    }
    return keys.get(service)

if __name__ == "__main__":
    print("✅ Configuration loaded successfully!")
    print(f"📁 Project root: {paths.ROOT}")
    print(f"🔑 OpenAI configured: {bool(api_config.openai_api_key)}")
    print(f"🌍 Environment: {app_config.app_env}")
