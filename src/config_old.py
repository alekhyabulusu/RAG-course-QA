"""
Configuration management for RAG Course QA Application.
Simplified version without Anthropic and Weaviate.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from dotenv import load_dotenv
from pydantic import Field, validator, BaseModel
from pydantic_settings import BaseSettings


# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =====================================================
# Enums for Configuration Options
# =====================================================

class Environment(str, Enum):
    """Application environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class VectorDBType(str, Enum):
    """Available vector database types"""
    CHROMADB = "chromadb"
    PINECONE = "pinecone"


class LLMProvider(str, Enum):
    """Available LLM providers"""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"


class EmbeddingModel(str, Enum):
    """Available embedding models"""
    OPENAI_ADA = "text-embedding-ada-002"
    OPENAI_3_SMALL = "text-embedding-3-small"
    OPENAI_3_LARGE = "text-embedding-3-large"
    SENTENCE_TRANSFORMER = "all-MiniLM-L6-v2"
    INSTRUCTOR = "hkunlp/instructor-large"


# =====================================================
# API Configuration
# =====================================================

class APIConfig(BaseSettings):
    """API Keys and External Service Configuration"""
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    openai_org_id: Optional[str] = Field(None, env="OPENAI_ORG_ID")
    
    # Hugging Face Configuration
    huggingface_api_key: Optional[str] = Field(None, env="HUGGINGFACE_API_KEY")
    
    # Cohere Configuration
    cohere_api_key: Optional[str] = Field(None, env="COHERE_API_KEY")
    
    # Pinecone Configuration
    pinecone_api_key: Optional[str] = Field(None, env="PINECONE_API_KEY")
    pinecone_environment: str = Field("gcp-starter", env="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field("course-qa-index", env="PINECONE_INDEX_NAME")
    
    # ChromaDB Configuration
    chroma_persist_dir: Path = Field(Path("./data/chroma"), env="CHROMA_PERSIST_DIRECTORY")
    chroma_collection: str = Field("course_materials", env="CHROMA_COLLECTION_NAME")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
    @validator("chroma_persist_dir")
    def ensure_chroma_dir_exists(cls, v):
        """Create ChromaDB directory if it doesn't exist"""
        v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for a specific service"""
        service_map = {
            "openai": self.openai_api_key,
            "huggingface": self.huggingface_api_key,
            "cohere": self.cohere_api_key,
            "pinecone": self.pinecone_api_tkey,
        }
        return service_map.get(service.lower())
    
    def validate_keys(self) -> Dict[str, bool]:
        """Check which API keys are configured"""
        return {
            "openai": bool(self.openai_api_key),
            "huggingface": bool(self.huggingface_api_key),
            "cohere": bool(self.cohere_api_key),
            "pinecone": bool(self.pinecone_api_key),
            "chromadb": True,  # Always available (local)
        }


# =====================================================
# Model Configuration
# =====================================================

class ModelConfig(BaseSettings):
    """LLM and Embedding Model Configuration"""
    
    # LLM Settings
    max_tokens: int = Field(2000, env="MAX_TOKENS")
    temperature: float = Field(0.7, env="TEMPERATURE")
    top_p: float = Field(1.0, env="TOP_P")
    frequency_penalty: float = Field(0.0, env="FREQUENCY_PENALTY")
    presence_penalty: float = Field(0.0, env="PRESENCE_PENALTY")
    
    # Retrieval Settings
    top_k_retrieval: int = Field(5, env="TOP_K_RETRIEVAL")
    chunk_size: int = Field(500, env="CHUNK_SIZE")
    chunk_overlap: int = Field(50, env="CHUNK_OVERLAP")
    
    # Embedding Settings
    embedding_model: str = Field("text-embedding-ada-002", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(1536, env="EMBEDDING_DIMENSION")
    
    # Model Selection
    default_llm: str = Field("gpt-3.5-turbo", env="DEFAULT_LLM")
    available_models: List[str] = [
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-turbo-preview",
        "llama-3-8b",
        "llama-3-70b",
        "command",
        "command-light",
    ]
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @validator("temperature")
    def validate_temperature(cls, v):
        """Ensure temperature is between 0 and 2"""
        if not 0 <= v <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v
    
    @validator("top_p")
    def validate_top_p(cls, v):
        """Ensure top_p is between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError("top_p must be between 0 and 1")
        return v
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for specific model"""
        configs = {
            "gpt-3.5-turbo": {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "provider": "openai",
                "cost_per_1k_tokens": 0.002,
            },
            "gpt-4": {
                "max_tokens": min(self.max_tokens, 8192),
                "temperature": self.temperature,
                "provider": "openai",
                "cost_per_1k_tokens": 0.03,
            },
            "llama-3-8b": {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "provider": "huggingface",
                "cost_per_1k_tokens": 0,  # Free on HF
            },
            "command": {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "provider": "cohere",
                "cost_per_1k_tokens": 0.001,
            },
        }
        return configs.get(model_name, configs["gpt-3.5-turbo"])


# =====================================================
# Application Configuration
# =====================================================

class AppConfig(BaseSettings):
    """General Application Configuration"""
    
    # Environment
    app_env: Environment = Field(Environment.DEVELOPMENT, env="APP_ENV")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    debug: bool = Field(True, env="DEBUG")
    
    # API Server Settings
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_reload: bool = Field(True, env="API_RELOAD")
    api_workers: int = Field(1, env="API_WORKERS")
    
    # Database
    database_url: str = Field("sqlite:///./data/app.db", env="DATABASE_URL")
    
    # Session Management
    session_timeout: int = Field(3600, env="SESSION_TIMEOUT")  # seconds
    max_sessions_per_user: int = Field(5, env="MAX_SESSIONS_PER_USER")
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(100, env="RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(60, env="RATE_LIMIT_PERIOD")  # seconds
    
    # File Upload Settings
    max_file_size: int = Field(10 * 1024 * 1024, env="MAX_FILE_SIZE")  # 10MB
    allowed_file_types: List[str] = [".pdf", ".txt", ".docx", ".md"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        use_enum_values = True
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Ensure log level is valid"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()


# =====================================================
# Project Paths Configuration
# =====================================================

class ProjectPaths:
    """Project directory paths and file management"""
    
    # Base paths
    ROOT = Path(__file__).parent.parent.resolve()
    SRC = ROOT / "src"
    DATA = ROOT / "data"
    
    # Data subdirectories
    RAW_DATA = DATA / "raw"
    PROCESSED_DATA = DATA / "processed"
    EMBEDDINGS = DATA / "embeddings"
    CHROMA_DATA = DATA / "chroma"
    
    # Other directories
    MODELS = ROOT / "models"
    LOGS = ROOT / "logs"
    CONFIGS = ROOT / "configs"
    TESTS = ROOT / "tests"
    NOTEBOOKS = ROOT / "notebooks"
    SCRIPTS = ROOT / "scripts"
    STATIC = ROOT / "static"
    TEMPLATES = ROOT / "templates"
    
    @classmethod
    def create_all_dirs(cls):
        """Create all necessary directories"""
        directories = [
            cls.DATA, cls.RAW_DATA, cls.PROCESSED_DATA, cls.EMBEDDINGS,
            cls.CHROMA_DATA, cls.MODELS, cls.LOGS, cls.CONFIGS,
            cls.STATIC, cls.TEMPLATES
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
    
    @classmethod
    def get_log_file(cls, name: str = "app") -> Path:
        """Get path for log file with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return cls.LOGS / f"{name}_{timestamp}.log"
    
    @classmethod
    def get_data_file(cls, filename: str, data_type: str = "raw") -> Path:
        """Get path for data file"""
        type_map = {
            "raw": cls.RAW_DATA,
            "processed": cls.PROCESSED_DATA,
            "embeddings": cls.EMBEDDINGS,
        }
        return type_map.get(data_type, cls.DATA) / filename
    
    @classmethod
    def list_files(cls, directory: Path, extension: str = None) -> List[Path]:
        """List all files in directory with optional extension filter"""
        if not directory.exists():
            return []
        
        if extension:
            return list(directory.glob(f"*{extension}"))
        return [f for f in directory.iterdir() if f.is_file()]


# =====================================================
# Vector Database Configuration
# =====================================================

class VectorDBConfig(BaseModel):
    """Vector Database Configuration"""
    
    db_type: VectorDBType = VectorDBType.CHROMADB
    
    # ChromaDB settings
    chroma_persist_directory: Path = Path("./data/chroma")
    chroma_collection_name: str = "course_materials"
    
    # Pinecone settings
    pinecone_index_name: str = "course-qa-index"
    pinecone_dimension: int = 1536
    pinecone_metric: str = "cosine"
    pinecone_replicas: int = 1
    
    # Common settings
    similarity_top_k: int = 5
    similarity_threshold: float = 0.7
    
    def get_db_config(self) -> Dict[str, Any]:
        """Get configuration for selected database type"""
        if self.db_type == VectorDBType.CHROMADB:
            return {
                "persist_directory": str(self.chroma_persist_directory),
                "collection_name": self.chroma_collection_name,
            }
        elif self.db_type == VectorDBType.PINECONE:
            return {
                "index_name": self.pinecone_index_name,
                "dimension": self.pinecone_dimension,
                "metric": self.pinecone_metric,
            }
        return {}


# =====================================================
# Configuration Manager
# =====================================================

class ConfigurationManager:
    """Central configuration manager for the application"""
    
    def __init__(self):
        """Initialize all configurations"""
        self.api = APIConfig()
        self.model = ModelConfig()
        self.app = AppConfig()
        self.paths = ProjectPaths()
        self.vector_db = VectorDBConfig()
        
        # Create necessary directories
        self.paths.create_all_dirs()
        
        # Set up logging
        self._setup_logging()
        
        # Validate configuration
        self._validate_config()
    
    def _setup_logging(self):
        """Configure application logging"""
        log_level = getattr(logging, self.app.log_level)
        log_file = self.paths.get_log_file()
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logger.info(f"Logging configured: Level={self.app.log_level}, File={log_file}")
    
    def _validate_config(self):
        """Validate configuration and warn about missing keys"""
        api_status = self.api.validate_keys()
        
        logger.info("API Key Status:")
        for service, is_configured in api_status.items():
            status = "✓" if is_configured else "✗"
            logger.info(f"  {status} {service.capitalize()}")
        
        # Check minimum requirements
        if not api_status["openai"]:
            logger.warning("⚠️ OpenAI API key not configured - Required for GPT models")
        
        if not any([api_status["pinecone"], api_status["chromadb"]]):
            logger.warning("⚠️ No vector database configured - At least one required")
    
    def get_llm_client(self, provider: str = None):
        """Get configured LLM client"""
        provider = provider or self.model.default_llm
        model_config = self.model.get_model_config(provider)
        
        if model_config["provider"] == "openai":
            if not self.api.openai_api_key:
                raise ValueError("OpenAI API key not configured")
            
            from openai import OpenAI
            return OpenAI(api_key=self.api.openai_api_key)
        
        elif model_config["provider"] == "huggingface":
            if not self.api.huggingface_api_key:
                logger.warning("Hugging Face API key not configured - Using public inference")
            return None  # HF client initialization here
        
        elif model_config["provider"] == "cohere":
            if not self.api.cohere_api_key:
                raise ValueError("Cohere API key not configured")
            
            import cohere
            return cohere.Client(self.api.cohere_api_key)
        
        raise ValueError(f"Unknown provider: {model_config['provider']}")
    
    def get_vector_db(self, db_type: VectorDBType = None):
        """Get configured vector database client"""
        db_type = db_type or self.vector_db.db_type
        
        if db_type == VectorDBType.CHROMADB:
            import chromadb
            return chromadb.PersistentClient(
                path=str(self.vector_db.chroma_persist_directory)
            )
        
        elif db_type == VectorDBType.PINECONE:
            if not self.api.pinecone_api_key:
                raise ValueError("Pinecone API key not configured")
            
            from pinecone import Pinecone
            pc = Pinecone(api_key=self.api.pinecone_api_key)
            return pc.Index(self.vector_db.pinecone_index_name)
        
        raise ValueError(f"Unknown vector DB type: {db_type}")
    
    def export_config(self, file_path: Path = None) -> Dict[str, Any]:
        """Export current configuration to JSON"""
        config_dict = {
            "api": {
                "configured_services": self.api.validate_keys(),
                "pinecone_environment": self.api.pinecone_environment,
                "pinecone_index": self.api.pinecone_index_name,
            },
            "model": {
                "default_llm": self.model.default_llm,
                "max_tokens": self.model.max_tokens,
                "temperature": self.model.temperature,
                "embedding_model": self.model.embedding_model,
                "chunk_size": self.model.chunk_size,
                "top_k_retrieval": self.model.top_k_retrieval,
            },
            "app": {
                "environment": self.app.app_env,
                "debug": self.app.debug,
                "api_host": self.app.api_host,
                "api_port": self.app.api_port,
                "database_url": self.app.database_url,
            },
            "vector_db": {
                "type": self.vector_db.db_type,
                "similarity_threshold": self.vector_db.similarity_threshold,
            },
            "paths": {
                "root": str(self.paths.ROOT),
                "data": str(self.paths.DATA),
                "models": str(self.paths.MODELS),
            }
        }
        
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            logger.info(f"Configuration exported to {file_path}")
        
        return config_dict
    
    def __repr__(self) -> str:
        """String representation of configuration"""
        api_status = self.api.validate_keys()
        configured = [k for k, v in api_status.items() if v]
        return (
            f"ConfigurationManager(\n"
            f"  Environment: {self.app.app_env}\n"
            f"  Configured APIs: {', '.join(configured)}\n"
            f"  Default LLM: {self.model.default_llm}\n"
            f"  Vector DB: {self.vector_db.db_type}\n"
            f"  Data Path: {self.paths.DATA}\n"
            f")"
        )


# =====================================================
# Global Configuration Instance
# =====================================================

# Create singleton configuration instance
config = ConfigurationManager()

# Export commonly used objects
api_config = config.api
model_config = config.model
app_config = config.app
paths = config.paths
vector_db_config = config.vector_db


# =====================================================
# Utility Functions
# =====================================================

def get_api_key(service: str) -> Optional[str]:
    """Get API key for a specific service"""
    return config.api.get_api_key(service)


def validate_environment() -> bool:
    """Validate that minimum requirements are met"""
    api_status = config.api.validate_keys()
    
    # Minimum requirement: OpenAI key
    if not api_status["openai"]:
        logger.error("OpenAI API key is required but not configured")
        return False
    
    # At least one vector DB should be available
    if not any([api_status["chromadb"], api_status["pinecone"]]):
        logger.warning("No vector database configured, using ChromaDB locally")
    
    return True


def print_config_summary():
    """Print configuration summary to console"""
    print("\n" + "=" * 50)
    print("RAG Course QA - Configuration Summary")
    print("=" * 50)
    print(config)
    print("\nAPI Keys Status:")
    
    for service, is_configured in config.api.validate_keys().items():
        status = "✅" if is_configured else "❌"
        key_preview = ""
        if is_configured and service != "chromadb":
            key = config.api.get_api_key(service)
            if key:
                key_preview = f" ({key[:10]}...)"
        print(f"  {status} {service.capitalize()}{key_preview}")
    
    print(f"\n📁 Project Root: {paths.ROOT}")
    print(f"📊 Data Directory: {paths.DATA}")
    print(f"🧠 Models Directory: {paths.MODELS}")
    print(f"📝 Logs Directory: {paths.LOGS}")
    print("\n" + "=" * 50)


# =====================================================
# Main Execution (for testing)
# =====================================================

if __name__ == "__main__":
    """Test configuration when run directly"""
    
    print_config_summary()
    
    # Test configuration export
    export_path = paths.CONFIGS / "current_config.json"
    config.export_config(export_path)
    print(f"\n✅ Configuration exported to: {export_path}")
    
    # Validate environment
    if validate_environment():
        print("\n✅ Environment validation passed!")
    else:
        print("\n❌ Environment validation failed - check your .env file")
        print("\nMinimum requirements:")
        print("  1. OpenAI API key (required)")
        print("  2. At least one vector database (ChromaDB works locally)")
    
    # Test getting clients (only if keys are configured)
    try:
        if config.api.openai_api_key:
            client = config.get_llm_client("gpt-3.5-turbo")
            print("\n✅ OpenAI client initialized successfully")
    except Exception as e:
        print(f"\n❌ Failed to initialize LLM client: {e}")
    
    try:
        vector_db = config.get_vector_db(VectorDBType.CHROMADB)
        print("✅ ChromaDB client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize vector DB: {e}")