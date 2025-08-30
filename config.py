"""
Configuration file for the Ticket Data Analysis & Summarization Tool
Updated for Ollama integration, enhanced UI, and smart caching
"""

import os
from typing import Optional

class Config:
    """Configuration settings for the application"""
    
    # Ollama Configuration (Local LLM)
    OLLAMA_ENABLED: bool = True
    OLLAMA_MODEL: str = "llama3.1:8b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_TIMEOUT: int = 120  # seconds
    
    # LLM Generation Parameters
    LLM_TEMPERATURE: float = 0.7
    LLM_TOP_P: float = 0.9
    LLM_MAX_TOKENS: int = 1000
    
    # Application Configuration
    APP_TITLE: str = "Ticket Data Analysis & Summarization"
    APP_ICON: str = "📊"
    PAGE_LAYOUT: str = "wide"
    INITIAL_SIDEBAR_STATE: str = "expanded"
    
    # Data Processing Configuration
    VALID_CATEGORIES: list = ['HDW', 'NET', 'KAI', 'KAV', 'GIGA', 'VOD', 'KAD']
    
    # Enhanced Category Mapping (HDW now mapped to Hardware)
    CATEGORY_MAPPING: dict = {
        'Broadband': ['KAI', 'NET'],
        'Voice': ['KAV'],
        'TV': ['KAD'],
        'GIGA': ['GIGA'],
        'VOD': ['VOD'],
        'Hardware': ['HDW']  # NEW: HDW properly classified
    }
    
    # File Upload Configuration
    SUPPORTED_FILE_TYPES: list = ['txt', 'csv']
    MAX_FILE_SIZE: int = 200 * 1024 * 1024  # 200MB (increased)
    
    # Enhanced UI Configuration
    CHART_HEIGHT: int = 500
    CHART_WIDTH: str = "100%"
    ENABLE_HOVER_EFFECTS: bool = True
    ENABLE_GRADIENTS: bool = True
    ENABLE_ANIMATIONS: bool = True
    
    # Caching Configuration
    ENABLE_CACHING: bool = True
    DATA_CACHE_TTL: int = 3600  # 1 hour
    SUMMARY_CACHE_TTL: int = 1800  # 30 minutes
    MAX_CACHE_SIZE: int = 100  # maximum cached items
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_MAX_SIZE: int = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: int = 5
    
    # Business Intelligence Configuration
    CHURN_RISK_THRESHOLD: int = 3  # tickets per customer
    TREND_ANALYSIS_DAYS: int = 30
    PERFORMANCE_BENCHMARK_THRESHOLD: float = 0.8  # 80%
    
    @classmethod
    def validate_ollama_config(cls) -> bool:
        """Validate Ollama configuration"""
        try:
            import ollama
            models = ollama.list()
            available_models = [model.model for model in models.models]
            return cls.OLLAMA_MODEL in available_models
        except Exception:
            return False
    
    @classmethod
    def get_ollama_config(cls) -> dict:
        """Get Ollama configuration as dictionary"""
        return {
            'enabled': cls.OLLAMA_ENABLED,
            'model': cls.OLLAMA_MODEL,
            'base_url': cls.OLLAMA_BASE_URL,
            'timeout': cls.OLLAMA_TIMEOUT,
            'temperature': cls.LLM_TEMPERATURE,
            'top_p': cls.LLM_TOP_P,
            'max_tokens': cls.LLM_MAX_TOKENS
        }
    
    @classmethod
    def get_cache_config(cls) -> dict:
        """Get caching configuration as dictionary"""
        return {
            'enabled': cls.ENABLE_CACHING,
            'data_ttl': cls.DATA_CACHE_TTL,
            'summary_ttl': cls.SUMMARY_CACHE_TTL,
            'max_size': cls.MAX_CACHE_SIZE
        }
    
    @classmethod
    def get_ui_config(cls) -> dict:
        """Get UI configuration as dictionary"""
        return {
            'chart_height': cls.CHART_HEIGHT,
            'chart_width': cls.CHART_WIDTH,
            'hover_effects': cls.ENABLE_HOVER_EFFECTS,
            'gradients': cls.ENABLE_GRADIENTS,
            'animations': cls.ENABLE_ANIMATIONS
        }

# Usage example:
# from config import Config
# 
# if Config.validate_ollama_config():
#     ollama_config = Config.get_ollama_config()
#     cache_config = Config.get_cache_config()
#     ui_config = Config.get_ui_config()
#     # Use the configurations
# else:
#     # Handle missing Ollama or model
#     pass
