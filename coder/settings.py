"""
Created by Analitika at 18/02/2025
contact@analitika.fr
"""
"""
Created by Analitika at 12/02/2025
contact@analitika.fr
"""

# External improts
from dotenv import load_dotenv
import os
from pathlib import Path

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_COMPLETIONS_MODEL = os.getenv("OPENAI_COMPLETIONS_MODEL")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
SMALL_EMBEDDINGS_MODEL = os.getenv("SMALL_EMBEDDINGS_MODEL")

PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_FOLDER = PROJ_ROOT / "coder/data"
