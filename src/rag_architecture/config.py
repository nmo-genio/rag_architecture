# src/rag_architecture/config.py

import os
from dotenv import load_dotenv

load_dotenv()

REQUIRED_ENV_VARS = [
    "OPENAI_API_KEY",
]

missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]

if missing:
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(missing)}"
    )

# Export safe access to your env vars
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
