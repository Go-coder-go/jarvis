import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # required
OPENAI_API_URL = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_DEFAULT_MODEL = os.environ.get("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")

# JWT secret for simple token validation
JWT_SECRET = os.environ.get("JWT_SECRET", "JWT_SECRET","dev-secret-change-me")
JWT_ALGO = os.environ.get("JWT_ALGO", "HS256")