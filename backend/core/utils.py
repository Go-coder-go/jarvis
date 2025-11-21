# core/utils.py
import os
import jwt
from typing import Optional, Dict, Any

def verify_jwt(token: str) -> Optional[Dict[str, Any]]:
    secret = os.environ.get("JWT_SECRET")
    algo = os.environ.get("JWT_ALGO", "HS256")
    # print(secret , 'this is the secret')
    if not secret:
        # Fail fast in dev so you know to set the env var
        return None
    try:
        payload = jwt.decode(token, secret, algorithms=[algo])
        return payload
    except jwt.PyJWTError:
        return None