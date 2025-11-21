# jarvis/core/views.py
import os
import requests
import threading
import uuid
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, serializers
from django.conf import settings
from typing import Optional, Dict, Any, List
from .utils import verify_jwt
import time
import asyncio
from agent.agent import call_agent

# In-memory thread store (prototype only)
# THREADS: threadId -> {
#   "lock": threading.Lock(),
#   "messages": [ {role, content}, ... ],
#   "owner": "email|githubId",
#   "created_at": timestamp,
#   "updated_at": timestamp
# }
THREADS: Dict[str, Dict[str, Any]] = {}
THREADS_LOCK = threading.Lock()  # protects THREADS dict itself

# Configurable caps
MAX_MESSAGES_PER_THREAD = 40   # total messages (user + assistant + system)
SYSTEM_PROMPT = "You are a helpful assistant."

class ChatSerializer(serializers.Serializer):
    message = serializers.CharField()

def _owner_id_from_payload(payload: Dict[str, Any]) -> str:
    """Create a stable owner id string from JWT payload."""
    email = payload.get("email", "")
    github = payload.get("githubId", "")
    return f"{email}|{github}"

def _ensure_thread(thread_id: str, owner_id: str) -> Dict[str, Any]:
    """
    Ensure thread exists and return the thread dict.
    If thread doesn't exist, create it and set owner to owner_id.
    """
    with THREADS_LOCK:
        thread = THREADS.get(thread_id)
        if thread is None:
            thread = {
                "lock": threading.Lock(),
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
                "owner": owner_id,
                "created_at": time.time(),
                "updated_at": time.time(),
            }
            THREADS[thread_id] = thread
    return thread

def _trim_thread_messages(messages: List[Dict[str, str]]):
    """Trim oldest user/assistant messages to keep within MAX_MESSAGES_PER_THREAD."""
    # keep system prompt (assumed at index 0)
    if len(messages) <= MAX_MESSAGES_PER_THREAD:
        return
    # Remove oldest after system prompt
    to_remove = len(messages) - MAX_MESSAGES_PER_THREAD
    del messages[1 : 1 + to_remove]

class ChatForwardView(APIView):
    """
    POST /api/chat/ or /api/chat/?threadId=<id>
    Header: Authorization: Bearer <jwt>
    Body: { "message": "..." }

    Behavior:
    - No threadId: create a brand new thread and return threadId.
    - threadId provided:
        - If exists and owner matches JWT owner: continue thread.
        - If exists and owner differs: 403.
        - If does not exist: create thread and set owner to JWT owner.
    """
    permission_classes = []  # custom token check inside

    def post(self, request, *args, **kwargs):
        # 1) Authorization header
        auth_header = request.headers.get("Authorization") or request.META.get("HTTP_AUTHORIZATION")
        if not auth_header or " " not in auth_header:
            return Response({"error": "Unauthorized"}, status=status.HTTP_401_UNAUTHORIZED)
        parts = auth_header.split()
        if len(parts) != 2:
            return Response({"error": "Unauthorized"}, status=status.HTTP_401_UNAUTHORIZED)
        token = parts[1]

        decoded = verify_jwt(token)
        if not decoded:
            return Response({"error": "Invalid token"}, status=status.HTTP_403_FORBIDDEN)

        # Owner identifier for thread ownership checks
        owner_id = _owner_id_from_payload(decoded)

        # 2) Validate body
        serializer = ChatSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        user_message = serializer.validated_data["message"]

        # 3) Get thread id from query param (optional)
        provided_thread_id = request.query_params.get("threadId")
        if provided_thread_id:
            # If provided, we must either use it or create it and assign owner
            with THREADS_LOCK:
                existing = THREADS.get(provided_thread_id)
                if existing:
                    # Thread exists: verify owner
                    if existing.get("owner") != owner_id:
                        return Response({"error": "Thread belongs to another user"}, status=status.HTTP_403_FORBIDDEN)
                    # Use existing thread (no change)
                    thread = existing
                else:
                    # Create new thread with provided id and assign owner
                    thread = {
                        "lock": threading.Lock(),
                        "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
                        "owner": owner_id,
                        "created_at": time.time(),
                        "updated_at": time.time(),
                    }
                    THREADS[provided_thread_id] = thread
            thread_id = provided_thread_id
        else:
            # No threadId provided → create a new server-generated thread id
            thread_id = str(uuid.uuid4())
            thread = _ensure_thread(thread_id, owner_id)

        # 4) Append user's message to thread under per-thread lock
        lock: threading.Lock = thread["lock"]
        with lock:
            thread_messages: List[Dict[str, str]] = thread["messages"]
            thread_messages.append({"role": "user", "content": user_message})
            _trim_thread_messages(thread_messages)
            thread["updated_at"] = time.time()
            # Build payload using the current thread messages
            payload = {
                "model": getattr(settings, "OPENAI_DEFAULT_MODEL", "gpt-4o-mini"),
                "messages": thread_messages,
                # optional: "temperature": 0.7, "max_tokens": 800
            }

        # # 5) Call OpenAI (outside lock) - keep a timeout
        # openai_api_key = getattr(settings, "OPENAI_API_KEY", None)
        # if not openai_api_key:
        #     return Response({"error": "OpenAI API key not configured"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # headers = {
        #     "Authorization": f"Bearer {openai_api_key}",
        #     "Content-Type": "application/json",
        # }

        # try:
        #     resp = requests.post(
        #         getattr(settings, "OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"),
        #         json=payload,
        #         headers=headers,
        #         timeout=60,
        #     )
        #     resp.raise_for_status()
        # except requests.RequestException as e:
        #     # on failure you may want to rollback the last user append; left as-is for now
        #     return Response({"error": "Failed to contact OpenAI", "details": str(e)}, status=status.HTTP_502_BAD_GATEWAY)

        # j = resp.json()

        # # 6) Extract assistant reply and append it under lock
        # try:
        #     assistant_text = j["choices"][0]["message"]["content"]
        # except Exception:
        #     assistant_text = j  # fallback to raw response

        # with lock:
        #     thread_messages.append({"role": "assistant", "content": assistant_text})
        #     _trim_thread_messages(thread_messages)
        #     thread["updated_at"] = time.time()

        try:
            if asyncio.iscoroutinefunction(call_agent):
                # run async call in its own event loop
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    agent_result = loop.run_until_complete(call_agent(query=user_message, thread_id=thread_id))
                finally:
                    try:
                        loop.run_until_complete(loop.shutdown_asyncgens())
                    except Exception:
                        pass
                    asyncio.set_event_loop(None)
                    loop.close()
            else:
                # synchronous call_agent
                agent_result = call_agent(query=user_message, thread_id=thread_id)
        except Exception as e:
            # Optionally rollback the appended user message:
            # with lock:
            #     if thread["messages"] and thread["messages"][-1]["role"] == "user":
            #         thread["messages"].pop()
            return Response({"error": "Internal agent error", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # --- Append agent response to thread (under lock) ---
        assistant_text = agent_result  # adapt if your agent returns a dict; extract text as needed
        # If agent_result is a dict like {"text": "..."} you might want:
        # assistant_text = agent_result.get("text") or str(agent_result)

        with lock:
            thread_messages.append({"role": "assistant", "content": assistant_text})
            _trim_thread_messages(thread_messages)
            thread["updated_at"] = time.time()

        # 7) Return assistant reply and threadId so client can continue
        return Response({
            "threadId": thread_id,
            "response": assistant_text,
            "agent_raw": agent_result,
        }, status=status.HTTP_200_OK)
        
        

# backend: views.py
import json
import requests
import jwt  # PyJWT
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime, timedelta

GOOGLE_CLIENT_ID = "1234567890-abc123def456ghi789.apps.googleusercontent.com"
JWT_SECRET = "replace-with-strong-secret"  # move to env
JWT_ALGO = "HS256"

@csrf_exempt
def google_auth(request):
    if request.method != "POST":
        return HttpResponseBadRequest(json.dumps({"error": "POST required"}), content_type="application/json")

    try:
        body = json.loads(request.body.decode("utf-8"))
        id_token = body.get("id_token")
        if not id_token:
            return JsonResponse({"error": "missing id_token"}, status=400)

        # Option A: Verify with Google's tokeninfo endpoint (easy)
        r = requests.get("https://oauth2.googleapis.com/tokeninfo", params={"id_token": id_token}, timeout=5)
        if r.status_code != 200:
            return JsonResponse({"error": "invalid id_token"}, status=400)
        token_info = r.json()

        # Verify audience (client_id)
        if token_info.get("aud") != GOOGLE_CLIENT_ID:
            return JsonResponse({"error": "invalid audience"}, status=400)

        # You can also check expiry, email_verified etc.
        email = token_info.get("email")
        name = token_info.get("name")

        # Create your own JWT for session (or use Django session)
        payload = {
            "sub": token_info.get("sub"),
            "email": email,
            "name": name,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=8),
        }
        jwt_token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)

        return JsonResponse({"jwt": jwt_token, "email": email, "name": name})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)