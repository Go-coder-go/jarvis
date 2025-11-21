# core/views_auth.py
import os
import jwt
from django.conf import settings
from django.contrib.auth import get_user_model
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponseBadRequest
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import json

User = get_user_model()

# Ensure these exist in settings (or env)
JWT_SECRET = getattr(settings, "JWT_SECRET", os.getenv("JWT_SECRET"))
JWT_ALGO = getattr(settings, "JWT_ALGO", os.getenv("JWT_ALGO", "HS256"))
GOOGLE_CLIENT_ID = getattr(settings, "GOOGLE_CLIENT_ID", os.getenv("GOOGLE_CLIENT_ID"))

@csrf_exempt
def google_auth_view(request):
    """
    Expects POST JSON: { "id_token": "<google id token>" }
    Verifies with Google, returns our JWT on success:
    { "jwt": "...", "email": "...", "threadId": null } (optional fields)
    """
    if request.method != "POST":
        return HttpResponseBadRequest(json.dumps({"error": "POST required"}), content_type="application/json")

    try:
        body = json.loads(request.body.decode())
        idtoken = body.get("id_token") or body.get("idToken")
        if not idtoken:
            return JsonResponse({"error": "id_token missing"}, status=400)

        # Verify token
        idinfo = id_token.verify_oauth2_token(idtoken, google_requests.Request(), GOOGLE_CLIENT_ID)

        # idinfo will contain: 'sub', 'email', 'email_verified', 'name', 'picture', ...
        google_sub = idinfo.get("sub")
        email = idinfo.get("email")
        email_verified = idinfo.get("email_verified", False)
        name = idinfo.get("name")

        if not email:
            return JsonResponse({"error": "Google token missing email"}, status=400)

        # Optionally require verified email
        # if not email_verified:
        #     return JsonResponse({"error": "Email not verified by Google"}, status=403)

        # Create or get a local user record
        # Adjust fields as per your auth model
        user, created = User.objects.get_or_create(
            email=email,
            defaults={"username": email.split("@")[0], "first_name": name or ""}
        )

        # Build your own JWT (payload minimal; add exp if you want)
        payload = {"email": email, "google_sub": google_sub}
        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)

        return JsonResponse({"jwt": token, "email": email})
    except ValueError as ve:
        # token invalid
        return JsonResponse({"error": "Invalid token", "details": str(ve)}, status=400)
    except Exception as e:
        return JsonResponse({"error": "Internal error", "details": str(e)}, status=500)