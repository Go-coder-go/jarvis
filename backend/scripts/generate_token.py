import jwt, os
secret = os.environ.get("JWT_SECRET","replace-me-locally")
token = jwt.encode({"email":"test@example.com","githubId":"123"}, secret, algorithm="HS256")
print(token)