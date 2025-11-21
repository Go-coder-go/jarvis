# services/tools.py
import os
import json
import random
import requests
from typing import Any, Dict, List

from langchain_core.tools import tool

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")


@tool(
    "github_profile_lookup",
    description="Fetches a GitHub user's profile, repositories, and followers.",
    return_direct=False,
)
def github_profile_lookup(
    username: str,
    includeRepos: bool = False,
    includeFollowers: bool = False,
) -> str:
    """
    Fetches a user's GitHub profile, repositories, and followers
    directly from the GitHub API.
    """
    api = f"https://api.github.com/users/{username}"
    headers = {
        "User-Agent": "AgenticGitPayBit/1.0.0",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    try:
        profile_res = requests.get(api, headers=headers, timeout=20)
        if not profile_res.ok:
            return json.dumps(
                {
                    "error": f"profile status {profile_res.status_code}",
                    "headers": dict(profile_res.headers),
                }
            )

        profile = profile_res.json()
        out: Dict[str, Any] = {"profile": profile}

        if includeRepos:
            repos_res = requests.get(f"{api}/repos", headers=headers, timeout=20)
            if not repos_res.ok:
                return json.dumps(
                    {
                        "error": f"repos status {repos_res.status_code}",
                        "headers": dict(repos_res.headers),
                    }
                )
            repos_raw: List[Dict[str, Any]] = repos_res.json()
            out["repos"] = [
                {
                    "name": r.get("name"),
                    "description": r.get("description"),
                    "html_url": r.get("html_url"),
                    "stargazers_count": r.get("stargazers_count"),
                    "forks_count": r.get("forks_count"),
                    "language": r.get("language"),
                }
                for r in repos_raw
            ]

        if includeFollowers:
            followers_res = requests.get(
                f"{api}/followers", headers=headers, timeout=20
            )
            if not followers_res.ok:
                return json.dumps(
                    {
                        "error": f"followers status {followers_res.status_code}",
                        "headers": dict(followers_res.headers),
                    }
                )
            followers_raw: List[Dict[str, Any]] = followers_res.json()
            out["followers"] = [
                {
                    "login": f.get("login"),
                    "html_url": f.get("html_url"),
                    "avatar_url": f.get("avatar_url"),
                }
                for f in followers_raw
            ]

        return json.dumps(out)
    except Exception as e:
        return json.dumps({"error": f"exception: {e}"})


@tool(
    "generate_random_name",
    description="Generates random name variations from a given base name. No external API calls.",
    return_direct=False,
)
def generate_random_name(
    base_name: str,
    count: int = 1,
    style: str = "modern",  # options: modern, hacker, fantasy, mixed
) -> str:
    """
    Creates random name variations from a given base name.
    Does not call any external API. Fully offline.
    """
    try:
        # Common modification patterns
        prefixes = ["Neo", "Alpha", "Nova", "Shadow", "Cyber", "Ghost", "Mega", "Zen"]
        suffixes = ["X", "Zero", "Prime", "One", "King", "Storm", "Vortex", "Knight"]
        hacker_suffix = ["_dev", "_ops", "_sec", "404", "1337", "_sys"]
        fantasy_adds = ["dor", "mir", "thas", "ion", "wyn", "riel"]

        variations: List[str] = []

        for _ in range(count):
            name = base_name.capitalize()

            # choose style
            if style == "modern":
                name = f"{random.choice(prefixes)}{name}{random.choice(suffixes)}"

            elif style == "hacker":
                name = f"{name}{random.choice(hacker_suffix)}"

            elif style == "fantasy":
                name = f"{name}{random.choice(fantasy_adds)}"

            elif style == "mixed":
                mix = random.choice(["modern", "hacker", "fantasy"])
                # recurse into one of the other styles
                return generate_random_name(base_name, count, mix)

            else:
                # fallback random numeric suffix
                name = f"{name}_{random.randint(1000, 9999)}"

            variations.append(name)

        return json.dumps({"names": variations})

    except Exception as e:
        return json.dumps({"error": f"exception: {str(e)}"})


@tool(
    "rotate_strings",
    description="Rotates multiple input strings by 5 characters. No external API calls.",
    return_direct=False,
)
def rotate_strings(strings: List[str]) -> str:
    """
    Rotates each input string by 5 characters.
    Example: 'abcdef' -> 'fabcde'
    """
    try:
        rotated: List[str] = []

        for s in strings:
            if not isinstance(s, str):
                rotated.append(None)  # keep position, signal invalid
                continue

            if not s:
                rotated.append(s)
                continue

            # rotate last 5 chars to the front
            shift = 5 % len(s) if len(s) > 0 else 0
            rotated_s = s[-shift:] + s[:-shift] if shift != 0 else s
            rotated_s = rotated_s + 'mukul'
            rotated.append(rotated_s)

        return json.dumps({"rotated": rotated})

    except Exception as e:
        return json.dumps({"error": f"exception: {str(e)}"})
