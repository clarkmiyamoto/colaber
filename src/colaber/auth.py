"""OAuth2 authentication for Google Colab."""

import json
from datetime import datetime, timezone

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from colaber.config import (
    COLAB_CLIENT_ID,
    COLAB_CLIENT_SECRET,
    OAUTH_SCOPES,
    TOKEN_CACHE_DIR,
    TOKEN_CACHE_PATH,
)

# Client config in the format expected by InstalledAppFlow
_CLIENT_CONFIG = {
    "installed": {
        "client_id": COLAB_CLIENT_ID,
        "client_secret": COLAB_CLIENT_SECRET,
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["http://localhost"],
    }
}


def _load_cached_credentials() -> Credentials | None:
    """Load credentials from the token cache file."""
    if not TOKEN_CACHE_PATH.exists():
        return None

    try:
        data = json.loads(TOKEN_CACHE_PATH.read_text())
        expiry = None
        if data.get("expiry"):
            expiry = datetime.fromisoformat(data["expiry"])
        creds = Credentials(
            token=data["token"],
            refresh_token=data.get("refresh_token"),
            token_uri=data.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=data.get("client_id", COLAB_CLIENT_ID),
            client_secret=data.get("client_secret", COLAB_CLIENT_SECRET),
            scopes=data.get("scopes", OAUTH_SCOPES),
            expiry=expiry,
        )
        return creds
    except (json.JSONDecodeError, KeyError):
        return None


def _save_credentials(creds: Credentials) -> None:
    """Save credentials to the token cache file with restricted permissions."""
    TOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes) if creds.scopes else OAUTH_SCOPES,
        "expiry": creds.expiry.isoformat() if creds.expiry else None,
    }

    TOKEN_CACHE_PATH.write_text(json.dumps(data))
    TOKEN_CACHE_PATH.chmod(0o600)


def clear_cached_credentials() -> None:
    """Delete the cached token file."""
    if TOKEN_CACHE_PATH.exists():
        TOKEN_CACHE_PATH.unlink()


def get_credentials() -> Credentials:
    """Get valid Google OAuth2 credentials, prompting login if needed.

    Checks for cached credentials first. If expired or missing expiry
    (stale), attempts a refresh. Otherwise, opens a browser for the
    OAuth2 consent flow.
    """
    creds = _load_cached_credentials()

    if creds and creds.valid:
        return creds

    # Treat missing expiry as stale — always try to refresh
    if creds and creds.refresh_token:
        from google.auth.transport.requests import Request

        try:
            creds.refresh(Request())
            _save_credentials(creds)
            return creds
        except Exception:
            # Refresh failed — clear stale cache and fall through to full auth
            clear_cached_credentials()

    # Run the full OAuth2 browser flow
    flow = InstalledAppFlow.from_client_config(_CLIENT_CONFIG, scopes=OAUTH_SCOPES)
    creds = flow.run_local_server(port=0, open_browser=True)
    _save_credentials(creds)
    return creds
