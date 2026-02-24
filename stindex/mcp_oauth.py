"""
Self-contained OAuth 2.0 authorization server for STIndex MCP.

Mirrors the Talk2Metadata / Docs2Synth pattern: the MCP server itself
advertises OAuth endpoints so mcp-remote (and Claude Desktop) can do the
full authorization-code + PKCE flow automatically — no hardcoded token in
the client config.

Implements:
  RFC 8707 — OAuth 2.0 Protected Resource Metadata  (/.well-known/oauth-protected-resource)
  RFC 8414 — Authorization Server Metadata          (/.well-known/oauth-authorization-server)
  RFC 7591 — Dynamic Client Registration            (/oauth/register)
  RFC 7636 — PKCE                                   (code_challenge / code_verifier)
  Bearer JWT — HS256 tokens signed with MCP_API_KEY (/oauth/token)

MCP_API_KEY is the "password" shown on the browser authorization page.
mcp-remote stores the resulting JWT; subsequent requests use it as a Bearer.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory stores  (single-instance; fine for a personal Mac Mini server)
# ---------------------------------------------------------------------------
_pending: Dict[str, Dict[str, Any]] = {}   # form sessions & auth codes
_clients: Dict[str, Dict[str, Any]] = {}   # dynamically registered clients


# ---------------------------------------------------------------------------
# JWT helpers (HS256 — no external dependency)
# ---------------------------------------------------------------------------

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "==")


def _issue_jwt(api_key: str, sub: str, expires_in: int = 86400 * 30) -> str:
    """Issue an HS256 JWT signed with the API key (30-day default expiry)."""
    now = int(time.time())
    header  = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64url(json.dumps({"iss": "stindex-mcp", "sub": sub,
                                   "iat": now, "exp": now + expires_in}).encode())
    signing_input = f"{header}.{payload}"
    sig = hmac.new(api_key.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url(sig)}"


def _verify_jwt(token: str, api_key: str) -> Optional[Dict[str, Any]]:
    """Verify an HS256 JWT. Returns the payload dict or None."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        signing_input = f"{parts[0]}.{parts[1]}"
        expected = hmac.new(api_key.encode(), signing_input.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64url_decode(parts[2])):
            return None
        payload: Dict[str, Any] = json.loads(_b64url_decode(parts[1]))
        if payload.get("exp", 0) < time.time():
            logger.debug("JWT expired")
            return None
        return payload
    except Exception as exc:
        logger.debug("JWT verification failed: %s", exc)
        return None


def _pkce_verify(verifier: str, challenge: str, method: str = "S256") -> bool:
    if method == "S256":
        return _b64url(hashlib.sha256(verifier.encode()).digest()) == challenge
    return verifier == challenge   # plain


# ---------------------------------------------------------------------------
# OAuthServer
# ---------------------------------------------------------------------------

class OAuthServer:
    """Self-contained OAuth 2.0 server backed by MCP_API_KEY as the password."""

    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key  = api_key

    # ------------------------------------------------------------------
    # Public helper — used by the auth middleware
    # ------------------------------------------------------------------

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Accept JWT (from OAuth flow) or raw API key (backward compat)."""
        payload = _verify_jwt(token, self.api_key)
        if payload:
            return payload
        # Fall back: direct key comparison (lets existing --header setups keep working)
        if self.api_key and hmac.compare_digest(token.encode(), self.api_key.encode()):
            return {"sub": "direct", "iss": "stindex-mcp"}
        return None

    # ------------------------------------------------------------------
    # Starlette sub-app
    # ------------------------------------------------------------------

    def build_app(self) -> Starlette:
        return Starlette(routes=[
            Route("/health",                                    self._health,               methods=["GET"]),
            Route("/.well-known/oauth-protected-resource",     self._protected_resource,    methods=["GET"]),
            Route("/.well-known/oauth-authorization-server",   self._oauth_metadata,        methods=["GET"]),
            Route("/.well-known/openid-configuration",         self._oauth_metadata,        methods=["GET"]),
            Route("/oauth/register",                           self._client_registration,   methods=["POST", "GET"]),
            Route("/oauth/authorize",                          self._authorize,             methods=["GET", "POST"]),
            Route("/oauth/token",                              self._token,                 methods=["POST"]),
        ])

    # ------------------------------------------------------------------
    # Route handlers
    # ------------------------------------------------------------------

    async def _health(self, request: Request) -> JSONResponse:
        return JSONResponse({"status": "healthy", "service": "STIndex MCP"})

    async def _protected_resource(self, request: Request) -> JSONResponse:
        """RFC 8707 — tells mcp-remote where the authorization server is."""
        return JSONResponse({
            "resource":                  self.base_url,
            "authorization_servers":     [self.base_url],
            "bearer_methods_supported":  ["header"],
        })

    async def _oauth_metadata(self, request: Request) -> JSONResponse:
        """RFC 8414 — authorization server discovery."""
        return JSONResponse({
            "issuer":                                    self.base_url,
            "authorization_endpoint":                   f"{self.base_url}/oauth/authorize",
            "token_endpoint":                           f"{self.base_url}/oauth/token",
            "registration_endpoint":                    f"{self.base_url}/oauth/register",
            "response_types_supported":                 ["code"],
            "grant_types_supported":                    ["authorization_code"],
            "code_challenge_methods_supported":         ["S256"],
            "token_endpoint_auth_methods_supported":    ["none"],
            "scopes_supported":                         ["mcp"],
        })

    async def _client_registration(self, request: Request) -> JSONResponse:
        """RFC 7591 — dynamic client registration."""
        try:
            body: Dict[str, Any] = await request.json() if request.method == "POST" else {}
        except Exception:
            body = {}
        client_id = secrets.token_urlsafe(16)
        redirect_uris: List[str] = body.get("redirect_uris", [])
        _clients[client_id] = {
            "redirect_uris": redirect_uris,
            "client_name":   body.get("client_name", "MCP Client"),
        }
        logger.info("Registered dynamic client: %s", client_id)
        return JSONResponse({
            "client_id":            client_id,
            "client_id_issued_at":  int(time.time()),
            "redirect_uris":        redirect_uris,
            "grant_types":          ["authorization_code"],
            "response_types":       ["code"],
            "token_endpoint_auth_method": "none",
        }, status_code=201)

    async def _authorize(self, request: Request) -> Response:
        """Authorization endpoint — show API-key form, issue auth code on success."""
        params = dict(request.query_params)

        if request.method == "POST":
            form          = await request.form()
            entered_key   = str(form.get("api_key", "")).strip()
            session_key   = str(form.get("_session", ""))

            session = _pending.get(session_key)
            if not session or session.get("expires_at", 0) < time.time():
                return HTMLResponse("<h1>Session expired — please try again.</h1>", status_code=400)

            orig_params = session["params"]

            if not self.api_key or not hmac.compare_digest(
                entered_key.encode(), self.api_key.encode()
            ):
                return self._authorize_page(orig_params, session_key, error="Invalid API key.")

            # Issue authorization code
            code = secrets.token_urlsafe(32)
            _pending[code] = {
                "client_id":            orig_params.get("client_id"),
                "redirect_uri":         orig_params.get("redirect_uri"),
                "code_challenge":       orig_params.get("code_challenge"),
                "code_challenge_method":orig_params.get("code_challenge_method", "S256"),
                "expires_at":           time.time() + 300,
            }
            del _pending[session_key]

            redirect_uri = orig_params.get("redirect_uri", "")
            state        = orig_params.get("state", "")
            sep          = "&" if "?" in redirect_uri else "?"
            location     = f"{redirect_uri}{sep}code={code}"
            if state:
                location += f"&state={state}"
            logger.info("Authorization code issued for client %s", orig_params.get("client_id"))
            return RedirectResponse(location, status_code=302)

        # GET — store params in session, show form
        session_key = secrets.token_urlsafe(16)
        _pending[session_key] = {"params": params, "expires_at": time.time() + 600}
        return self._authorize_page(params, session_key)

    async def _token(self, request: Request) -> JSONResponse:
        """Token endpoint — validate auth code + PKCE, return JWT."""
        try:
            ct = request.headers.get("content-type", "")
            body: Dict[str, Any] = (
                await request.json() if "application/json" in ct
                else dict(await request.form())
            )
        except Exception:
            return JSONResponse({"error": "invalid_request"}, status_code=400)

        code          = str(body.get("code", ""))
        code_verifier = str(body.get("code_verifier", ""))

        pending = _pending.pop(code, None)
        if not pending or pending.get("expires_at", 0) < time.time():
            return JSONResponse(
                {"error": "invalid_grant", "error_description": "Invalid or expired code"},
                status_code=400,
            )

        challenge = pending.get("code_challenge")
        method    = pending.get("code_challenge_method", "S256")
        if challenge and not _pkce_verify(code_verifier, challenge, method):
            return JSONResponse(
                {"error": "invalid_grant", "error_description": "PKCE verification failed"},
                status_code=400,
            )

        expires_in   = 86400 * 30   # 30 days
        client_id    = pending.get("client_id") or "mcp-client"
        access_token = _issue_jwt(self.api_key, sub=client_id, expires_in=expires_in)
        logger.info("Access token issued for client %s", client_id)

        return JSONResponse({
            "access_token": access_token,
            "token_type":   "Bearer",
            "expires_in":   expires_in,
            "scope":        "mcp",
        })

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------

    def _authorize_page(self, params: dict, session_key: str, error: str = "") -> HTMLResponse:
        query      = urlencode(params)
        error_html = f'<p class="err">{error}</p>' if error else ""
        return HTMLResponse(f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>STIndex MCP — Authorize</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body   {{ font-family: system-ui, -apple-system, sans-serif;
              display: flex; align-items: center; justify-content: center;
              min-height: 100vh; margin: 0; background: #f5f5f5; }}
    .card  {{ background: #fff; border-radius: 10px; padding: 2rem 2.5rem;
              box-shadow: 0 2px 16px rgba(0,0,0,.12); width: 360px; }}
    h1     {{ margin: 0 0 .4rem; font-size: 1.3rem; }}
    p      {{ color: #555; font-size: .9rem; margin: 0 0 1rem; }}
    .err   {{ color: #c0392b; }}
    label  {{ display: block; font-size: .85rem; font-weight: 600;
              margin-bottom: .3rem; }}
    input  {{ width: 100%; padding: .55rem .75rem; font-size: 1rem;
              border: 1px solid #ccc; border-radius: 6px; }}
    input:focus {{ outline: none; border-color: #0070f3;
                   box-shadow: 0 0 0 2px rgba(0,112,243,.2); }}
    button {{ margin-top: 1rem; width: 100%; padding: .6rem;
              font-size: 1rem; font-weight: 600; cursor: pointer;
              background: #0070f3; color: #fff; border: none;
              border-radius: 6px; }}
    button:hover {{ background: #0058c7; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>STIndex MCP</h1>
    <p>Enter your <code>MCP_API_KEY</code> to authorize this client.</p>
    {error_html}
    <form method="POST" action="/oauth/authorize?{query}">
      <input type="hidden" name="_session" value="{session_key}">
      <label for="api_key">API Key</label>
      <input type="password" id="api_key" name="api_key"
             placeholder="paste your MCP_API_KEY here" autofocus required>
      <button type="submit">Authorize</button>
    </form>
  </div>
</body>
</html>""")


# ---------------------------------------------------------------------------
# Auth middleware  (mirrors Talk2Metadata / Docs2Synth JWTAuthMiddleware)
# ---------------------------------------------------------------------------

class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validate Bearer tokens for protected MCP endpoints.

    Returns 401 with RFC 8707 resource_metadata header so mcp-remote can
    discover and initiate the OAuth flow automatically.
    """

    def __init__(self, app: Any, oauth_server: OAuthServer,
                 protected_paths: List[str]) -> None:
        super().__init__(app)
        self.oauth_server   = oauth_server
        self.protected_paths = protected_paths

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        if not any(request.url.path.startswith(p) for p in self.protected_paths):
            return await call_next(request)

        resource_metadata = (
            f"{self.oauth_server.base_url}/.well-known/oauth-protected-resource"
        )

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                {"error": "unauthorized",
                 "error_description": "Missing or invalid Authorization header"},
                status_code=401,
                headers={"WWW-Authenticate":
                         f'Bearer resource_metadata="{resource_metadata}"'},
            )

        token_data = self.oauth_server.verify_token(auth_header[7:])
        if not token_data:
            return JSONResponse(
                {"error": "unauthorized",
                 "error_description": "Invalid or expired token"},
                status_code=401,
                headers={"WWW-Authenticate":
                         f'Bearer resource_metadata="{resource_metadata}"'},
            )

        logger.debug("Authenticated MCP request: sub=%s", token_data.get("sub"))
        return await call_next(request)


# ---------------------------------------------------------------------------
# ASGI router — OAuth paths go to OAuthServer, everything else to MCP
# ---------------------------------------------------------------------------

_OAUTH_PREFIXES = ("/.well-known/", "/oauth/", "/health")


class OAuthASGIRouter:
    """Thin ASGI router: OAuth/discovery → oauth_app, MCP traffic → mcp_app."""

    def __init__(self, mcp_app: Any, oauth_app: Any) -> None:
        self.mcp_app   = mcp_app
        self.oauth_app = oauth_app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            path = scope.get("path", "")
            if any(path == p or path.startswith(p) for p in _OAUTH_PREFIXES):
                await self.oauth_app(scope, receive, send)
                return
        await self.mcp_app(scope, receive, send)
