"""OIDC proxy mode for STIndex MCP server.

When OIDC_DISCOVERY_URL is set, STIndex acts as an OIDC Resource Server
with KAIAPlatform (Django OAuth Toolkit) as the Authorization Server.

Follows the Talk2Metadata pattern:
  /oauth/authorize  — captures mcp-remote's redirect_uri, rewrites to STIndex
                      callback, then forwards to KAIA's authorize endpoint
  /oauth/callback   — KAIA redirects here; decodes original redirect_uri from
                      state and sends the code back to mcp-remote
  /oauth/token      — proxies to KAIA's token endpoint, injecting client creds
  /oauth/register   — dynamic client registration (self-contained)
  /.well-known/*    — serves OAuth metadata pointing to STIndex proxy endpoints;
                      token validation uses KAIA's JWKS / introspection
"""

from __future__ import annotations

import base64
import logging
import os
import secrets
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, urlparse

import httpx
import jwt
from jwt import PyJWKClient
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Route

logger = logging.getLogger(__name__)

_clients: Dict[str, Any] = {}   # dynamically registered clients


# ---------------------------------------------------------------------------
# OIDC Resource Server  (validates tokens against KAIA's JWKS / introspection)
# Ported from Talk2Metadata's oidc_client.py
# ---------------------------------------------------------------------------

class OIDCResourceServer:
    def __init__(
        self,
        discovery_url: str,
        client_id: str,
        client_secret: str,
        use_introspection: bool = True,
        verify_ssl: bool = True,
        timeout: float = 30.0,
    ) -> None:
        self.discovery_url    = discovery_url
        self.client_id        = client_id
        self.client_secret    = client_secret
        self.use_introspection = use_introspection
        self.verify_ssl       = verify_ssl
        self.timeout          = timeout

        # Discovered
        self.issuer:                  Optional[str] = None
        self.authorization_endpoint:  Optional[str] = None
        self.token_endpoint:          Optional[str] = None
        self.jwks_uri:                Optional[str] = None
        self.introspection_endpoint:  Optional[str] = None
        self.jwks_client:             Optional[PyJWKClient] = None

    def _fix_url(self, url: str, base: str) -> str:
        """Rewrite host to match the discovery URL base (handles Docker networking)."""
        p, b = urlparse(url), urlparse(base)
        return f"{b.scheme}://{b.netloc}{p.path}"

    async def discover(self) -> bool:
        if self.issuer:
            return True
        try:
            async with httpx.AsyncClient(verify=self.verify_ssl) as c:
                r = await c.get(self.discovery_url, timeout=self.timeout)
                if r.status_code != 200:
                    logger.error("OIDC discovery failed: HTTP %s", r.status_code)
                    return False
                cfg = r.json()
                base = self.discovery_url.split("/.well-known")[0]
                self.issuer                 = cfg.get("issuer")
                self.authorization_endpoint = cfg.get("authorization_endpoint")
                self.token_endpoint         = cfg.get("token_endpoint")
                self.jwks_uri               = self._fix_url(cfg.get("jwks_uri", ""), base)
                ep = cfg.get("introspection_endpoint", f"{base}/introspect/")
                self.introspection_endpoint = self._fix_url(ep, base)
                if not self.use_introspection and self.jwks_uri:
                    self.jwks_client = PyJWKClient(self.jwks_uri)
                logger.info("OIDC discovered: issuer=%s", self.issuer)
                return True
        except Exception as exc:
            logger.error("OIDC discovery error: %s", exc)
            return False

    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        if not await self.discover():
            return None
        return (
            await self._introspect(token)
            if self.use_introspection
            else await self._verify_jwt(token)
        )

    async def _introspect(self, token: str) -> Optional[Dict[str, Any]]:
        try:
            async with httpx.AsyncClient(verify=self.verify_ssl) as c:
                r = await c.post(
                    self.introspection_endpoint,
                    data={"token": token, "client_id": self.client_id,
                          "client_secret": self.client_secret},
                    timeout=self.timeout,
                )
                if r.status_code != 200:
                    return None
                data = r.json()
                return data if data.get("active") else None
        except Exception as exc:
            logger.error("Introspection error: %s", exc)
            return None

    async def _verify_jwt(self, token: str) -> Optional[Dict[str, Any]]:
        if not self.jwks_client:
            return None
        try:
            key = self.jwks_client.get_signing_key_from_jwt(token)
            return jwt.decode(token, key.key, algorithms=["RS256", "HS256"],
                              issuer=self.issuer,
                              options={"verify_signature": True,
                                       "verify_exp": True, "verify_iss": True})
        except jwt.ExpiredSignatureError:
            logger.warning("JWT expired")
            return None
        except jwt.InvalidTokenError as exc:
            logger.warning("Invalid JWT: %s", exc)
            return None


# ---------------------------------------------------------------------------
# OIDC Proxy Server  (OAuth endpoints that proxy to KAIA)
# ---------------------------------------------------------------------------

class OIDCProxyServer:
    """OAuth 2.0 proxy: STIndex acts as middleware between mcp-remote and KAIA."""

    def __init__(self, base_url: str, oidc: OIDCResourceServer) -> None:
        self.base_url = base_url.rstrip("/")
        self.oidc     = oidc

    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        return await self.oidc.verify_token(token)

    def build_app(self) -> Starlette:
        return Starlette(routes=[
            Route("/health",                                   self._health,             methods=["GET"]),
            Route("/.well-known/oauth-protected-resource",    self._protected_resource, methods=["GET"]),
            Route("/.well-known/oauth-authorization-server",  self._oauth_metadata,     methods=["GET"]),
            Route("/.well-known/openid-configuration",        self._oauth_metadata,     methods=["GET"]),
            Route("/oauth/register",  self._client_registration, methods=["POST", "GET"]),
            Route("/oauth/authorize", self._authorize,            methods=["GET"]),
            Route("/oauth/callback",  self._callback,             methods=["GET"]),
            Route("/oauth/token",     self._token,                methods=["POST"]),
        ])

    async def _health(self, request: Request) -> JSONResponse:
        return JSONResponse({"status": "healthy", "service": "STIndex MCP"})

    async def _protected_resource(self, request: Request) -> JSONResponse:
        """RFC 8707 — points mcp-remote at this server as the authorization server."""
        return JSONResponse({
            "resource":                 self.base_url,
            "authorization_servers":    [self.base_url],
            "bearer_methods_supported": ["header"],
        })

    async def _oauth_metadata(self, request: Request) -> JSONResponse:
        """RFC 8414 — STIndex proxy endpoints (not KAIA's directly)."""
        return JSONResponse({
            "issuer":                               self.base_url,
            "authorization_endpoint":              f"{self.base_url}/oauth/authorize",
            "token_endpoint":                      f"{self.base_url}/oauth/token",
            "registration_endpoint":               f"{self.base_url}/oauth/register",
            "response_types_supported":            ["code"],
            "grant_types_supported":               ["authorization_code"],
            "code_challenge_methods_supported":    ["S256"],
            "token_endpoint_auth_methods_supported": ["none"],
            "scopes_supported":                    ["openid", "profile", "email", "read"],
        })

    async def _client_registration(self, request: Request) -> JSONResponse:
        """RFC 7591 — self-contained dynamic registration (not forwarded to KAIA)."""
        try:
            body: Dict[str, Any] = await request.json() if request.method == "POST" else {}
        except Exception:
            body = {}
        client_id     = secrets.token_urlsafe(16)
        redirect_uris: List[str] = body.get("redirect_uris", [])
        _clients[client_id] = {"redirect_uris": redirect_uris,
                                "client_name": body.get("client_name", "MCP Client")}
        return JSONResponse({
            "client_id":                  client_id,
            "client_id_issued_at":        int(time.time()),
            "redirect_uris":              redirect_uris,
            "grant_types":                ["authorization_code"],
            "response_types":             ["code"],
            "token_endpoint_auth_method": "none",
        }, status_code=201)

    async def _authorize(self, request: Request) -> Response:
        """Capture mcp-remote's redirect_uri, encode in state, forward to KAIA."""
        if not await self.oidc.discover():
            return JSONResponse({"error": "oidc_unavailable"}, status_code=503)

        params = dict(request.query_params)

        # Encode the original redirect_uri into the state so _callback can recover it
        original_redirect_uri = params.get("redirect_uri", "")
        original_state        = params.get("state", "")
        encoded_uri           = base64.urlsafe_b64encode(
            original_redirect_uri.encode()
        ).rstrip(b"=").decode()
        composite_state       = f"{original_state}|{encoded_uri}"

        # Build the upstream authorize URL pointing back to our /oauth/callback
        upstream_params = {
            **params,
            "redirect_uri": f"{self.base_url}/oauth/callback",
            "state":        composite_state,
        }
        upstream_url = f"{self.oidc.authorization_endpoint}?{urlencode(upstream_params)}"
        logger.info("Proxying authorize to KAIA: %s", self.oidc.authorization_endpoint)
        return RedirectResponse(upstream_url, status_code=302)

    async def _callback(self, request: Request) -> Response:
        """KAIA redirects here after login — extract code, send to mcp-remote."""
        params        = dict(request.query_params)
        code          = params.get("code", "")
        composite     = params.get("state", "")

        # Decode original state and redirect_uri
        if "|" in composite:
            original_state, encoded_uri = composite.rsplit("|", 1)
            padding = "=" * (4 - len(encoded_uri) % 4)
            original_redirect_uri = base64.urlsafe_b64decode(
                encoded_uri + padding
            ).decode()
        else:
            original_state        = composite
            original_redirect_uri = ""

        if not original_redirect_uri:
            return JSONResponse({"error": "missing_redirect_uri"}, status_code=400)

        sep      = "&" if "?" in original_redirect_uri else "?"
        location = f"{original_redirect_uri}{sep}code={code}"
        if original_state:
            location += f"&state={original_state}"

        logger.info("OAuth callback — redirecting code to mcp-remote")
        return RedirectResponse(location, status_code=302)

    async def _token(self, request: Request) -> Response:
        """Proxy token exchange to KAIA, injecting client credentials server-side."""
        if not await self.oidc.discover():
            return JSONResponse({"error": "oidc_unavailable"}, status_code=503)

        try:
            ct = request.headers.get("content-type", "")
            body: Dict[str, Any] = (
                await request.json() if "application/json" in ct
                else dict(await request.form())
            )
        except Exception:
            return JSONResponse({"error": "invalid_request"}, status_code=400)

        # Replace the client's redirect_uri with our callback (must match what we sent)
        upstream_data = {
            **body,
            "redirect_uri":  f"{self.base_url}/oauth/callback",
            "client_id":     self.oidc.client_id,
            "client_secret": self.oidc.client_secret,
        }

        try:
            async with httpx.AsyncClient(verify=self.oidc.verify_ssl) as c:
                r = await c.post(
                    self.oidc.token_endpoint,
                    data=upstream_data,
                    timeout=self.oidc.timeout,
                )
            logger.info("Token exchange: HTTP %s from KAIA", r.status_code)
            return Response(content=r.content, status_code=r.status_code,
                            headers={"Content-Type": r.headers.get(
                                "content-type", "application/json")})
        except Exception as exc:
            logger.error("Token proxy error: %s", exc)
            return JSONResponse({"error": "token_proxy_failed"}, status_code=502)


# ---------------------------------------------------------------------------
# Auth middleware (async verify_token for OIDC)
# ---------------------------------------------------------------------------

class OIDCAuthMiddleware(BaseHTTPMiddleware):
    """Validate KAIA Bearer tokens for protected MCP endpoints."""

    def __init__(self, app: Any, proxy: OIDCProxyServer,
                 protected_paths: List[str]) -> None:
        super().__init__(app)
        self.proxy           = proxy
        self.protected_paths = protected_paths

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        if not any(request.url.path.startswith(p) for p in self.protected_paths):
            return await call_next(request)

        resource_metadata = (
            f"{self.proxy.base_url}/.well-known/oauth-protected-resource"
        )
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                {"error": "unauthorized",
                 "error_description": "Missing Authorization header"},
                status_code=401,
                headers={"WWW-Authenticate":
                         f'Bearer resource_metadata="{resource_metadata}"'},
            )

        token_data = await self.proxy.verify_token(auth_header[7:])
        if not token_data:
            return JSONResponse(
                {"error": "unauthorized",
                 "error_description": "Invalid or expired token"},
                status_code=401,
                headers={"WWW-Authenticate":
                         f'Bearer resource_metadata="{resource_metadata}"'},
            )

        logger.debug("KAIA token verified: sub=%s", token_data.get("sub"))
        return await call_next(request)
