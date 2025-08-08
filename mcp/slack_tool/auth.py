import os
from pydantic import AnyHttpUrl
from mcp.server.auth.settings import AuthSettings
from mcp.server.auth.provider import AccessToken, TokenVerifier

class SimpleTokenVerifier(TokenVerifier):
    """Simple token verifier for demonstration."""

    async def verify_token(self, token: str) -> AccessToken | None:
        print(token)
        return AccessToken(
                    token=token,
                    client_id="hard-coded-client-id",
                    scopes=[],
                    expires_at=None,
                    resource=None,  # Include resource in token
                )

def get_token_verifier():
    issuer = os.getenv("ISSUER")
    if issuer is None:
        return None
    else:
        return SimpleTokenVerifier()

def get_auth():
    issuer = os.getenv("ISSUER")
    if issuer is None:
        return None
    else:
        return AuthSettings(
                issuer_url=AnyHttpUrl(issuer),  # Authorization Server URL
                resource_server_url=AnyHttpUrl("http://localhost:8000"),  # TODO This server's URL
                required_scopes=[],
              )
