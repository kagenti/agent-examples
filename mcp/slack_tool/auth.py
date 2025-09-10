import os
import sys
import logging
import httpx
from fastmcp.server.auth.providers.jwt import JWTVerifier
from fastmcp.server.auth.auth import AuthProvider, AccessToken

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "DEBUG"), stream=sys.stdout, format='%(levelname)s: %(message)s')


class SimpleIntrospectionAuthProvider(AuthProvider):
    """Simple JWT verifier for FastMCP."""
    def __init__(self, introspection_endpoint=None, 
                 client_id=None, 
                 client_secret=None,
                 expected_audience=None,
                 base_url=None,
                 required_scopes=None):
        self.introspection_endpoint = introspection_endpoint
        self.client_id = client_id
        self.client_secret = client_secret
        self.expected_audience = expected_audience
        self.base_url = base_url
        self.required_scopes = required_scopes or []

    def _validate_resource(self, data) -> bool:
        # check aud claim in data
        if self.expected_audience is None:
            logger.warning(f"No AUDIENCE env var set - failing resource validation")
            return False
        if not "aud" in data:
            logger.error(f"No aud claim in token - failing resource validation")
            return False
        else:
            # needs client id
            audiences = data["aud"]
            return self.expected_audience in audiences

    def _dummy_token(self, token: str) -> AccessToken:
        return AccessToken(
            token=token,
            client_id="hard-coded-client-id",
            scopes=[],
            expires_at=None,
            claims={"sub": "hard-coded-client-id", "client_id": "hard-coded-client-id"}
        )

    def _verify_token(self, token: str) -> AccessToken | None:
        timeout = httpx.Timeout(10.0, connect=5.0)
        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
        with httpx.Client (
            timeout=timeout,
            limits=limits,
            verify=False,
        ) as client:
            try:
                response = client.post(
                    self.introspection_endpoint,
                    data={
                        "token": token,
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                logger.debug(f"Response to token introspection: {response}")

                if response.status_code != 200:
                    logger.error(f"Token introspection returned status {response.status_code}")
                    return None

                data = response.json()
                logger.debug(f"data: {data}")
                if not data.get("active", False):
                    return None

                # RFC 8707 resource validation
                if self.expected_audience is None:
                    logger.warning(f"No expected audience set. Skipping token resource validation")
                elif not self._validate_resource(data):
                    logger.warning(f"Token resource validation failed. Expected: {self.expected_audience}")
                    return None

                access_token = AccessToken(
                    token=token,
                    client_id=data.get("client_id", "unknown"),
                    scopes=data.get("scope", "").split() if data.get("scope") else [],
                    expires_at=data.get("exp"),
                    claims=data  # Include all original claims
                )
                return access_token
            except Exception as e:
                logger.error(f"Token introspection failed: {e}")
                return None

    async def verify_token(self, token: str) -> AccessToken | None:
        logger.info(f"Received Access Token: {token}")
        if self.client_id is None: # no verification
            return self._dummy_token(token)
        # if id and secret are defined, verify against issuer
        return self._verify_token(token)

def get_auth_provider():
    introspection_endpoint = os.getenv("INTROSPECTION_ENDPOINT")
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")
    expected_audience = os.getenv("AUDIENCE")
    issuer = os.getenv("ISSUER")
    
    if introspection_endpoint is None:
        return None
    else:
        return SimpleIntrospectionAuthProvider(introspection_endpoint=introspection_endpoint, 
                                             client_id=client_id,
                                             client_secret=client_secret,
                                             expected_audience=expected_audience,
                                             base_url=issuer)
