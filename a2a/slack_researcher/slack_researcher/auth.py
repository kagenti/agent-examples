import httpx
import logging
import sys

from starlette.responses import JSONResponse
from starlette.requests import Request
from starlette.authentication import AuthCredentials, SimpleUser, AuthenticationBackend
from starlette.authentication import AuthenticationError as StarletteAuthenticationError

from authlib.jose import jwt
from authlib.common.errors import AuthlibBaseError

from slack_researcher.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.LOG_LEVEL, stream=sys.stdout, format='%(levelname)s: %(message)s')

def on_auth_error(request: Request, e: Exception):
    status_code = 401
    message = "Authentication failed"
    headers = {"WWW-Authenticate": "Bearer"}

    if isinstance(e, AuthenticationError):
        status_code = e.status_code
    message = str(e)

    return JSONResponse(
        {"error": message},
        status_code = status_code,
        headers=headers if status_code == 401 else None
    )

class AuthenticationError(StarletteAuthenticationError):
    def __init__(self, message: str, status_code: int = 401):
        self.status_code = status_code
        super().__init__(message)

class BearerAuthBackend(AuthenticationBackend):
    def __init__(self):
        if settings.JWKS_URI is None: # TODO implement oidc discovery in this case
            raise Exception("JWKS_URI env var not set. ")
        self.jwks_url = settings.JWKS_URI

        self.claims_options = {}

        if settings.CLIENT_ID is None:
            logger.debug(f"CLIENT_ID is not set. No audience check will be performed. ")
        else:
            self.claims_options["aud"] = {"essential": True, "value": settings.CLIENT_ID}
        if settings.ISSUER is None:
            logger.debug(f"ISSUER env var no set. No issuer check will be performed")
        else:
            self.claims_options["iss"] = {"essential": True, "value": settings.ISSUER}

    async def get_jwks(self):
        logger.debug(f"Fetching JWKS from {self.jwks_url}")
        jwks = None
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.jwks_url)
                response.raise_for_status()
                jwks = response.json() # TODO should probably use caching
                return jwks
        except httpx.HTTPStatusError as e:
            logger.debug(f"Could not retrieve JWKS from {self.jwks_url}: {e}")
            return None

    async def get_token(self, conn):
        logger.debug("Obtaining bearer token...")
        headers = conn.headers
        logger.debug(f"Headers obtained: {headers}")
        auth = headers.get("authorization")
        if not auth or not auth.lower().startswith("bearer "):
            logger.error("Expected `Authorization: Bearer` access token; None provided. ")
            return None
        token = auth.split(" ", 1)[1]
        return token

    """This is run upon every received request"""
    async def authenticate(self, conn):
        # bypass authentication for agent card
        if conn.scope.get("path") == "/.well-known/agent.json":
            logger.debug("Bypassing authentication for public agent card path")
            return None

        # extract token
        token = await self.get_token(conn)
        if token is None:
            raise AuthenticationError(message = "Bearer token not found in Authorization header.")

        # fetch jwks
        jwks = await self.get_jwks()

        try:
            # decode and validate claims
            claims = jwt.decode(s=token, key=jwks, claims_options=self.claims_options)
            claims.validate()
            logger.debug("Token successfully validated.")

            # return user
            user = AgentUser(token=token, claims=claims)
            return AuthCredentials(user.scopes()), user
        except AuthlibBaseError as e:
            logger.error(f"Token validation failed: {e}")
            raise AuthenticationError(f"Invalid token: {e}, status_code=401")

class AgentUser(SimpleUser):
    def __init__(self, token, claims) -> None:
        super().__init__(username=claims.get("sub"))
        self.access_token = token
        self.claims = claims

    def scopes(self) -> list[str]:
        scope = self.claims.get("scope", "")
        return scope.split()

class TokenExchanger:
    def __init__(self):
        if None in [settings.TOKEN_URL, settings.CLIENT_ID, settings.CLIENT_SECRET]:
            raise Exception("One of TOKEN_URL, CLIENT_ID, CLIENT_SECRET env vars not set - token exchange will not be performed")
        self.token_url = settings.TOKEN_URL
        self.client_id = settings.CLIENT_ID
        self.client_secret = settings.CLIENT_SECRET

    async def exchange(self, subject_token: str, audience: str = None, scope: str = None) -> str:
        # headers
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        # data
        data = {
            'grant_type': 'urn:ietf:params:oauth:grant-type:token-exchange',
            'subject_token_type': 'urn:ietf:params:oauth:token-type:access_token',
            'requested_token_type': 'urn:ietf:params:oauth:token-type:access_token',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'subject_token': subject_token,
        }
        if not audience is None:
            data['audience'] = audience
        if not scope is None:
            data['scope'] = scope
        # make token endpoint call
        logger.debug('Performing token exchange')
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.token_url, data=data, headers=headers)
                response.raise_for_status() # raise exception if Http status error
                token_data = response.json()
                if "access_token" in token_data:
                    new_token = token_data["access_token"]
                    logger.debug(f"Successful token exchange. Using token: {new_token}")
                    return new_token
                logger.error("Token exchange failed.")
                raise AuthenticationError("Token exchange failed. Identity provider response did not include 'access_token'")
            except httpx.HTTPStatusError as e:
                logger.error(f"Token exchange failed with status {e.response.status_code}: {e}")
                raise AuthenticationError("Token endpoint call failed.")

async def auth_headers(access_token, target_audience = None, target_scopes = None):
    headers = {}
    if not access_token:
        return headers
    try:
        token_exchanger = TokenExchanger()
        access_token = await token_exchanger.exchange(access_token, audience=target_audience, scope=target_scopes)
    except AuthenticationError as e:
        logging.error(f"Error performing token exchange - returning empty headers: {e}")
        return headers #
    except Exception as e:
        logging.debug(f"Error creating token exchanger - will passthrough token")

    headers["Authorization"] = f"Bearer {access_token}"
    return headers