"""Guard the installed Streamlit/Starlette pair against broken image serving.

Streamlit 1.60's gzip middleware subclasses Starlette internals that changed in
Starlette 1.8; with that pair every st.image request failed with HTTP 500 while
the page itself still loaded. requirements-landmarks.txt pins Starlette for this.
"""

import asyncio

import pytest

gzip_mw = pytest.importorskip("streamlit.web.server.starlette.starlette_gzip_middleware")
from starlette.responses import Response  # noqa: E402


def test_compressible_image_response_starts_before_its_body():
    async def app(scope, receive, send):
        await Response(b"x" * 5000, media_type="image/png")(scope, receive, send)

    messages = []

    async def send(message):
        messages.append(message["type"])

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/media/demo.png",
        "headers": [(b"accept-encoding", b"gzip")],
        "query_string": b"",
    }
    asyncio.run(gzip_mw.SelectiveGZipMiddleware(app)(scope, receive, send))

    assert messages[0] == "http.response.start"
