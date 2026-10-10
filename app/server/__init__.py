"""零依赖 HTTP 服务。"""

from .http_app import Application, create_server, serve_forever

__all__ = ["Application", "create_server", "serve_forever"]
