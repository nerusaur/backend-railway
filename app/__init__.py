"""
ChildFocus - Flask App Factory
backend/app/__init__.py
"""

import threading
import time
from flask import Flask
from app.config import DevelopmentConfig


def create_app(config=DevelopmentConfig):
    app = Flask(__name__)
    app.config.from_object(config)

    # ── Auto-load cookies from env on startup ──────────────────────────────
    from app.utils.cookie_manager import ensure_cookies, load_cookies_from_env
    ensure_cookies()

    # ── Refresh cookies every 4 hours using background thread ─────────────
    def _cookie_refresh_loop():
        while True:
            time.sleep(1* 60)
            load_cookies_from_env()

    thread = threading.Thread(target=_cookie_refresh_loop, daemon=True)
    thread.start()

    # Register blueprints
    from app.routes.classify import classify_bp
    from app.routes.metadata import metadata_bp
    from app.routes.cookies import cookies_bp

    app.register_blueprint(classify_bp)
    app.register_blueprint(metadata_bp)
    app.register_blueprint(cookies_bp)

    return app
