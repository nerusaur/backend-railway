"""
ChildFocus - Flask App Factory
backend/app/__init__.py
"""

from flask import Flask
from app.config import DevelopmentConfig


def create_app(config=DevelopmentConfig):
    app = Flask(__name__)
    app.config.from_object(config)

    # ── Auto-load cookies from env on startup ──────────────────────────────
    from app.utils.cookie_manager import ensure_cookies
    ensure_cookies()

    # Register blueprints
    from app.routes.classify import classify_bp
    from app.routes.metadata import metadata_bp
    from app.routes.cookies import cookies_bp       # <-- ADD THIS

    app.register_blueprint(classify_bp)
    app.register_blueprint(metadata_bp)
    app.register_blueprint(cookies_bp)              # <-- ADD THIS

    return app
