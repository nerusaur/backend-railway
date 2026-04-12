"""
ChildFocus - Cookie Refresh Route
app/routes/cookies.py
Protected endpoint to push fresh cookies.txt from local machine.
"""

import os
import time
from flask import Blueprint, request, jsonify

cookies_bp = Blueprint("cookies", __name__)

COOKIES_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "cookies.txt")


@cookies_bp.route("/refresh-cookies", methods=["POST"])
def refresh_cookies():
    secret = os.environ.get("REFRESH_SECRET", "")
    if not secret or request.headers.get("X-Secret") != secret:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_data()
    if not data:
        return jsonify({"error": "No cookie data provided"}), 400

    with open(COOKIES_PATH, "wb") as f:
        f.write(data)

    return jsonify({"status": "cookies updated"}), 200


@cookies_bp.route("/cookie-status", methods=["GET"])
def cookie_status():
    exists   = os.path.isfile(COOKIES_PATH)
    size     = os.path.getsize(COOKIES_PATH) if exists else 0
    modified = os.path.getmtime(COOKIES_PATH) if exists else None
    return jsonify({
        "cookies_exists":     exists,
        "cookies_size_bytes": size,
        "last_modified":      time.strftime('%Y-%m-%d %H:%M:%S',
                              time.localtime(modified)) if modified else None
    }), 200
