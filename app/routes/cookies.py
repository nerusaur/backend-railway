"""
ChildFocus - Cookie Refresh Route
app/routes/cookies.py
Protected endpoint to push fresh cookies.txt from local machine.
"""

import os
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
@cookies_bp.route("/debug-secret", methods=["GET"])
def debug_secret():
    secret = os.environ.get("REFRESH_SECRET", "NOT SET")
    return jsonify({
        "secret_length": len(secret),
        "secret_value": secret  # remove this line after debugging
    }), 200
