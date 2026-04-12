"""
ChildFocus - Cookie Manager
app/utils/cookie_manager.py
Loads cookies.txt from env var on startup and checks staleness.
"""

import os
import time
import logging

# Same root path as frame_sampler.py uses
COOKIES_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "cookies.txt")
COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 7  # 7 days


def is_cookie_stale() -> bool:
    if not os.path.isfile(COOKIES_PATH) or os.path.getsize(COOKIES_PATH) == 0:
        return True
    age = time.time() - os.path.getmtime(COOKIES_PATH)
    return age > COOKIE_MAX_AGE_SECONDS


def load_cookies_from_env() -> bool:
    content = os.environ.get("YOUTUBE_COOKIES", "").strip()
    if not content:
        logging.warning("[COOKIES] YOUTUBE_COOKIES env var not set or empty.")
        return False

    # Ensure Netscape header is present — required by yt-dlp
    if not content.startswith("# Netscape HTTP Cookie File"):
        content = "# Netscape HTTP Cookie File\n" + content

    os.makedirs(os.path.dirname(os.path.abspath(COOKIES_PATH)), exist_ok=True)
    with open(COOKIES_PATH, "w") as f:
        f.write(content)
    logging.info("[COOKIES] cookies.txt written from YOUTUBE_COOKIES env var.")
    return True


def ensure_cookies():
    if is_cookie_stale():
        logging.warning("[COOKIES] cookies.txt missing or stale — reloading from env.")
        load_cookies_from_env()
    else:
        logging.info("[COOKIES] cookies.txt is fresh, skipping reload.")
