"""Deployment-facing Streamlit controls must not drift back to broad defaults."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_streamlit_network_and_upload_controls_are_explicit():
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    server = config["server"]

    assert server["enableCORS"] is True
    assert server["enableXsrfProtection"] is True
    assert server["xsrfCookieSameSite"] == "lax"
    assert server["allowedHosts"] == [
        "localhost", "127.0.0.1", "*.onrender.com", "*.streamlit.app",
    ]
    assert server["maxUploadSize"] == 12
    assert 12 <= server["maxMessageSize"] <= 16
