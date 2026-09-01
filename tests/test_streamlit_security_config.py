"""Deployment-facing Streamlit controls must not drift back to broad defaults.

One of them is asserted the other way round, and the comment on it is the whole
lesson: a security control that can silently brick the deployment it protects
belongs where somebody can verify it, not in a file that ships everywhere.
"""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))


def test_streamlit_network_and_upload_controls_are_explicit():
    server = CONFIG["server"]

    assert server["enableCORS"] is True
    assert server["enableXsrfProtection"] is True
    assert server["xsrfCookieSameSite"] == "lax"
    assert server["maxUploadSize"] == 12
    assert 12 <= server["maxMessageSize"] <= 16


def test_the_committed_config_does_not_pin_a_host_allow_list():
    """`server.allowedHosts` is an allow-list on the WebSocket Host header.

    A value that does not match what the platform's proxy forwards does not
    degrade — it 403s the socket, and a Streamlit app whose socket is refused
    renders as a permanent "Please wait…" with nothing in the page to say why.
    Streamlit's own default is an empty list, and its documentation gives the
    reason: to preserve compatibility with dynamically configured reverse
    proxies and custom domains.

    The public demo runs on Streamlit Community Cloud, whose internal forwarded
    Host this repository cannot verify from outside it. Pinning an unverifiable
    allow-list here would put a recruiter-facing link one proxy change away from
    a blank page, in order to defend synthetic data behind no login against DNS
    rebinding. That is the wrong trade, and this test is what stops a later
    hardening pass from quietly making it again.
    """
    assert "allowedHosts" not in CONFIG["server"], (
        "pin the host allow-list in the deployment that can verify its own "
        "proxy (see the Dockerfile), never in the config that ships everywhere")


def test_the_container_that_knows_its_hostname_does_pin_one():
    """The other half. The control is not dropped, it is relocated to the
    deployment that can check it — which is what makes it a control rather than
    a hope."""
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    match = re.search(r'STREAMLIT_SERVER_ALLOWED_HOSTS="([^"]+)"', dockerfile)
    assert match, "the image must pin the host allow-list it can verify"
    hosts = [h.strip() for h in match.group(1).split(",")]
    assert "localhost" in hosts
    assert any(h.startswith("*.") for h in hosts), "the served domain must be named"
    assert "*" not in hosts, "a wildcard allow-list is not an allow-list"
