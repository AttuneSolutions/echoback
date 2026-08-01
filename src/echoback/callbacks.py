"""`callback_url` admission policy.

The service POSTs job results to a URL the caller supplies, which makes it a
server-side request forgery primitive for anyone holding the API token: without a
policy, `callback_url` can name the cloud metadata endpoint, the loopback
interface (including this service's own whisper-server), or any host the container
can reach.

The policy is deliberately narrow. Private ranges (RFC1918) are **allowed** — a
self-hosted Activepieces on the same LAN is the expected receiver — while loopback,
link-local (which is what blocks 169.254.169.254), unspecified, multicast and
reserved addresses are refused. Hostnames are resolved and every address they
resolve to must pass, so a DNS name pointing at a blocked address is refused too.
`CALLBACK_ALLOWED_HOSTS` narrows this further to a named set of receivers.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

from .config import Config

INVALID_CALLBACK_URL = "INVALID_CALLBACK_URL"
CALLBACK_HOST_NOT_ALLOWED = "CALLBACK_HOST_NOT_ALLOWED"

IpAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


class CallbackUrlError(Exception):
    """A `callback_url` the service refuses to accept.

    ``message`` is returned to the caller, so it names the host but never the full
    URL — callback URLs are capability URLs and their path is the secret.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def resolve_ips(host: str) -> list[str]:
    """Every address ``host`` resolves to. Patched out in tests; blocking, so call
    this from a worker thread rather than the event loop."""
    infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return [str(info[4][0]) for info in infos]


def _blocked_reason(ip: IpAddress) -> str | None:
    """Why this address is off limits, or None if it is acceptable."""
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        # ::ffff:127.0.0.1 must be judged as 127.0.0.1, not as a plain v6 address.
        ip = mapped
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        return "link-local"  # 169.254.0.0/16 and fe80::/10 — cloud metadata lives here
    if ip.is_unspecified:
        return "unspecified"
    if ip.is_multicast:
        return "multicast"
    if ip.is_reserved:
        return "reserved"
    return None


def _host_matches(host: str, allowed: str) -> bool:
    """Exact hostname match, or any subdomain when the entry starts with a dot."""
    if allowed.startswith("."):
        return host == allowed[1:] or host.endswith(allowed)
    return host == allowed


def _target_ips(host: str) -> list[IpAddress]:
    try:
        return [ipaddress.ip_address(host)]
    except ValueError:
        pass  # not a literal — resolve it
    try:
        resolved = resolve_ips(host)
    except OSError as exc:
        raise CallbackUrlError(
            CALLBACK_HOST_NOT_ALLOWED,
            f"callback_url host {host!r} could not be resolved",
        ) from exc
    addresses: list[IpAddress] = []
    for candidate in resolved:
        try:
            addresses.append(ipaddress.ip_address(candidate.split("%", 1)[0]))
        except ValueError:  # pragma: no cover — getaddrinfo returning a non-address
            continue
    if not addresses:
        raise CallbackUrlError(
            CALLBACK_HOST_NOT_ALLOWED,
            f"callback_url host {host!r} could not be resolved",
        )
    return addresses


def validate_callback_url(raw: str | None, config: Config) -> str:
    """Return the accepted URL, or raise :class:`CallbackUrlError`.

    Performs DNS resolution, so run it in a thread.
    """
    value = (raw or "").strip()
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise CallbackUrlError(INVALID_CALLBACK_URL, "callback_url must be an absolute http(s) URL")
    if parsed.username or parsed.password:
        raise CallbackUrlError(INVALID_CALLBACK_URL, "callback_url must not embed credentials")

    host = parsed.hostname.lower()

    if config.host_url:
        own = urlparse(config.host_url)
        if own.hostname and host == own.hostname.lower() and parsed.port == own.port:
            raise CallbackUrlError(
                CALLBACK_HOST_NOT_ALLOWED,
                "callback_url points back at this service (HOST_URL)",
            )

    if config.callback_allowed_hosts and not any(
        _host_matches(host, allowed) for allowed in config.callback_allowed_hosts
    ):
        raise CallbackUrlError(
            CALLBACK_HOST_NOT_ALLOWED,
            f"callback_url host {host!r} is not in CALLBACK_ALLOWED_HOSTS",
        )

    for ip in _target_ips(host):
        reason = _blocked_reason(ip)
        if reason is not None:
            raise CallbackUrlError(
                CALLBACK_HOST_NOT_ALLOWED,
                f"callback_url host {host!r} resolves to a {reason} address",
            )
    return value
