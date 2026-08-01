from __future__ import annotations

import dataclasses

import pytest

from echoback.callbacks import CallbackUrlError, validate_callback_url
from echoback.config import Config


def check(config: Config, url: str) -> str:
    return validate_callback_url(url, config)


def refusal(config: Config, url: str) -> CallbackUrlError:
    with pytest.raises(CallbackUrlError) as excinfo:
        validate_callback_url(url, config)
    return excinfo.value


# ---- shape --------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_url", ["", "   ", "not-a-url", "ftp://receiver.test/x", "/relative/path", "http://"]
)
def test_malformed_urls_are_refused(config: Config, bad_url: str) -> None:
    assert refusal(config, bad_url).code == "INVALID_CALLBACK_URL"


def test_embedded_credentials_are_refused(config: Config) -> None:
    error = refusal(config, "https://user:pass@receiver.test/resume")
    assert error.code == "INVALID_CALLBACK_URL"
    assert "credentials" in error.message


def test_ordinary_public_receiver_is_accepted(config: Config) -> None:
    assert check(config, "https://receiver.test/resume/abc") == "https://receiver.test/resume/abc"


# ---- address policy -----------------------------------------------------


@pytest.mark.parametrize(
    "url,reason",
    [
        ("http://127.0.0.1:8910/inference", "loopback"),
        ("http://127.10.20.30/x", "loopback"),
        ("http://[::1]/x", "loopback"),
        ("http://169.254.169.254/latest/meta-data/", "link-local"),
        ("http://[fe80::1]/x", "link-local"),
        ("http://0.0.0.0/x", "unspecified"),
        ("http://239.1.2.3/x", "multicast"),
        ("http://[::ffff:127.0.0.1]/x", "loopback"),
    ],
)
def test_blocked_address_literals(config: Config, url: str, reason: str) -> None:
    error = refusal(config, url)
    assert error.code == "CALLBACK_HOST_NOT_ALLOWED"
    assert reason in error.message


@pytest.mark.parametrize(
    "url",
    [
        "http://10.4.0.9:8080/resume",  # the Activepieces-on-the-same-LAN case
        "http://192.168.1.50/resume",
        "http://172.20.0.7/resume",
        "https://203.0.113.10/resume",
    ],
)
def test_private_and_public_literals_are_allowed(config: Config, url: str) -> None:
    assert check(config, url) == url


def test_a_hostname_resolving_to_metadata_is_refused(config: Config) -> None:
    error = refusal(config, "https://metadata.evil.test/resume")
    assert error.code == "CALLBACK_HOST_NOT_ALLOWED"
    assert "link-local" in error.message


def test_every_resolved_address_must_pass(config: Config) -> None:
    """A host with one acceptable and one blocked answer is refused outright."""
    assert refusal(config, "https://sneaky.evil.test/resume").code == "CALLBACK_HOST_NOT_ALLOWED"


def test_unresolvable_host_is_refused(config: Config) -> None:
    error = refusal(config, "https://nowhere.invalid/resume")
    assert error.code == "CALLBACK_HOST_NOT_ALLOWED"
    assert "could not be resolved" in error.message


def test_the_refusal_message_never_echoes_the_url_path(config: Config) -> None:
    """Callback URLs are capability URLs — the path must not come back in an error."""
    error = refusal(config, "http://169.254.169.254/resume/super-secret-token")
    assert "super-secret-token" not in error.message


# ---- allow-list ---------------------------------------------------------


def test_allow_list_admits_only_named_hosts(config: Config) -> None:
    restricted = dataclasses.replace(config, callback_allowed_hosts=("activepieces.internal",))
    assert check(restricted, "http://activepieces.internal/resume")
    error = refusal(restricted, "https://receiver.test/resume")
    assert error.code == "CALLBACK_HOST_NOT_ALLOWED"
    assert "CALLBACK_ALLOWED_HOSTS" in error.message


def test_allow_list_leading_dot_matches_subdomains(config: Config) -> None:
    restricted = dataclasses.replace(config, callback_allowed_hosts=(".example.com",))
    assert check(restricted, "https://flows.example.com/resume")
    assert refusal(restricted, "https://receiver.test/resume")


def test_allow_list_does_not_override_the_address_policy(config: Config) -> None:
    """Naming a host does not license it to point at the metadata endpoint."""
    restricted = dataclasses.replace(config, callback_allowed_hosts=("metadata.evil.test",))
    assert refusal(restricted, "https://metadata.evil.test/x").code == "CALLBACK_HOST_NOT_ALLOWED"


# ---- self-reference -----------------------------------------------------


def test_callback_pointing_back_at_this_service_is_refused(config: Config) -> None:
    hosted = dataclasses.replace(config, host_url="https://receiver.test")
    error = refusal(hosted, "https://receiver.test/jobs/abc")
    assert error.code == "CALLBACK_HOST_NOT_ALLOWED"
    assert "HOST_URL" in error.message


def test_a_different_port_on_the_same_host_is_still_a_valid_receiver(config: Config) -> None:
    hosted = dataclasses.replace(config, host_url="https://receiver.test")
    assert check(hosted, "https://receiver.test:9000/resume")
