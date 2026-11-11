import pytest

from src.deploy.auth import LoginLimiter, check_password, issue_token, verify_token

KEY = b"k" * 32


def test_token_roundtrip_and_expiry():
    token, expires_at = issue_token(KEY, ttl_s=60, now=1000.0)
    assert expires_at == 1060
    assert verify_token(KEY, token, now=1059.0)
    assert not verify_token(KEY, token, now=1060.0)


@pytest.mark.parametrize("bad", ["", "abc", "a.b", "a.b.c", "....", "é.é"])
def test_malformed_tokens_are_rejected(bad):
    assert not verify_token(KEY, bad, now=0.0)


def test_tampered_or_foreign_token_is_rejected():
    token, _ = issue_token(KEY, ttl_s=60, now=0.0)
    payload, sig = token.split(".")
    forged_payload, _ = issue_token(KEY, ttl_s=10**9, now=0.0)
    assert not verify_token(KEY, forged_payload.split(".")[0] + "." + sig, now=1.0)
    assert not verify_token(b"other" * 8, token, now=1.0)
    assert not verify_token(KEY, payload + "." + sig[:-2] + "AA", now=1.0)


def test_check_password():
    assert check_password("secret", "secret")
    assert not check_password("secreT", "secret")
    assert not check_password("", "secret")
    assert not check_password("anything", "")  # unset password never matches


def test_limiter_per_client_and_window():
    lim = LoginLimiter(max_failures=2, window_s=10, max_global_failures=100)
    assert not lim.blocked("a", now=0)
    lim.record_failure("a", now=0)
    lim.record_failure("a", now=1)
    assert lim.blocked("a", now=2)
    assert not lim.blocked("b", now=2)
    assert not lim.blocked("a", now=11.5)  # both failures left the window


def test_limiter_global_cap():
    lim = LoginLimiter(max_failures=100, window_s=10, max_global_failures=3)
    for i in range(3):
        lim.record_failure(f"ip{i}", now=0)
    assert lim.blocked("fresh", now=1)
