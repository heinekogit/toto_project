import time
import requests


RETRY_STATUSES = {429, 500, 502, 503, 504}


def get_with_retry(
    url,
    *,
    params=None,
    headers=None,
    timeout=(5, 20),
    max_retries=3,
    backoff_base=1.0,
):
    """requests.get with retry for transient network/HTTP failures."""
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            if response.status_code in RETRY_STATUSES:
                last_error = requests.exceptions.HTTPError(
                    f"HTTP {response.status_code} for {response.url}",
                    response=response,
                )
                if attempt < max_retries:
                    wait_sec = backoff_base * (2 ** (attempt - 1))
                    print(
                        f"[WARN][HTTP] transient status={response.status_code} "
                        f"attempt={attempt}/{max_retries} wait={wait_sec:.1f}s url={url}"
                    )
                    time.sleep(wait_sec)
                    continue
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            last_error = e
            response = getattr(e, "response", None)
            status_code = response.status_code if response is not None else None
            retryable = isinstance(
                e,
                (
                    requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError,
                ),
            ) or status_code in RETRY_STATUSES
            if retryable and attempt < max_retries:
                wait_sec = backoff_base * (2 ** (attempt - 1))
                print(
                    f"[WARN][HTTP] retry attempt={attempt}/{max_retries} "
                    f"wait={wait_sec:.1f}s url={url} error={repr(e)}"
                )
                time.sleep(wait_sec)
                continue
            break
    raise last_error
