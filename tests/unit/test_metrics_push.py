from serving import metrics_push


def test_noop_when_no_url():
    assert metrics_push.start_metrics_push(None, None, None) is False


def test_starts_when_url_present(monkeypatch):
    started = {}

    class FakeThread:
        def __init__(self, *a, **k):
            started["made"] = True

        def start(self):
            started["started"] = True

    monkeypatch.setattr(metrics_push.threading, "Thread", FakeThread)
    ok = metrics_push.start_metrics_push("https://x/api/prom/push", "123", "key")
    assert ok is True
    assert started.get("started") is True
