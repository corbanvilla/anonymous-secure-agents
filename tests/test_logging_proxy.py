import base64

from src.api_clients.logging import OpenAILoggingProxy


class FakeCreate:
    def __call__(self, *args, **kwargs):
        return "ok"


class FakeCompletions:
    def __init__(self):
        self.create = FakeCreate()


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeClient:
    def __init__(self):
        self.chat = FakeChat()


def test_truncate_base64_image_url():
    b64 = "data:image/png;base64," + base64.b64encode(b"0" * 1400).decode()
    messages = [
        {"role": "system", "content": "hi"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": b64}},
            ],
        },
    ]
    log = []
    client = OpenAILoggingProxy(FakeClient(), log)
    client.chat.completions.create(model="gpt", messages=messages)
    logged = log[0]["messages"][1]["content"][1]
    assert logged == {"type": "text", "text": "[BASE64_DATA_TRUNCATED]"}


def test_truncate_plain_base64():
    b64 = base64.b64encode(b"0" * 1400).decode()
    messages = [{"role": "user", "content": b64}]
    log = []
    client = OpenAILoggingProxy(FakeClient(), log)
    client.chat.completions.create(model="gpt", messages=messages)
    logged = log[0]["messages"][0]["content"]
    assert logged == "[BASE64_DATA_TRUNCATED]"
