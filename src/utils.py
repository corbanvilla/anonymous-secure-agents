from openai import OpenAI


def get_default_openai_client():
    return OpenAI(
        api_key="sk-PCG3chp7epg_zuz5rng",
        base_url="http://localhost:4011/v1",
        max_retries=25,
    )
