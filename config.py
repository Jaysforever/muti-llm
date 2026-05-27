"""
config.py — 环境配置、多模型注册中心、带重试机制的 LLM 调用函数。

每个模型独立配置 base_url + api_key + model_id，天然支持多厂商混合调用。
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 修复 conda 环境中 SSL_CERT_FILE 指向不存在路径的问题
_ssl_cert = os.environ.get("SSL_CERT_FILE", "")
if _ssl_cert and not os.path.isfile(_ssl_cert):
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# 1. 加载 .env 文件（项目根目录）
# ---------------------------------------------------------------------------
load_dotenv(Path(__file__).parent / ".env")

# ---------------------------------------------------------------------------
# 2. 模型注册中心 — 每个模型有独立的 endpoint / key / model_id
#    格式: { 简称: (model_id, base_url, api_key) }
# ---------------------------------------------------------------------------
_MODEL_REGISTRY: dict[str, tuple[str, str, str]] = {
    "deepseek": (
        os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        os.getenv("DEEPSEEK_API_KEY", ""),
    ),
    "minimax": (
        os.getenv("MINIMAX_MODEL", "MiniMax-M2.7"),
        os.getenv("MINIMAX_BASE_URL", "https://api.minimax.chat/v1"),
        os.getenv("MINIMAX_API_KEY", ""),
    ),
    "mimo": (
        os.getenv("MIMO_MODEL", "mimo-v2.5-pro"),
        os.getenv("MIMO_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1"),
        os.getenv("MIMO_API_KEY", ""),
    ),
    # 论文原始模型（按需配置）
    "mistral": (
        os.getenv("MISTRAL_MODEL", "mistralai/Mistral-7B-Instruct-v0.1"),
        os.getenv("MISTRAL_BASE_URL", "https://api.openai.com/v1"),
        os.getenv("MISTRAL_API_KEY", os.getenv("OPENAI_API_KEY", "")),
    ),
    "chatgpt": (
        os.getenv("CHATGPT_MODEL", "gpt-3.5-turbo"),
        os.getenv("CHATGPT_BASE_URL", "https://api.openai.com/v1"),
        os.getenv("CHATGPT_API_KEY", os.getenv("OPENAI_API_KEY", "")),
    ),
}

# 默认使用的模型简称
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "mimo")

# 默认推理参数
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 512

# ---------------------------------------------------------------------------
# 3. LLM 客户端缓存 — 按 (base_url, api_key) 去重
# ---------------------------------------------------------------------------
_clients: dict[tuple[str, str], OpenAI] = {}


def _get_client(model_name: str) -> tuple[OpenAI, str]:
    """根据模型简称获取 (客户端实例, 实际model_id)。"""
    if model_name not in _MODEL_REGISTRY:
        # 可能是直接传了 model_id（非简称），用默认 key/url
        key = (os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
               os.getenv("OPENAI_API_KEY", ""))
        if key not in _clients:
            _clients[key] = OpenAI(api_key=key[1], base_url=key[0])
        return _clients[key], model_name

    model_id, base_url, api_key = _MODEL_REGISTRY[model_name]
    cache_key = (base_url, api_key)
    if cache_key not in _clients:
        _clients[cache_key] = OpenAI(api_key=api_key, base_url=base_url)
    return _clients[cache_key], model_id


# ---------------------------------------------------------------------------
# 4. 带重试机制的 LLM 调用函数
# ---------------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
    reraise=True,
)
def call_llm(
    prompt: str,
    model_name: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    system_prompt: str | None = None,
) -> str:
    """
    调用 LLM，返回生成的文本内容。

    Parameters
    ----------
    prompt : str
        用户 prompt。
    model_name : str, optional
        模型简称（"deepseek", "minimax", "chatgpt" 等）或直接 model_id。
        默认 DEFAULT_MODEL。
    temperature : float
        采样温度，默认 0.1。
    max_tokens : int
        最大生成 token 数。
    system_prompt : str, optional
        系统消息。

    Returns
    -------
    str — LLM 返回的文本内容。
    """
    if model_name is None:
        model_name = DEFAULT_MODEL

    client, model_id = _get_client(model_name)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    logger.info(f"→ Calling [{model_id}] t={temperature} | prompt_len={len(prompt)}")

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or ""
        logger.info(f"← [{model_id}] response_len={len(content)}")
        return content

    except Exception as e:
        logger.error(f"API call failed for [{model_id}]: {e}")
        raise


def call_llm_raw(
    prompt: str,
    model_name: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    system_prompt: str | None = None,
) -> tuple[str, object]:
    """
    调用 LLM，返回 (文本内容, 完整 response 对象)。
    用于需要 token 概率等额外信息的场景（如 Token Probability baseline）。
    """
    if model_name is None:
        model_name = DEFAULT_MODEL

    client, model_id = _get_client(model_name)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    logger.info(f"→ Calling [{model_id}] t={temperature} | prompt_len={len(prompt)}")

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or ""
        logger.info(f"← [{model_id}] response_len={len(content)}")
        return content, response

    except Exception as e:
        logger.error(f"API call failed for [{model_id}]: {e}")
        raise


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
    reraise=True,
)
def call_llm_with_logprobs(
    prompt: str,
    model_name: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = 64,
    system_prompt: str | None = None,
    top_logprobs: int = 20,
) -> tuple[str, dict | None]:
    """
    调用 LLM 并请求 token-level logprobs。

    Returns
    -------
    (content, logprobs_dict)
        logprobs_dict 包含:
        - "tokens": list[str]
        - "token_logprobs": list[float]
        - "top_logprobs": list[dict[str, float]]
        如果 API 不支持则返回 (content, None)。
    """
    if model_name is None:
        model_name = DEFAULT_MODEL

    client, model_id = _get_client(model_name)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    logger.info(
        f"→ Calling [{model_id}] t={temperature} logprobs={top_logprobs} | prompt_len={len(prompt)}"
    )

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=top_logprobs,
        )
        content = response.choices[0].message.content or ""
        logprobs = None
        if hasattr(response.choices[0], "logprobs") and response.choices[0].logprobs:
            lp = response.choices[0].logprobs
            logprobs = {
                "tokens": [t.token for t in lp.content],
                "token_logprobs": [t.logprob for t in lp.content],
                "top_logprobs": [
                    {top.token: top.logprob for top in t.top_logprobs} if t.top_logprobs else {}
                    for t in lp.content
                ],
            }
        logger.info(f"← [{model_id}] response_len={len(content)} logprobs_ok={logprobs is not None}")
        return content, logprobs

    except Exception as e:
        logger.error(f"API call failed for [{model_id}]: {e}")
        raise


def list_models() -> list[str]:
    """返回当前已注册的模型简称列表。"""
    return list(_MODEL_REGISTRY.keys())


def get_other_models(exclude: str | None = None) -> list[str]:
    """返回除 exclude 之外的所有已注册模型简称。可用于 COOPERATE-Others 等场景。"""
    return [m for m in _MODEL_REGISTRY if m != exclude]
