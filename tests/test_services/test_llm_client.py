import pytest
from unittest.mock import AsyncMock, patch
from src.services.llm_client import LLMClient


@pytest.mark.asyncio
async def test_llm_client_initialization():
    """验证 LLMClient 初始化"""
    client = LLMClient(provider="deepseek", model="deepseek-v4-flash")
    assert client.provider == "deepseek"
    assert client.model == "deepseek-v4-flash"


@pytest.mark.asyncio
async def test_llm_client_generate_with_mock():
    """验证 LLM 生成调用（mock api）"""
    client = LLMClient(provider="deepseek", model="deepseek-v4-flash")
    with patch.object(client, "_call_deepseek", new=AsyncMock(return_value="Mocked response")):
        result = await client.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="Hello!",
            max_tokens=100,
        )
        assert result == "Mocked response"


@pytest.mark.asyncio
async def test_llm_client_generate_unsupported_provider():
    """验证不支持的 provider 抛出错误"""
    client = LLMClient(provider="unsupported", model="test")
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        await client.generate(system_prompt="", user_prompt="test")


@pytest.mark.asyncio
async def test_generate_with_temperature():
    """验证 temperature 参数传递"""
    client = LLMClient(provider="deepseek", model="deepseek-v4-flash")
    with patch.object(client, "_call_deepseek", new=AsyncMock(return_value="response")):
        result = await client.generate(
            system_prompt="", user_prompt="test", temperature=0.7
        )
        assert result == "response"
