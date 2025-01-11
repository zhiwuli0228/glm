from __future__ import annotations

import logging
from typing import Dict, List, Any
from langchain.embeddings.base import Embeddings
from pydantic import BaseModel, model_validator

logger = logging.getLogger(__name__)


class ZhipuAIEmbeddings(BaseModel, Embeddings):
    """`Zhipuai Embeddings` embudding models."""

    client: Any
    """`zhipuai.ZhipuAI`"""

    @model_validator(mode="before")
    def validate_envirioment(cls, values: Dict) -> Dict:
        """
        实例化ZhipuAI为values["client"]

        args:
            value: (Dict): 包含信息的字典，必须包含client的字段
        Returns：
            values(Dict): 包含配置信息的字典，如果环境中有zhipuai的库，这将返回实例化的ZhipiAI，否则将报错'ModuleNotFoundError: No module named 'zhipuai''

        """
        from zhipuai import ZhipuAI
        values["client"] = ZhipuAI(
        )
        return values

    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding.

        Args:
            text (str): 要生成 embedding 的文本.

        Return:
            embeddings (List[float]): 输入文本的 embedding，一个浮点数值列表.
        """
        embeddings = self.client.embeddings.create(
            model="embedding-3",  # 填写需要调用的模型编码
            input=text,
        )
        return embeddings.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成输入文本的列表的 embedding

        Args:
            texts (List[str]): 要生存embedding 的文本列表

        Returns:
            List[List[float]]: 输入列表中每个文档的 embedding 列表。每个 embedding 都表示为一个浮点值列表。
        """
        return [self.embed_query(text) for text in texts]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        raise NotImplementedError("Please use `embed_documents`. Official does not support asynchronous requests")

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        raise NotImplementedError("Please use `aembed_query`. Official does not support asynchronous requests")
