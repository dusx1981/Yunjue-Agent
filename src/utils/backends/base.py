# Copyright (c) 2026 Yunjue Tech
# SPDX-License-Identifier: Apache-2.0
"""
代码生成后端基类
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional


class CodeGeneratorBackend(ABC):
    """代码生成器后端抽象基类"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """后端名称标识"""
        pass
    
    @abstractmethod
    async def generate_code(self, prompt: str, output_file: Optional[str] = None) -> Tuple[str, bool]:
        """生成代码并返回结果"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查后端是否可用"""
        pass
