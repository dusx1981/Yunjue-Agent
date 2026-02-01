# Copyright (c) 2026 Yunjue Tech
# SPDX-License-Identifier: Apache-2.0
"""
代码生成器桥接层 - 简化版

通过环境变量 CODE_GEN_BACKEND 切换后端：
- codex_cli: 原始 Codex CLI (默认)
- qwen: 阿里云千问 API
"""

import os
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# 全局缓存
_backend_instance = None
_backend_name = None

def _get_backend():
    """获取并缓存后端实例"""
    global _backend_instance, _backend_name
    
    if _backend_instance is not None:
        return _backend_instance
    
    # 确定后端类型
    backend_name = os.environ.get('CODE_GEN_BACKEND', '').lower()
    if not backend_name and os.environ.get('USE_QWEN_CODE_GEN') == '1':
        backend_name = 'qwen'
    if not backend_name:
        backend_name = 'codex_cli'
    
    # 导入并实例化后端
    if backend_name == 'qwen':
        from src.utils.backends.qwen_backend import QwenBackend
        _backend_instance = QwenBackend()
    else:
        from src.utils.backends.codex_cli_backend import CodexCLIBackend
        _backend_instance = CodexCLIBackend()
    
    _backend_name = backend_name
    logger.info(f"Using code generator backend: {backend_name}")
    
    return _backend_instance

async def generate_code(prompt: str, output_file: Optional[str] = None) -> Tuple[str, bool]:
    """生成代码的主入口"""
    backend = _get_backend()
    if not backend.is_available():
        logger.error(f"Backend '{_backend_name}' is not available")
        return "", False
    return await backend.generate_code(prompt, output_file)

# 保持与原有接口兼容
call_codex_exec = generate_code
