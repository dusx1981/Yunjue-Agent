# Copyright (c) 2026 Yunjue Tech
# SPDX-License-Identifier: Apache-2.0
"""
Kimi 后端实现（可选）

使用 Moonshot AI Kimi API 生成代码
"""

import asyncio
import logging
import os
import re
from typing import Tuple, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.utils.backends.base import CodeGeneratorBackend
from src.schema.types import LLMType
from src.services.llms.llm import create_llm
from src.tools.dynamic_tool_loader import count_text_tokens
from src.tools.utils import extract_tool_info
from src.utils.venv import ISOLATED_PYTHON_PATH

logger = logging.getLogger(__name__)

# Timeout for Kimi API calls
KIMI_CODE_TIMEOUT_SECONDS = 20 * 60
MAX_GENERATION_RETRIES = 3


class KimiBackend(CodeGeneratorBackend):
    """Kimi API 代码生成后端（可选）"""
    
    @property
    def name(self) -> str:
        return "kimi"
    
    def is_available(self) -> bool:
        """检查 Kimi API 是否可用"""
        api_key = os.environ.get('MOONSHOT_API_KEY', '')
        return bool(api_key and api_key != 'your_moonshot_api_key_here')
    
    async def generate_code(self, prompt: str, output_file: Optional[str] = None) -> Tuple[str, bool]:
        """使用 Kimi API 生成代码"""
        try:
            logger.info(f"Calling Kimi API with prompt length: {len(prompt)}")
            input_tokens = count_text_tokens(prompt)
            
            # 创建 LLM 实例
            llm = create_llm(LLMType.BASIC)
            
            system_prompt = """You are a Python tool generator. Generate complete, working Python tool code.

Requirements:
1. Generate ONLY Python code, no explanations
2. Code must include __TOOL_META__ with name, description, and dependencies
3. Define InputModel using Pydantic BaseModel
4. Implement run(input_model) function
5. Include error handling and type hints

Response format:
```python
# Your complete tool code here
```
"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ]
            
            generated_code = ""
            for attempt in range(1, MAX_GENERATION_RETRIES + 1):
                try:
                    response = await asyncio.wait_for(
                        llm.ainvoke(messages),
                        timeout=KIMI_CODE_TIMEOUT_SECONDS
                    )
                    
                    content = response.content if hasattr(response, 'content') else str(response)
                    if isinstance(content, list):
                        content = "\n".join([str(c) if isinstance(c, str) else str(c.get('text', '')) for c in content])
                    generated_code = str(content)
                    
                    code_blocks = re.findall(r"```python\s*(.*?)\s*```", generated_code, re.DOTALL)
                    if code_blocks:
                        generated_code = code_blocks[-1].strip()
                        logger.info(f"Successfully extracted Python code ({len(generated_code)} chars)")
                        break
                    else:
                        if attempt == MAX_GENERATION_RETRIES:
                            return "", False
                        
                except asyncio.TimeoutError:
                    if attempt == MAX_GENERATION_RETRIES:
                        return "", False
                except Exception:
                    if attempt == MAX_GENERATION_RETRIES:
                        return "", False
            
            # 保存并验证
            if output_file:
                import subprocess
                os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
                
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(generated_code)
                
                extraction_success, tool_info, _ = extract_tool_info(output_file)
                if not extraction_success:
                    os.remove(output_file)
                    return "", False
                
                tool_meta = tool_info.get("tool_meta", {})
                if not tool_meta:
                    os.remove(output_file)
                    return "", False
                
                deps = tool_meta.get("dependencies", [])
                if deps:
                    try:
                        subprocess.run(
                            ["uv", "pip", "install", "--python", str(ISOLATED_PYTHON_PATH)] + deps,
                            check=True, capture_output=True, text=True
                        )
                    except subprocess.CalledProcessError:
                        os.remove(output_file)
                        return "", False
                        
            return generated_code if not output_file else output_file, True

        except Exception as e:
            logger.error(f"Kimi backend error: {e}")
            return "", False
