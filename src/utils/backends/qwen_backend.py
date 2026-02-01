# Copyright (c) 2026 Yunjue Tech
# SPDX-License-Identifier: Apache-2.0
"""
千问(Qwen) 后端实现

使用阿里云 DashScope 千问 API 生成代码
"""

import asyncio
import logging
import os
import re
import subprocess
from typing import Tuple, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.utils.backends.base import CodeGeneratorBackend
from src.schema.types import LLMType
from src.services.llms.llm import create_llm
from src.tools.dynamic_tool_loader import count_text_tokens
from src.tools.utils import extract_tool_info
from src.utils.venv import ISOLATED_PYTHON_PATH

logger = logging.getLogger(__name__)

# Timeout for Qwen API calls (20 minutes to match original)
QWEN_CODE_TIMEOUT_SECONDS = 20 * 60
# Max retries for code generation
MAX_GENERATION_RETRIES = 3


class QwenBackend(CodeGeneratorBackend):
    """千问 API 代码生成后端"""
    
    @property
    def name(self) -> str:
        return "qwen"
    
    def is_available(self) -> bool:
        """检查千问 API 是否可用"""
        try:
            # 检查是否有配置好的 DashScope API Key
            llm = create_llm(LLMType.BASIC)
            # 如果能创建 LLM，说明配置正确
            return True
        except Exception as e:
            logger.debug(f"Qwen backend not available: {e}")
            return False
    
    async def generate_code(self, prompt: str, output_file: Optional[str] = None) -> Tuple[str, bool]:
        """
        使用千问 API 生成代码
        
        Args:
            prompt: 代码生成提示词
            output_file: 可选，输出文件路径
            
        Returns:
            Tuple of (generated_code_or_path, success)
        """
        try:
            logger.info(f"Calling Qwen API with prompt length: {len(prompt)}")
            input_tokens = count_text_tokens(prompt)
            
            # 创建 LLM 实例（使用 BASIC_MODEL，配置为千问）
            llm = create_llm(LLMType.BASIC)
            
            # 构建系统提示词
            system_prompt = """You are a Python tool generator. Your task is to generate complete, working Python tool code based on the user's requirements.

Requirements:
1. Generate ONLY Python code, no explanations
2. Code must include __TOOL_META__ with name, description, and dependencies
3. Define InputModel using Pydantic BaseModel
4. Implement run(input_model) function
5. Include error handling and type hints
6. Use only allowed API keys: TAVILY_API_KEY for web search

Response format:
```python
# Your complete tool code here
```
"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ]
            
            # 调用 LLM，带重试机制
            generated_code = ""
            for attempt in range(1, MAX_GENERATION_RETRIES + 1):
                try:
                    logger.info(f"Qwen generation attempt {attempt}/{MAX_GENERATION_RETRIES}")
                    
                    # 调用 LLM，设置超时
                    response = await asyncio.wait_for(
                        llm.ainvoke(messages),
                        timeout=QWEN_CODE_TIMEOUT_SECONDS
                    )
                    
                    # 确保 content 是字符串类型
                    content = response.content if hasattr(response, 'content') else str(response)
                    if isinstance(content, list):
                        # 如果是列表（多模态消息），转换为字符串
                        content = "\n".join([str(c) if isinstance(c, str) else str(c.get('text', '')) for c in content])
                    generated_code = str(content)
                    output_tokens = count_text_tokens(generated_code)
                    logger.info(f"Qwen API input tokens: {input_tokens}, output tokens: {output_tokens}")
                    
                    # 从 markdown 中提取代码
                    code_blocks = re.findall(r"```python\s*(.*?)\s*```", generated_code, re.DOTALL)
                    if code_blocks:
                        generated_code = code_blocks[-1].strip()
                        logger.info(f"Successfully extracted Python code ({len(generated_code)} chars)")
                        break
                    else:
                        logger.warning(f"Attempt {attempt}: No Python code block found, retrying...")
                        if attempt == MAX_GENERATION_RETRIES:
                            return "", False
                        
                except asyncio.TimeoutError:
                    logger.error(f"Attempt {attempt}: Qwen API timed out after {QWEN_CODE_TIMEOUT_SECONDS} seconds")
                    if attempt == MAX_GENERATION_RETRIES:
                        return "", False
                except Exception as e:
                    logger.error(f"Attempt {attempt}: Error calling Qwen API: {type(e).__name__}: {str(e)}")
                    if attempt == MAX_GENERATION_RETRIES:
                        return "", False
            
            # 如果指定了 output_file，保存并验证
            if output_file:
                try:
                    # 确保目录存在
                    output_dir = os.path.dirname(output_file)
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                    
                    # 写入代码到文件
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(generated_code)
                    logger.info(f"Generated code saved to: {output_file}")
                    
                    # 提取并验证工具信息
                    extraction_success, tool_info, extraction_error = extract_tool_info(output_file)
                    if not extraction_success:
                        logger.error(f"Failed to extract tool info for {output_file}: {extraction_error}")
                        os.remove(output_file)
                        return "", False
                    
                    tool_meta = tool_info.get("tool_meta", {})
                    if not tool_meta:
                        logger.warning(f"Could not extract __TOOL_META__ from {output_file}")
                        os.remove(output_file)
                        return "", False
                    
                    # 安装依赖
                    deps = tool_meta.get("dependencies", [])
                    if deps:
                        try:
                            logger.info(f"Installing dependencies for tool {output_file}: {deps}")
                            subprocess.run(
                                ["uv", "pip", "install", "--python", str(ISOLATED_PYTHON_PATH)] + deps,
                                check=True,
                                capture_output=True,
                                text=True
                            )
                            logger.info(f"Successfully installed dependencies: {deps}")
                        except subprocess.CalledProcessError as e:
                            error_message = f"Failed to install dependencies for tool {output_file}: {e}"
                            logger.error(error_message)
                            logger.error(f"stdout: {e.stdout}")
                            logger.error(f"stderr: {e.stderr}")
                            os.remove(output_file)
                            return error_message, False
                            
                except Exception as e:
                    logger.error(f"Failed to save code to {output_file}: {e}")
                    return "", False

            return generated_code if not output_file else output_file, True

        except Exception as e:
            logger.error(f"Error calling Qwen API: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return "", False
