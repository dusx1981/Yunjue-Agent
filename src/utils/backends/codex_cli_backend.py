# Copyright (c) 2026 Yunjue Tech
# SPDX-License-Identifier: Apache-2.0
"""
Codex CLI 后端实现

使用 OpenAI Codex CLI 工具生成代码
"""

import asyncio
import logging
import os
import re
import subprocess
from typing import Tuple, Optional

from src.utils.backends.base import CodeGeneratorBackend
from src.tools.dynamic_tool_loader import count_text_tokens
from src.tools.utils import extract_tool_info
from src.utils.venv import ISOLATED_PYTHON_PATH

logger = logging.getLogger(__name__)

# Timeout for Codex exec calls (20 minutes)
CODEX_EXEC_TIMEOUT_SECONDS = 20 * 60


class CodexCLIBackend(CodeGeneratorBackend):
    """Codex CLI 代码生成后端"""
    
    @property
    def name(self) -> str:
        return "codex_cli"
    
    def is_available(self) -> bool:
        """检查 Codex CLI 是否可用"""
        try:
            subprocess.run(
                ["codex", "--version"],
                capture_output=True,
                check=True,
                timeout=5
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    async def generate_code(self, prompt: str, output_file: Optional[str] = None) -> Tuple[str, bool]:
        """
        使用 Codex CLI 生成代码
        
        Args:
            prompt: 代码生成提示词
            output_file: 可选，输出文件路径
            
        Returns:
            Tuple of (generated_code_or_path, success)
        """
        try:
            # Build codex exec command
            codex_profile = os.environ.get("CODEX_PROFILE", None)
            command = ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox"]
            if codex_profile:
                command += ["--profile", codex_profile]

            logger.info(f"Calling codex exec with prompt length: {len(prompt)}")
            input_tokens = count_text_tokens(prompt)
            
            # Create async subprocess for true concurrent execution
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Send prompt to stdin and wait for completion
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=prompt.encode("utf-8")),
                    timeout=CODEX_EXEC_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.error("Codex exec timed out after %s seconds", CODEX_EXEC_TIMEOUT_SECONDS)
                process.kill()
                await process.communicate()
                return "", False
            
            output_tokens = count_text_tokens(stdout.decode("utf-8") + stderr.decode("utf-8"))
            logger.info(f"Codex exec input tokens: {input_tokens}, output tokens: {output_tokens}")
            
            # Decode output
            generated_code = stdout.decode("utf-8").strip() if stdout else ""
            error_output = stderr.decode("utf-8").strip() if stderr else ""

            logger.info(f"Codex exec stdout: {generated_code}")
            if error_output:
                logger.debug(f"Codex exec stderr: {error_output}")

            if process.returncode != 0:
                error_msg = f"Codex exec failed with return code {process.returncode}"
                logger.error(error_msg)
                return "", False

            if not generated_code:
                logger.warning("Codex exec returned empty output")

            # If output_file is specified, save the code to file
            if output_file:
                try:
                    # extract the last ```python ... ``` code block
                    code_blocks = re.findall(r"```python\s*(.*?)\s*```", generated_code, re.DOTALL)
                    if not code_blocks:
                        logger.warning("No python code block found in Codex exec output")
                        return "", False

                    generated_code = code_blocks[-1].strip()
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(generated_code)
                    logger.info(f"Generated code saved to: {output_file}")
                    
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
                    
                    # Install dependencies for the tool if specified
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
                            os.remove(output_file)
                            return error_message, False
                            
                except Exception as e:
                    logger.error(f"Failed to save code to {output_file}: {e}")
                    return "", False

            return generated_code if not output_file else output_file, True

        except FileNotFoundError:
            logger.error("Codex exec command not found. Please ensure 'codex' is installed and in PATH")
            return "", False
        except Exception as e:
            logger.error(f"Error calling codex exec: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return "", False
