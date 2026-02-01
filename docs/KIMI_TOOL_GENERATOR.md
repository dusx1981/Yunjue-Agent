# Kimi Tool Generator 使用说明

## 📋 简介

`kimi_tool_generator` 是一个使用 **Moonshot AI (kimi 2.5)** 直接生成 Python 工具代码的模块，作为 **Codex CLI** 的纯国产替代方案。

### 优势
- ✅ **无需额外工具**：直接使用已配置的 kimi 2.5 模型
- ✅ **网络友好**：国内 API 节点，访问稳定
- ✅ **成本可控**：kimi 2.5 目前免费或低成本
- ✅ **无缝替换**：API 接口与原 `call_codex_exec` 兼容

---

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `src/utils/kimi_tool_generator.py` | 核心模块，提供工具生成功能 |
| `test_kimi_tool_generator.py` | 测试脚本，验证模块功能 |
| `docs/KIMI_TOOL_GENERATOR.md` | 本文档 |

---

## 🚀 快速开始

### 1. 基本使用

```python
import asyncio
from src.utils.kimi_tool_generator import call_kimi_exec

async def main():
    # 定义提示词（工具生成要求）
    prompt = """
创建一个 Python 工具，用于计算两个日期之间的天数。

要求：
1. 工具名称: date_diff_calculator
2. 输入：两个日期字符串（格式: YYYY-MM-DD）
3. 输出：天数差（整数）
4. 依赖: 只允许使用标准库 datetime
5. 包含错误处理
"""
    
    # 生成工具代码
    code, success = await call_kimi_exec(prompt, output_file="my_tool.py")
    
    if success:
        print(f"✅ 工具生成成功: {code}")
    else:
        print("❌ 生成失败")

asyncio.run(main())
```

### 2. 在 Yunjue Agent 中使用

#### 方法 A：替换单个调用（推荐用于测试）

在需要替换的文件中（如 `src/core/nodes.py`）：

```python
# 原代码：
from src.utils.utils import call_codex_exec

# 替换为：
from src.utils.kimi_tool_generator import call_kimi_exec as call_codex_exec
```

#### 方法 B：全局替换（推荐用于生产）

修改 `src/utils/utils.py`，在文件末尾添加：

```python
# 根据环境变量选择工具生成器
import os

if os.getenv("USE_KIMI_TOOL_GENERATOR", "0") == "1":
    from src.utils.kimi_tool_generator import call_kimi_exec as call_codex_exec
```

然后在 `.env` 中设置：

```bash
USE_KIMI_TOOL_GENERATOR=1
```

---

## 🔧 高级功能

### 1. 带历史记录的工具增强

用于工具失败后的自动修复：

```python
from src.utils.kimi_tool_generator import call_kimi_exec_with_history

# 历史失败记录
historical_attempts = [
    {
        "code": "def broken():\n    return error",
        "error": "NameError: name 'error' is not defined"
    }
]

# 使用历史记录生成修复版本
result, success = await call_kimi_exec_with_history(
    prompt="修复上述代码",
    historical_attempts=historical_attempts,
    output_file="fixed_tool.py"
)
```

### 2. 仅生成代码（不保存文件）

```python
# 不提供 output_file 参数
code, success = await call_kimi_exec(prompt)

if success:
    print(f"生成的代码:\n{code}")
```

---

## ⚙️ 配置说明

### 环境变量

确保 `.env` 文件中已配置：

```bash
# 必需的配置
MOONSHOT_API_KEY=sk-your-moonshot-api-key

# 可选配置
USE_KIMI_TOOL_GENERATOR=1  # 启用 kimi 工具生成器
MAX_GENERATION_RETRIES=3   # 生成重试次数（默认: 3）
KIMI_TOOL_TIMEOUT=600      # 超时时间（秒，默认: 600）
```

### conf.yaml

确保 BASIC_MODEL 配置为 kimi：

```yaml
BASIC_MODEL:
  base_url: https://api.moonshot.cn/v1
  model: "kimi-k2.5"
  api_key: ${MOONSHOT_API_KEY}
  temperature: 0.7
  token_limit: 128000
```

---

## 🧪 测试

运行测试脚本：

```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行测试
python test_kimi_tool_generator.py
```

预期输出：
```
======================================================================
Kimi Tool Generator 测试脚本
======================================================================
✓ API Key 已配置: sk-0njOl3oaMSuWgJ...
======================================================================
测试 1: 基本工具生成
======================================================================
生成工具: test_output/date_diff_tool.py
提示词长度: 256
✅ 成功生成工具: test_output/date_diff_tool.py
...
🎉 所有测试通过！kimi_tool_generator 工作正常
```

---

## 📝 API 参考

### `call_kimi_exec(prompt, output_file=None)`

使用 kimi 2.5 生成代码。

**参数：**
- `prompt` (str): 工具生成提示词
- `output_file` (str, optional): 保存路径，如果提供则保存并验证

**返回：**
- `Tuple[str, bool]`: (生成结果, 是否成功)
  - 如果 `output_file` 提供，返回 (文件路径, 成功状态)
  - 如果未提供，返回 (代码字符串, 成功状态)

**示例：**
```python
code, success = await call_kimi_exec("创建一个计算器工具", "calc.py")
```

### `call_kimi_exec_with_history(prompt, historical_attempts=None, output_file=None)`

使用历史失败记录改进生成。

**参数：**
- `prompt` (str): 基础提示词
- `historical_attempts` (List[dict]): 历史尝试记录
- `output_file` (str, optional): 保存路径

**返回：**
- `Tuple[str, bool]`: (生成结果, 是否成功)

**示例：**
```python
history = [{"code": "def bad():", "error": "SyntaxError"}]
result, success = await call_kimi_exec_with_history(
    "修复代码", history, "fixed.py"
)
```

---

## 🔍 故障排查

### 问题 1：API key 错误

**症状：**
```
Error: Incorrect API key provided
```

**解决：**
1. 检查 `.env` 中的 `MOONSHOT_API_KEY`
2. 确认 key 格式为 `sk-` 开头
3. 从 https://platform.moonshot.cn/ 重新生成

### 问题 2：生成的代码无法运行

**症状：**
工具生成成功但执行失败

**解决：**
1. 检查 `__TOOL_META__` 是否正确
2. 确认依赖项已安装
3. 使用 `call_kimi_exec_with_history` 进行增强修复

### 问题 3：超时

**症状：**
```
TimeoutError: Kimi exec timed out
```

**解决：**
增加超时时间：
```python
import os
os.environ["KIMI_TOOL_TIMEOUT"] = "1200"  # 20分钟
```

---

## 🆚 与 Codex CLI 对比

| 特性 | Codex CLI | Kimi Tool Generator |
|------|-----------|---------------------|
| 依赖 | 需要安装 CLI 工具 | 仅依赖 Python 库 |
| API | OpenAI / 自定义 | Moonshot (kimi) |
| 网络 | 可能需要代理 | 国内直连 |
| 成本 | OpenAI 付费 | kimi 免费/低成本 |
| 沙盒 | 支持多种沙盒 | 依赖项目沙盒 |
| 功能 | 完整 IDE 集成 | 专注工具生成 |

---

## 📞 支持

如有问题：
1. 查看项目文档：`docs/code/architecture.md`
2. 运行测试脚本：`python test_kimi_tool_generator.py`
3. 检查日志输出

---

**最后更新：2026-02-01**
