# 代码生成桥接层架构

## 概述

代码生成桥接层提供了一个统一的接口，用于对接不同的代码生成模型（后端）。
通过桥接模式，可以轻松切换不同的 AI 模型来生成代码，而无需修改业务逻辑。

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                     业务层 (evolve.sh)                       │
├─────────────────────────────────────────────────────────────┤
│              桥接层 (CodeGeneratorBridge)                    │
│                    - 统一接口                                │
│                    - 后端管理                                │
│                    - 自动选择                                │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Codex CLI    │  │ 千问 API     │  │ Kimi API     │      │
│  │ 后端         │  │ 后端         │  │ 后端 (可选)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. 桥接层接口
- **文件**: `src/utils/code_generator_bridge.py`
- **类**: `CodeGeneratorBridge`
- **职责**: 
  - 统一管理所有后端
  - 根据配置自动选择合适的后端
  - 提供统一的代码生成接口

### 2. 后端实现
所有后端都继承自 `CodeGeneratorBackend` 抽象基类：

#### Codex CLI 后端
- **文件**: `src/utils/backends/codex_cli_backend.py`
- **名称**: `codex_cli`
- **依赖**: 需要安装 `codex` CLI 工具
- **配置**: 需要 `OPENAI_API_KEY` 环境变量

#### 千问(Qwen) 后端
- **文件**: `src/utils/backends/qwen_backend.py`
- **名称**: `qwen`
- **依赖**: 需要配置 `conf.yaml` 中的 DashScope API
- **配置**: 需要有效的 DashScope API Key

#### Kimi 后端（可选）
- **文件**: `src/utils/backends/kimi_backend.py`
- **名称**: `kimi`
- **依赖**: 需要 `MOONSHOT_API_KEY` 环境变量
- **状态**: 可选实现

## 使用方法

### 方式 1: 命令行脚本（推荐）

```bash
# 使用千问后端运行
./run_with_bridge.sh qwen DEEPSEARCHQA my_run 1 0

# 使用 Codex CLI 运行（默认）
./run_with_bridge.sh codex_cli DEEPSEARCHQA my_run 1 0

# 查看帮助
./run_with_bridge.sh help
```

### 方式 2: 环境变量

```bash
# 设置后端
export CODE_GEN_BACKEND=qwen  # 或 codex_cli, kimi

# 运行项目
./scripts/evolve.sh --dataset DEEPSEARCHQA --run_name test --batch_size 1 --start 0
```

### 方式 3: 在代码中使用

```python
from src.utils.code_generator_bridge import CodeGeneratorBridge

# 获取桥接层实例
bridge = CodeGeneratorBridge.get_instance()

# 查看当前后端
print(f"当前后端: {bridge.backend_name}")

# 生成代码
result, success = await bridge.generate_code(prompt, output_file="tool.py")

# 切换后端（如果需要）
import os
os.environ['CODE_GEN_BACKEND'] = 'qwen'
bridge = CodeGeneratorBridge.get_instance(force_reselect=True)
```

## 配置优先级

桥接层按以下优先级选择后端：

1. **环境变量**: `CODE_GEN_BACKEND=qwen|codex_cli|kimi`
2. **旧环境变量**: `USE_QWEN_CODE_GEN=1`（向后兼容）
3. **配置文件**: `conf.yaml` 中的 `code_generator.backend`
4. **默认值**: `codex_cli`

## 后端可用性检查

每个后端都有 `is_available()` 方法，用于检查是否满足运行条件：

```python
from src.utils.backends.qwen_backend import QwenBackend

backend = QwenBackend()
if backend.is_available():
    print("千问后端可用")
else:
    print("千问后端不可用，请检查配置")
```

## 扩展新后端

要添加新的代码生成后端，只需：

1. 创建新文件 `src/utils/backends/new_backend.py`
2. 继承 `CodeGeneratorBackend` 基类
3. 实现 `name`, `is_available()`, `generate_code()` 方法
4. 在 `code_generator_bridge.py` 的 `_register_backends()` 中注册

示例：

```python
from src.utils.code_generator_bridge import CodeGeneratorBackend

class NewBackend(CodeGeneratorBackend):
    @property
    def name(self) -> str:
        return "new_backend"
    
    def is_available(self) -> bool:
        # 检查必要的配置
        return True
    
    async def generate_code(self, prompt: str, output_file: str = None):
        # 实现代码生成逻辑
        return generated_code, success
```

## 文件结构

```
src/utils/
├── code_generator_bridge.py      # 桥接层核心
├── utils.py                      # 原始工具函数（未修改）
├── qwen_code_generator.py        # 旧的千问实现（可选保留）
├── kimi_tool_generator.py        # Kimi 工具生成器
└── backends/
    ├── __init__.py
    ├── codex_cli_backend.py     # Codex CLI 后端
    ├── qwen_backend.py          # 千问后端
    └── kimi_backend.py          # Kimi 后端
```

## 优势

1. **零侵入**: 原始 `src/utils/utils.py` 完全未修改
2. **可扩展**: 轻松添加新后端，不影响现有代码
3. **配置驱动**: 通过环境变量或配置切换后端
4. **向后兼容**: 支持旧的环境变量 `USE_QWEN_CODE_GEN`
5. **自动降级**: 如果首选后端不可用，自动尝试其他后端

## 注意事项

1. 桥接层使用单例模式，首次创建后不会自动重新选择后端
2. 如需强制切换后端，使用 `CodeGeneratorBridge.get_instance(force_reselect=True)`
3. 或在切换前调用 `CodeGeneratorBridge.reset_instance()`

## 测试

运行测试脚本验证桥接层：

```bash
source .venv/bin/activate
python test_bridge.py
```
