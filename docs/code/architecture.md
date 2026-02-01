# Yunjue Agent 架构文档

## 1. 项目概述

Yunjue Agent 是一个**自进化智能体系统**，采用 *In-Situ Self-Evolving* 范式。系统能够在开放式环境中通过连续的任务交互实现实时适应和能力扩展，无需额外的监督信号。

### 1.1 核心特性

- **自进化范式**: 将离散的交互重构为连续的经验流，通过内部反馈循环将短期推理转化为长期能力
- **工具优先进化**: 优先通过工具进化而非记忆或工作流来扩展能力，利用代码执行的二元反馈（成功/失败）作为可靠的内部监督信号
- **并行批处理进化**: 引入并行批处理进化策略优化进化效率
- **零启动能力**: 从空工具库开始，通过推理时生成、验证和归纳实现 SOTA 性能

### 1.2 技术栈

- **Python 3.12+**
- **LangGraph**: 工作流编排
- **LangChain**: LLM 交互和工具管理
- **Codex**: 代码生成（工具创建和增强）
- **Pydantic**: 数据验证和序列化

## 2. 整体架构

### 2.1 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户查询层                               │
│                    (User Query Interface)                        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                        主控制流图                                │
│                      (Main Workflow)                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │  Manager │───▶│  Tool    │───▶│ Executor │───▶│Integrator│   │
│  │   Node   │    │Developer │    │  Node    │    │  Node    │   │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘   │
│       ▲                                              │          │
│       └──────────────────────────────────────────────┘          │
│                    (循环直到任务完成)                             │
└─────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      工作线程 (ReAct Agent)                      │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐       │
│  │  Agent  │───▶│  Tools  │───▶│Enhance  │───▶│ Context │       │
│  │  Node   │    │  Node   │    │  Tools  │    │ Summary │       │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘       │
│       ▲                                              │          │
│       └──────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     工具管理系统                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Private Tool │    │  Public Tool │    │  Preset Tool │      │
│  │   Storage    │───▶│   Storage    │◄───│   Library    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件

#### 2.2.1 主控制节点 (Main Control Nodes)

| 节点 | 职责 | 文件位置 |
|------|------|----------|
| **Manager Node** | 任务分析和工具规划 | `src/core/nodes.py:97` |
| **Tool Developer Node** | 动态工具生成和管理 | `src/core/nodes.py:279` |
| **Executor Node** | 协调 ReAct Agent 执行任务 | `src/core/nodes.py:370` |
| **Integrator Node** | 整合结果生成最终答案 | `src/core/nodes.py:43` |

#### 2.2.2 工作线程 (ReAct Agent)

| 组件 | 职责 | 文件位置 |
|------|------|----------|
| **ReAct Agent** | 执行具体任务，调用工具 | `src/agents/react.py:48` |
| **Tool Enhancement** | 工具性能分析和增强 | `src/agents/react.py:287` |
| **Context Trimmer** | 上下文管理和截断 | `src/utils/context_trimmer.py` |

#### 2.2.3 工具系统 (Tool System)

| 组件 | 职责 | 文件位置 |
|------|------|----------|
| **Dynamic Tool Loader** | 动态加载和缓存工具 | `src/tools/dynamic_tool_loader.py:611` |
| **Tool Utils** | 工具信息提取和测试 | `src/tools/utils.py` |
| **Preset Tools** | 预置工具（如图像查询） | `src/tools/image_text_query.py` |

## 3. 数据流

### 3.1 任务执行流程

```
1. 用户查询 → Manager Node
   └─ 分析任务需求
   └─ 确定需要使用的工具（现有或新建）
   
2. Manager Node → Tool Developer Node
   └─ 如果有新工具需求，调用 Codex 生成工具代码
   └─ 加载现有工具（从 Private/Public 目录）
   
3. Tool Developer Node → Executor Node
   └─ 绑定工具到 ReAct Agent
   └─ 配置任务执行上下文
   
4. Executor Node → ReAct Agent
   └─ 执行 ReAct 循环（推理-行动-观察）
   └─ 工具调用和增强
   
5. ReAct Agent → Manager Node
   └─ 返回执行结果和上下文摘要
   
6. Manager Node → Integrator Node (当任务完成)
   └─ 验证结果质量
   └─ 整合所有发现生成最终答案
   
7. Integrator Node → 输出最终答案
```

### 3.2 工具进化流程

```
1. 工具执行监控
   └─ 记录每次工具调用的参数和结果
   
2. 执行结果分析
   └─ 分类：输入错误 / 执行失败 / 成功
   
3. 工具增强决策
   └─ 失败工具 → 调用 Codex 生成增强版本
   └─ 成功工具 → 移至 Public 目录共享
   
4. 工具合并优化（Batch 结束时）
   └─ 聚类相似工具
   └─ 合并冗余工具
   └─ 生成通用工具版本
```

## 4. 状态管理

### 4.1 主工作流状态 (State)

```python
class State(MessagesState):
    user_query: str                    # 用户原始查询
    final_answer: str                  # 最终答案
    pending_tool_requests: List[ToolRequest]  # 待创建的工具请求
    task_execution_context: TaskExecutionContext  # 任务执行上下文
    task_failure_report: Optional[str] # 任务失败报告
    tool_usage_guidance: Optional[str] # 工具使用指导
    pending_step_response: str         # 待处理的步骤响应
    task_execution_count: int          # 任务执行计数
    required_tool_names: List[str]     # 所需工具名称列表
    execution_res: str                 # 执行结果
```

### 4.2 ReAct Agent 状态 (AgentState)

```python
class AgentState(TypedDict):
    messages: List[BaseMessage]        # 对话历史
    tool_steps: int                    # 工具执行步数
    retry_count: int                   # 重试计数
```

## 5. 配置系统

### 5.1 环境变量配置 (.env)

```bash
# API Keys
OPENAI_API_KEY=your_key
TAVILY_API_KEY=your_key

# 执行限制
MAX_WORKER_RECURSION_LIMIT=10        # 工作线程最大递归深度
MAX_TASK_EXECUTION_CNT=5             # 任务最大执行次数
WORKER_TOOL_ENHANCE_INTERVAL=2       # 工具增强间隔

# 代理设置
PROXY_URL=http://proxy:port
```

### 5.2 YAML 配置 (conf.yaml)

```yaml
# LLM 模型配置
BASIC_MODEL:
  model: gemini-2.5-pro
  base_url: https://api.example.com/v1
  api_key: ${OPENAI_API_KEY}
  token_limit: 128000

VISION_MODEL:
  model: gemini-2.5-pro
  # ... 视觉模型配置

SUMMARIZE_MODEL:
  model: gemini-2.5-pro
  # ... 摘要模型配置

# 动态工具配置
DYNAMIC_TOOL:
  max_response_tokens: 64000          # 工具响应最大 Token 数
```

## 6. 提示词系统

### 6.1 提示词模板结构

```
src/prompts/templates/
├── give_answer.md          # 答案生成提示词
├── worker.md               # 工作线程系统提示词
├── step_tool_analyzer.md   # 工具分析提示词
├── toolsmiths_agent.md     # 工具生成提示词
├── tool_enhancement.md     # 工具增强提示词
├── tool_merge.md           # 工具合并提示词
├── tool_cluster.md         # 工具聚类提示词
├── analyze_response.md     # 响应分析提示词
└── context_summarizer.md   # 上下文摘要提示词
```

### 6.2 提示词加载器

```python
# src/prompts/loader.py
class PromptLoader:
    def get_prompt(template_name: str, **kwargs) -> str:
        """加载 Jinja2 模板并渲染变量"""
```

## 7. 批处理与进化

### 7.1 批处理流程 (evolve.py)

```
1. 数据加载 (dataloader.py)
   └─ 支持多个数据集：HLE, DeepSearchQA, FinSearchComp, xbench
   
2. 并行任务执行
   └─ 使用 multiprocessing 并行处理 batch 内任务
   └─ 每个任务在独立进程中运行
   
3. 工具优化 (optimize_tools)
   └─ 收集所有成功的工具
   └─ 工具聚类和合并
   └─ 生成公共工具库
   
4. 检查点保存
   └─ 保存工具库状态
   └─ 记录执行结果
```

### 7.2 工具优化策略

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| **Naive Merge** | 选择最小的工具版本作为代表 | 简单重复工具 |
| **LLM Merge** | 使用 LLM 合并相似工具 | 功能相似但实现不同的工具 |
| **Permutation Cluster** | 识别命名变体（如 tool_01, tool_02）| 版本迭代产生的工具 |

## 8. 安全与隔离

### 8.1 工具执行隔离

- **独立 Python 环境**: 使用 `ISOLATED_PYTHON_PATH` 指定的 Python 解释器
- **子进程执行**: 每个工具在独立子进程中运行
- **超时控制**: 默认 120 秒执行超时

### 8.2 代码生成安全

- **Codex 沙箱**: 使用 `--dangerously-bypass-approvals-and-sandbox` 参数（需要人工审核）
- **依赖管理**: 自动安装工具声明的依赖（通过 `uv pip install`）
- **代码验证**: 生成后验证工具元数据和输入模型

## 9. 监控与日志

### 9.1 日志系统

- **结构化日志**: 使用 Python logging 模块
- **任务隔离**: 每个任务有独立的日志文件（`logs/task_{task_id}.log`）
- **Token 统计**: 记录输入/输出 Token 数量

### 9.2 追踪与调试

- **LangSmith 集成**: 支持通过 LangSmith 追踪执行流程
- **上下文变量**: 使用 ContextVar 管理任务 ID
- **调试模式**: 可通过 `enable_debug_logging()` 启用详细日志

## 10. 扩展与集成

### 10.1 添加新工具

1. 在 `src/tools/` 目录创建工具文件
2. 定义 `__TOOL_META__` 和 `InputModel`
3. 实现 `run()` 函数
4. 在 `get_preset_tools()` 中注册

### 10.2 添加新数据集

1. 在 `dataloader.py` 中实现加载逻辑
2. 支持 batch 格式返回
3. 配置数据集路径映射

## 11. 性能优化

### 11.1 缓存机制

- **工具缓存**: 基于文件修改时间的动态工具缓存
- **LLM 配置缓存**: 使用 `@lru_cache` 缓存 YAML 配置
- **响应截断**: 自动截断超长工具响应（默认 64000 tokens）

### 11.2 并发优化

- **异步执行**: 使用 `asyncio` 进行 I/O 并发
- **并行批处理**: 多进程并行处理 batch 任务
- **并发工具构建**: 同时构建多个新工具

## 12. 文件目录结构

```
Yunjue-Agent/
├── src/
│   ├── agents/              # 智能体实现
│   │   └── react.py         # ReAct Agent
│   ├── core/                # 核心工作流
│   │   ├── builder.py       # 图构建器
│   │   └── nodes.py         # 节点实现
│   ├── services/
│   │   └── llms/            # LLM 服务
│   │       └── llm.py       # LLM 工厂
│   ├── tools/               # 工具系统
│   │   ├── dynamic_tool_loader.py
│   │   ├── image_text_query.py
│   │   └── utils.py
│   ├── utils/               # 通用工具
│   │   ├── context_trimmer.py
│   │   ├── utils.py
│   │   └── venv.py
│   ├── prompts/             # 提示词系统
│   │   ├── loader.py
│   │   └── templates/
│   ├── schema/              # 数据类型定义
│   │   └── types.py
│   ├── config/              # 配置系统
│   │   └── config.py
│   └── main.py              # 主入口
├── evolve.py                # 批处理进化脚本
├── dataloader.py            # 数据加载器
├── scripts/                 # 执行脚本
│   ├── evolve.sh            # 进化启动脚本
│   └── evaluate.py          # 评估脚本
├── docs/                    # 文档
├── output/                  # 输出目录（运行时生成）
└── conf.yaml                # 配置文件
```

---

*文档版本: 1.0*  
*最后更新: 2026-02-01*
