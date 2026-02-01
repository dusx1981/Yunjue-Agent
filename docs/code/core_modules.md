# 核心模块详细文档

## 1. 核心节点模块 (src/core/nodes.py)

### 1.1 Manager Node (任务管理节点)

**位置**: `src/core/nodes.py:97`

**职责**:
- 分析用户查询和任务执行上下文
- 确定需要使用的工具（现有工具或新建工具）
- 协调任务重试和失败恢复

**关键逻辑**:
```python
async def manager_node(state: State, config: RunnableConfig):
    # 1. 检查任务执行计数和失败报告
    # 2. 如果有失败，分析响应内容
    # 3. 调用 analyze_task_tools 分析所需工具
    # 4. 返回工具请求和下一步跳转指令
```

**跳转逻辑**:
- `integrator`: 任务成功完成
- `tool_developer`: 需要创建新工具或加载现有工具

### 1.2 Tool Developer Node (工具开发节点)

**位置**: `src/core/nodes.py:279`

**职责**:
- 并发构建多个新工具
- 管理工具生成失败和重试
- 绑定工具到任务执行上下文

**工具构建流程**:
```python
async def _build_single_tool(tool_req, tool_index, total, dynamic_tools_dir):
    # 1. 生成工具文件名
    # 2. 调用 Codex 生成代码（最多 3 次重试）
    # 3. 验证生成的工具代码
    # 4. 安装工具依赖
```

### 1.3 Executor Node (执行节点)

**位置**: `src/core/nodes.py:370`

**职责**:
- 初始化 ReAct Agent
- 执行 ReAct 循环
- 处理递归限制和上下文摘要

**执行流程**:
1. 准备工具列表和系统提示词
2. 创建 ReActAgent 实例
3. 流式执行并监控步数
4. 检测递归限制并生成上下文摘要
5. 提取工具执行记录

### 1.4 Integrator Node (整合节点)

**位置**: `src/core/nodes.py:43`

**职责**:
- 整合执行结果生成最终答案
- 使用结构化输出确保答案格式

**输出格式**:
```python
class GiveAnswerResponse(BaseModel):
    final_answer: str           # 直接答案
    reasoning_summary: str      # 推理摘要（1-2 句）
```

## 2. ReAct Agent 模块 (src/agents/react.py)

### 2.1 ReActAgent 类

**位置**: `src/agents/react.py:48`

**核心功能**:
- ReAct 风格的多步推理和工具调用
- 自动工具增强
- 递归限制和重试机制
- 上下文摘要

**节点流程**:
```
START → agent → [should_continue] → tools → [need_enhance] → enhance_tools → context_summary → agent
                  ↓                                              ↓
                END                                          context_summary
                  ↑
                rollback (当响应为空时)
```

### 2.2 工具增强系统

**位置**: `src/agents/react.py:287`

**功能**: `enhance_tools()`

**流程**:
1. 从消息历史中提取工具执行记录
2. 分类执行结果：
   - **Input Error**: 输入验证失败
   - **Execution Failure**: 执行异常或空返回
   - **Success**: 成功执行
3. 对失败工具调用 Codex 生成增强版本
4. 更新工具绑定和消息历史

### 2.3 回滚机制

**位置**: `src/agents/react.py:193`

**触发条件**: LLM 返回空响应

**回滚逻辑**:
```python
def rollback(self, state: AgentState):
    # 1. 找到倒数第二个 AIMessage
    # 2. 移除该消息及其后续所有消息
    # 3. 如果有 ToolMessage 被移除，减少 tool_steps 计数
    # 4. 增加 retry_count
```

## 3. 工具加载器模块 (src/tools/dynamic_tool_loader.py)

### 3.1 动态工具加载

**位置**: `src/tools/dynamic_tool_loader.py:611`

**函数**: `load_dynamic_tools(tools_directory, user_query)`

**功能**:
- 从指定目录加载所有 Python 工具文件
- 基于文件修改时间的缓存机制
- 自动安装工具依赖

**缓存策略**:
```python
_dynamic_tools_cache: Dict[str, tuple[float, Any]] = {}
# 键: 文件路径
# 值: (修改时间, 工具对象)
```

### 3.2 工具创建

**位置**: `src/tools/dynamic_tool_loader.py:678`

**函数**: `create_tool_from_module(file_path, user_query)`

**流程**:
1. 在隔离环境中加载模块元数据
2. 创建 LangChain 工具函数
3. 在子进程中执行工具代码
4. 截断超长响应（默认 64000 tokens）

**隔离执行**:
```python
# 使用 ISOLATED_PYTHON_PATH 执行工具代码
proc = subprocess.run(
    [str(ISOLATED_PYTHON_PATH), "-c", code],
    input=input_json,
    capture_output=True,
    text=True,
    timeout=120,
)
```

## 4. 通用工具模块 (src/utils/utils.py)

### 4.1 Codex 调用

**位置**: `src/utils/utils.py:30`

**函数**: `call_codex_exec(prompt, output_file)`

**功能**:
- 调用 Codex CLI 生成代码
- 支持超时控制（默认 20 分钟）
- 自动提取 Python 代码块
- 验证生成的工具代码

### 4.2 任务工具分析

**位置**: `src/utils/utils.py:265`

**函数**: `analyze_task_tools(user_query, ...)`

**流程**:
1. 加载所有可用工具（预设 + 动态）
2. 构建工具分析提示词
3. 调用 LLM 分析所需工具
4. 返回工具名称列表、使用指导和新建工具请求

**输出**:
```python
return (
    required_tool_names,    # 现有工具名称列表
    tool_usage_guidance,    # 工具使用指导
    tool_requests          # 新建工具请求列表
)
```

### 4.3 响应分析

**位置**: `src/utils/utils.py:517`

**函数**: `analyze_response(pending_response)`

**分类**:
- **FINISH**: 任务成功完成
- **RETRY**: 需要重试，附带失败原因

### 4.4 上下文摘要

**位置**: `src/utils/utils.py:350`

**函数**: `summarize_context(user_query, history_tool_executions, context_summary, is_recur_limit_exceeded)`

**用途**:
- 当 ReAct Agent 达到递归限制时总结执行历史
- 提取关键发现和工具使用反馈
- 生成上下文摘要供 Manager Node 使用

### 4.5 工具增强

**位置**: `src/utils/utils.py:563`

**函数**: `tool_enhancement(tool_filename, historical_call_records, dynamic_tools_dir)`

**流程**:
1. 读取原始工具代码
2. 基于历史调用记录生成增强提示词
3. 调用 Codex 生成增强版本（最多 3 次重试）
4. 运行测试验证增强后的工具
5. 保存成功的增强版本

## 5. LLM 服务模块 (src/services/llms/llm.py)

### 5.1 LLM 工厂

**位置**: `src/services/llms/llm.py:40`

**函数**: `create_llm(llm_type: LLMType)`

**支持的 LLM 类型**:
```python
class LLMType(Enum):
    BASIC = "basic"           # 基础模型（主要用于推理）
    VISION = "vision"         # 视觉模型
    SUMMARIZE = "summarize"   # 摘要模型
    CLUSTER = "cluster"       # 聚类模型
    TOOL_ANALYZE = "tool_analyze"  # 工具分析模型
```

**配置映射**:
```python
LLM_CONFIG_MAP = {
    LLMType.BASIC: "BASIC_MODEL",
    LLMType.VISION: "VISION_MODEL",
    LLMType.SUMMARIZE: "SUMMARIZE_MODEL",
    LLMType.CLUSTER: "CLUSTER_MODEL",
    LLMType.TOOL_ANALYZE: "TOOL_ANALYZE_MODEL",
}
```

### 5.2 配置加载

**位置**: `src/services/llms/llm.py:23`

**缓存机制**: 使用 `@lru_cache` 缓存 YAML 配置

## 6. 配置系统 (src/config/config.py)

### 6.1 Configuration 类

**位置**: `src/config/config.py:12`

**配置项**:
```python
@dataclass(kw_only=True)
class Configuration:
    dynamic_tools_dir: str           # 私有动态工具目录
    dynamic_tools_public_dir: str    # 公共动态工具目录
    max_task_execution_cnt: int = 5  # 最大任务执行次数
```

**解析优先级**:
1. `config["configurable"]` 中的值
2. 环境变量（大写形式）
3. 默认值

### 6.2 YAML 配置加载

**位置**: `src/config/config.py:35`

**函数**: `load_yaml_config(file_path)`

**错误处理**:
- 文件不存在时记录错误并返回空字典
- YAML 解析错误时记录错误详情

## 7. 数据类型定义 (src/schema/types.py)

### 7.1 核心数据模型

**ToolRequest**: 工具创建请求
```python
class ToolRequest(BaseModel):
    name: str                  # 工具名称
    description: str           # 工具描述
    input_schema: Dict         # 输入参数模式
    output_schema: Dict        # 输出参数模式
```

**ToolExecutionRecord**: 工具执行记录
```python
class ToolExecutionRecord(BaseModel):
    tool_name: str
    tool_call_id: str
    arguments: Dict
    result: Optional[Any]
    error: Optional[str]
```

**TaskExecutionContext**: 任务执行上下文
```python
class TaskExecutionContext(BaseModel):
    bound_tools: List[BaseTool]           # 绑定的工具列表
    tool_executions: List[ToolExecutionRecord]  # 工具执行历史
    context_summary: str                  # 上下文摘要
```

**State**: 主工作流状态
```python
class State(MessagesState):
    user_query: str
    final_answer: str
    pending_tool_requests: List[ToolRequest]
    task_execution_context: Optional[TaskExecutionContext]
    # ... 其他状态字段
```

## 8. 提示词系统 (src/prompts/)

### 8.1 提示词加载器

**位置**: `src/prompts/loader.py`

**功能**: 使用 Jinja2 模板引擎加载和渲染提示词

### 8.2 提示词模板列表

| 模板文件 | 用途 | 关键变量 |
|---------|------|---------|
| `give_answer.md` | 答案生成 | `user_query` |
| `worker.md` | 工作线程系统提示 | `user_query`, `failure_report`, `context_summary` |
| `step_tool_analyzer.md` | 工具需求分析 | `user_query`, `available_tools`, `failure_report` |
| `toolsmiths_agent.md` | 工具代码生成 | `tool_request_json`, `proxy_url` |
| `tool_enhancement.md` | 工具增强 | `original_tool_code`, `historical_call_records` |
| `tool_merge.md` | 工具合并 | `tools_code`, `suggest_name` |
| `tool_cluster.md` | 工具聚类 | `available_tools` |
| `analyze_response.md` | 响应质量分析 | `pending_response` |
| `context_summarizer.md` | 上下文摘要 | `user_query`, `tool_execution_history`, `context_summary` |

## 9. 批处理进化模块 (evolve.py)

### 9.1 训练流程

**位置**: `evolve.py:302`

**函数**: `train(data_iter, train_steps, start, run_dir, prediction_file, timeout, merge_policy)`

**流程**:
1. 遍历数据集（按 batch）
2. 并行执行 batch 内任务
3. 收集执行结果
4. 优化工具库（聚类合并）
5. 保存检查点

### 9.2 工具优化

**位置**: `evolve.py:155`

**函数**: `optimize_tools(task_ids, step, run_dir, merge_policy)`

**流程**:
1. 收集所有成功的工具（Private + Public）
2. 提取工具元数据
3. 工具聚类
4. 根据策略合并工具
5. 保存优化后的工具库

### 9.3 工具聚类策略

**智能聚类**:
```python
def cluster_tools(tool_meta_list: List[dict]):
    # 使用 LLM 分析工具功能相似性
    # 返回工具簇列表
```

**朴素聚类**:
```python
def naive_cluster_tools(tool_meta_list: List[dict]):
    # 基于工具名称去除版本后缀聚类
    # 例如：tool_01, tool_02 → tool
```

## 10. 上下文管理 (src/utils/context_trimmer.py)

### 10.1 上下文截断器

**位置**: `src/utils/context_trimmer.py`

**功能**:
- 监控 Token 使用量
- 自动截断历史消息
- 保留关键上下文

**策略**:
```python
class ContextTrimmer:
    def trim(self, state: AgentState) -> AgentState:
        # 1. 检查当前 Token 数是否超限
        # 2. 如果超限，移除最早的消息对
        # 3. 保留 SystemMessage 和最新的消息
        # 4. 添加截断标记
```

---

*文档版本: 1.0*  
*最后更新: 2026-02-01*
