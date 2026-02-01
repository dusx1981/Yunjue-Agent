# Yunjue Agent 架构文档索引

本文档目录包含 Yunjue Agent 系统的完整架构文档。

## 文档清单

### 1. [architecture.md](./architecture.md) - 整体架构文档

包含内容：
- 项目概述和核心特性
- 整体架构图和组件说明
- 数据流和任务执行流程
- 状态管理设计
- 配置系统说明
- 提示词系统结构
- 批处理与进化流程
- 安全与隔离机制
- 监控与日志系统
- 文件目录结构

### 2. [core_modules.md](./core_modules.md) - 核心模块详细文档

包含内容：
- 核心节点模块（Manager, Tool Developer, Executor, Integrator）
- ReAct Agent 模块详解
- 工具加载器模块
- 通用工具模块（Codex 调用、工具分析、响应分析等）
- LLM 服务模块
- 配置系统
- 数据类型定义
- 提示词系统
- 批处理进化模块
- 上下文管理模块

### 3. [dataflow.md](./dataflow.md) - 数据流与执行流程文档

包含内容：
- 单任务执行的 8 个详细步骤
- ReAct Agent 内部节点流程
- 工具增强详细流程
- 批处理训练流程
- 工具生命周期管理
- 错误处理与恢复机制
- 性能优化点说明

## 快速导航

| 如果你想了解... | 查看文档 |
|----------------|---------|
| 系统整体架构 | [architecture.md](./architecture.md) |
| 某个具体模块的实现 | [core_modules.md](./core_modules.md) |
| 任务执行的详细流程 | [dataflow.md](./dataflow.md) |
| 工具系统如何工作 | [architecture.md#22-核心组件](./architecture.md#22-核心组件) + [core_modules.md#3-工具加载器模块](./core_modules.md#3-工具加载器模块) |
| 错误处理和恢复 | [dataflow.md#4-错误处理与恢复](./dataflow.md#4-错误处理与恢复) |
| 如何添加新工具 | [architecture.md#101-添加新工具](./architecture.md#101-添加新工具) |

## 文档使用建议

1. **新开发者**: 建议按顺序阅读
   - 先读 `architecture.md` 了解整体架构
   - 再读 `dataflow.md` 理解执行流程
   - 最后按需查阅 `core_modules.md`

2. **问题排查**: 根据问题类型选择文档
   - 工具相关问题 → `core_modules.md` 第 3、4 节
   - 执行流程问题 → `dataflow.md`
   - 配置问题 → `architecture.md` 第 5 节

3. **二次开发**: 参考具体模块文档
   - 修改节点逻辑 → `core_modules.md` 第 1 节
   - 添加新工具 → `architecture.md` 第 10.1 节
   - 修改提示词 → `core_modules.md` 第 8 节

## 文档更新

- 版本: 1.0
- 最后更新: 2026-02-01
- 基于代码版本: Initial Release

如有文档相关问题，请参考主项目的 README.md 获取支持信息。
