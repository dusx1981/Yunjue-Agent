#!/bin/bash
# 使用代码生成桥接层运行 Yunjue Agent
# 支持多种后端：codex_cli, qwen, kimi

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 显示帮助信息
show_help() {
    echo -e "${BLUE}Yunjue Agent - 代码生成桥接层${NC}"
    echo ""
    echo "用法: $0 [后端] [数据集] [运行名称] [批大小] [起始位置]"
    echo ""
    echo "后端选项:"
    echo "  codex_cli  - 使用 Codex CLI (默认，需要安装 codex 命令)"
    echo "  qwen       - 使用阿里云千问 API (推荐，无需额外安装)"
    echo "  kimi       - 使用 Moonshot Kimi API (可选)"
    echo ""
    echo "示例:"
    echo "  $0 qwen DEEPSEARCHQA my_run 1 0"
    echo "  $0 codex_cli HLE test_run 2 0"
    echo ""
    echo "环境变量:"
    echo "  CODE_GEN_BACKEND - 设置默认后端 (codex_cli|qwen|kimi)"
    echo "  HF_ENDPOINT      - HuggingFace 镜像地址"
    exit 0
}

# 如果没有参数或参数是 help，显示帮助
if [ $# -eq 0 ] || [ "$1" = "help" ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
fi

# 解析后端
BACKEND=$1
shift  # 移除第一个参数

# 验证后端
if [[ ! "$BACKEND" =~ ^(codex_cli|qwen|kimi)$ ]]; then
    echo -e "${YELLOW}错误: 未知的后端 '$BACKEND'${NC}"
    echo "可用的后端: codex_cli, qwen, kimi"
    exit 1
fi

# 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}错误: 虚拟环境不存在${NC}"
    echo "请先运行: uv venv .venv"
    exit 1
fi

# 激活虚拟环境
source .venv/bin/activate

# 设置环境变量
export CODE_GEN_BACKEND=$BACKEND
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}  Yunjue Agent - 代码生成桥接层${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""
echo -e "${GREEN}✓${NC} 虚拟环境已激活"
echo -e "${GREEN}✓${NC} 代码生成后端: $BACKEND"
echo -e "${GREEN}✓${NC} HF_ENDPOINT: $HF_ENDPOINT"
echo ""

# 解析其他参数
DATASET=${1:-DEEPSEARCHQA}
RUN_NAME=${2:-bridge_run_$(date +%Y%m%d_%H%M%S)}
BATCH_SIZE=${3:-1}
START=${4:-0}

echo -e "${BLUE}运行配置:${NC}"
echo "  数据集: $DATASET"
echo "  运行名称: $RUN_NAME"
echo "  批大小: $BATCH_SIZE"
echo "  起始位置: $START"
echo ""

# 根据后端提示
if [ "$BACKEND" = "qwen" ]; then
    echo -e "${YELLOW}提示: 使用千问 API 需要配置 conf.yaml 中的 DashScope API Key${NC}"
    echo ""
elif [ "$BACKEND" = "codex_cli" ]; then
    echo -e "${YELLOW}提示: 使用 Codex CLI 需要安装 codex 命令并配置 OPENAI_API_KEY${NC}"
    echo ""
fi

# 运行 evolve 脚本
./scripts/evolve.sh \
    --dataset "$DATASET" \
    --run_name "$RUN_NAME" \
    --batch_size "$BATCH_SIZE" \
    --start "$START" \
    "${@:5}"
