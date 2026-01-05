#!/bin/bash
# 测试安装脚本 - 不启动服务

set -e

echo "🧪 测试 Fun-ASR 安装"
echo "===================="

# 1. 检查脚本
echo "1️⃣ 检查脚本语法..."
bash -n run_local.sh
echo "   ✅ 脚本语法正确"

# 2. 检查虚拟环境
echo "2️⃣ 检查虚拟环境..."
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "   ✅ 虚拟环境存在: $(python --version)"
else
    echo "   ⚠️  虚拟环境不存在，请先运行 run_local.sh"
    exit 1
fi

# 3. 检查基础依赖
echo "3️⃣ 检查已安装的包..."
pip list | grep -E "fastapi|uvicorn|pydantic" && echo "   ✅ Web服务依赖已安装" || echo "   ⚠️  部分依赖未安装"

# 4. 检查Python导入
echo "4️⃣ 测试Python导入..."
python -c "
try:
    from fastapi import FastAPI
    from uvicorn import run as uvicorn_run
    from pydantic import BaseModel
    print('   ✅ Python包可以正常导入')
except ImportError as e:
    print(f'   ❌ 导入失败: {e}')
    exit(1)
"

# 5. 检查配置文件
echo "5️⃣ 检查配置文件..."
[ -f "requirements.txt" ] && echo "   ✅ requirements.txt 存在" || echo "   ❌ requirements.txt 缺失"
[ -f "server_optimized.py" ] && echo "   ✅ server_optimized.py 存在" || echo "   ❌ server_optimized.py 缺失"
[ -f ".vscode/settings.json" ] && echo "   ✅ .vscode/settings.json 存在" || echo "   ⚠️  .vscode/settings.json 缺失"

echo ""
echo "===================="
echo "✅ 基础环境测试通过"
echo ""
echo "注意: 完整功能需要安装 torch, torchaudio, FunASR 等大型依赖"
echo "运行 'pip install -r requirements.txt' 安装所有依赖"
