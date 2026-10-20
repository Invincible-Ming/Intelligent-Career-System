#!/bin/bash
# RAG 评测快速启动脚本

echo "================================"
echo "RAG 评测系统 - 快速启动"
echo "================================"
echo ""

# 1. 健康检查
read -r -p "请输入评测用户 UUID: " evaluation_owner_id
echo "1. 运行健康检查..."
python -m app.evaluation.health_check --owner-id "$evaluation_owner_id"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 健康检查失败，请先修复问题"
    exit 1
fi

echo ""

# 2. 选择评测模式
echo "请选择评测模式："
echo "  1) 快速评测（仅 baseline，约 3 分钟）"
echo "  2) 完整评测（6 个查询配置对比）"
echo "  3) 自定义评测"
echo ""
read -p "请输入选项 (1-3): " choice

case $choice in
    1)
        echo ""
        echo "🚀 开始快速评测..."
        python -m app.evaluation.cli quick --owner-id "$evaluation_owner_id"
        ;;
    2)
        echo ""
        echo "🚀 开始完整评测..."
        python -m app.evaluation.cli full --owner-id "$evaluation_owner_id"
        ;;
    3)
        read -p "请输入配置文件路径: " config_path
        echo ""
        echo "🚀 开始自定义评测..."
        python -m app.evaluation.cli custom --config "$config_path" --owner-id "$evaluation_owner_id"
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

if [ $? -ne 0 ]; then
    echo "评测执行失败"
    exit 1
fi

echo ""
echo "================================"
echo "✅ 评测完成！"
echo "================================"
