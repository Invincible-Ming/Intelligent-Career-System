"""
评测系统健康检查和验证脚本。

用法：
    python -m app.evaluation.health_check --owner-id <用户UUID>
"""

import asyncio
import json
import logging
from pathlib import Path

from app.core.prompts import MODEL_HEALTHCHECK_PROMPT
from app.evaluation.dataset import TestDataset
from app.evaluation.llm import evaluation_llm
from app.services.bm25_service import bm25_service
from app.services.milvus_service import milvus_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def check_services() -> dict[str, bool]:
    """检查依赖服务状态。"""

    logger.info("检查依赖服务...")

    results = {}

    # 检查 Milvus
    try:
        milvus_ok = await milvus_service.health_check()
        results["milvus"] = milvus_ok
        logger.info(f"✓ Milvus: {'可用' if milvus_ok else '不可用'}")
    except Exception as exc:
        results["milvus"] = False
        logger.error(f"✗ Milvus: 检查失败 - {exc}")

    # 检查实际评测模型，避免 MiMo 评测仍然消耗百炼聊天额度。
    try:
        await evaluation_llm.chat(
            messages=[{"role": "user", "content": MODEL_HEALTHCHECK_PROMPT}],
            max_output_tokens=32, max_retries=0,
        )
        results["evaluation_llm"] = True
        logger.info("✓ 评测模型 %s / %s: 可用", evaluation_llm.provider, evaluation_llm.model)
    except Exception as exc:
        results["evaluation_llm"] = False
        logger.error("✗ 评测模型: 检查失败 - %s", type(exc).__name__)

    # 检查 BM25
    try:
        chunk_count = bm25_service.chunk_count
        results["bm25"] = chunk_count > 0
        logger.info(f"✓ BM25: 已加载 {chunk_count} 条文档")
    except Exception as exc:
        results["bm25"] = False
        logger.error(f"✗ BM25: 检查失败 - {exc}")

    return results


async def validate_dataset(dataset_path: str) -> bool:
    """验证测试数据集。"""

    logger.info(f"验证测试数据集：{dataset_path}")

    try:
        dataset_file = Path(dataset_path)

        if not dataset_file.exists():
            logger.error(f"✗ 数据集文件不存在：{dataset_path}")
            return False

        data = json.loads(await asyncio.to_thread(dataset_file.read_text, encoding="utf-8"))

        dataset = TestDataset.model_validate(data)

        logger.info(f"✓ 数据集名称：{dataset.name}")
        logger.info(f"✓ 样本数量：{dataset.size}")

        if dataset.size < 1:
            logger.error("✗ 数据集为空")
            return False

        if dataset.size < 30:
            logger.warning(f"⚠ 数据集样本少于 30 条（当前 {dataset.size} 条）")

        # 检查样本质量
        for i, test_case in enumerate(dataset.test_cases[:5]):
            logger.info(f"  样本 {i + 1}: {test_case.query[:50]}...")

        logger.info("✓ 数据集验证通过")
        return True

    except Exception as exc:
        logger.error(f"✗ 数据集验证失败：{exc}")
        return False


async def test_evaluation_flow(owner_id: str) -> bool:
    """测试评测流程。"""

    logger.info("测试评测流程...")

    try:
        from app.evaluation.config import EvaluationConfig, ExperimentConfig, SearchConfig
        from app.evaluation.executor import rag_executor
        from app.evaluation.scope import prepare_evaluation_scope

        # 简单的查询测试
        config = EvaluationConfig(
            name="test",
            search=SearchConfig(
                top_k=3,
                search_mode="hybrid",
            ),
        )

        scope = await prepare_evaluation_scope(
            owner_id=owner_id,
            config=ExperimentConfig(
                experiment_name="health_check", baseline=config,
                test_dataset_path="app/evaluation/test_dataset_30.json",
            ),
        )
        result = await rag_executor.execute(
            query="什么是 Python？",
            config=config,
            owner_id=scope.owner_id,
            allowed_document_ids=list(scope.document_ids),
        )

        logger.info(f"✓ 检索到 {result['retrieved_count']} 条结果")
        logger.info(f"✓ 答案长度：{len(result['answer'])} 字符")

        if result["retrieved_count"] == 0:
            logger.warning("⚠ 未检索到任何结果，可能需要先上传文档")

        logger.info("✓ 评测流程测试通过")
        return True

    except Exception as exc:
        logger.error(f"✗ 评测流程测试失败：{exc}")
        import traceback

        traceback.print_exc()
        return False


async def main(owner_id: str):
    """运行健康检查。"""

    print("\n" + "=" * 80)
    print("🏥 RAG 评测系统健康检查")
    print("=" * 80 + "\n")

    all_ok = True

    # This command runs outside FastAPI's lifespan as well.
    from app.evaluation.scope import normalize_owner_id

    owner_id = normalize_owner_id(owner_id)
    try:
        await milvus_service.initialize()
        await bm25_service.rebuild(await milvus_service.list_chunks(owner_id=owner_id))
    except Exception as exc:
        logger.error("评测索引初始化失败：%s", type(exc).__name__)
        all_ok = False

    # 1. 检查服务
    services = await check_services()

    if not all(services.values()):
        logger.error("\n❌ 部分服务不可用，评测系统可能无法正常工作")
        all_ok = False

    print()

    # 2. 验证数据集
    dataset_ok = await validate_dataset(
        "app/evaluation/test_dataset_30.json"
    )

    if not dataset_ok:
        logger.error("\n❌ 数据集验证失败")
        all_ok = False

    print()

    # 3. 测试评测流程
    flow_ok = await test_evaluation_flow(owner_id)
    await evaluation_llm.close()

    if not flow_ok:
        logger.error("\n❌ 评测流程测试失败")
        all_ok = False

    print("\n" + "=" * 80)

    if all_ok:
        print("✅ 健康检查全部通过，评测系统可以正常使用！")
    else:
        print("❌ 健康检查发现问题，请修复后再运行评测")

    print("=" * 80 + "\n")

    if not all_ok:
        import sys

        sys.exit(1)


if __name__ == "__main__":
    import argparse
    import uuid

    parser = argparse.ArgumentParser(description="指定用户范围的评测健康检查")
    parser.add_argument("--owner-id", type=uuid.UUID, required=True)
    args = parser.parse_args()
    asyncio.run(main(str(args.owner_id)))
