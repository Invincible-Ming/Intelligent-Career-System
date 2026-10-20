"""兼容完整评测入口，复用 CLI 配置与用户范围校验。

python -m app.evaluation.run_evaluation --owner-id <用户UUID>
"""
import argparse
import asyncio
import uuid

from app.evaluation.cli import create_full_experiment, run_evaluation


async def main(owner_id: str) -> None:
    await run_evaluation(create_full_experiment(), owner_id=owner_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按用户范围运行完整 RAG 评测")
    parser.add_argument("--owner-id", type=uuid.UUID, required=True)
    args = parser.parse_args()
    asyncio.run(main(str(args.owner_id)))
