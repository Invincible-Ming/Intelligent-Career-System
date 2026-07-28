"""
MinIO 文件存储服务。

原始简历、JD 和知识文档统一保存到 career-documents Bucket。
"""

from __future__ import annotations

import asyncio
import io
import re
from pathlib import Path

from minio import Minio
from minio.error import S3Error

from app.config import settings


class MinioService:
    """MinIO 文件上传、下载和删除服务。"""

    def __init__(self) -> None:
        self.client = Minio(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        self.bucket = settings.MINIO_BUCKET

    async def initialize(self) -> None:
        """创建项目使用的 Bucket。"""

        exists = await asyncio.to_thread(
            self.client.bucket_exists,
            self.bucket,
        )

        if not exists:
            try:
                await asyncio.to_thread(
                    self.client.make_bucket,
                    self.bucket,
                )
            except S3Error as exc:
                # 多个进程同时启动时，Bucket 可能已被其他进程创建。
                if exc.code not in {
                    "BucketAlreadyExists",
                    "BucketAlreadyOwnedByYou",
                }:
                    raise

    async def upload(
        self,
        *,
        document_id: str,
        filename: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """
        上传文件并返回 MinIO Object Key。

        Object Key 示例：

            documents/文档ID/resume.pdf
        """

        if not data:
            raise ValueError("不能上传空文件")

        safe_filename = sanitize_filename(filename)
        object_key = (
            f"documents/{document_id}/{safe_filename}"
        )

        await asyncio.to_thread(
            self.client.put_object,
            self.bucket,
            object_key,
            io.BytesIO(data),
            len(data),
            content_type=content_type,
        )

        return object_key

    async def download(
        self,
        object_key: str,
    ) -> bytes:
        """下载文件并返回字节数据。"""

        return await asyncio.to_thread(
            self._download_sync,
            object_key,
        )

    def _download_sync(
        self,
        object_key: str,
    ) -> bytes:
        """同步下载文件，并正确释放 HTTP 连接。"""

        response = self.client.get_object(
            self.bucket,
            object_key,
        )

        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    async def delete(
        self,
        object_key: str,
    ) -> None:
        """删除指定文件。"""

        await asyncio.to_thread(
            self.client.remove_object,
            self.bucket,
            object_key,
        )

    async def exists(
        self,
        object_key: str,
    ) -> bool:
        """判断文件是否存在。"""

        try:
            await asyncio.to_thread(
                self.client.stat_object,
                self.bucket,
                object_key,
            )
            return True
        except S3Error as exc:
            if exc.code in {
                "NoSuchKey",
                "NoSuchObject",
                "NotFound",
            }:
                return False
            raise


def sanitize_filename(filename: str) -> str:
    """清理文件名，防止路径穿越和特殊字符问题。"""

    filename = Path(filename).name.strip()
    filename = re.sub(
        r"[^A-Za-z0-9._\-\u4e00-\u9fff]",
        "_",
        filename,
    )

    return filename[:200] or "document"


minio_service = MinioService()