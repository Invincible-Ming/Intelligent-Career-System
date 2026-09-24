"""Local-only initialization and explicitly requested legacy cleanup."""
from __future__ import annotations
import argparse
import asyncio
import json
import os
from pathlib import Path
import secrets
import sys
from datetime import datetime, timezone

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)
from sqlalchemy import delete, select, text
from app.core.database import AsyncSessionLocal, close_database, engine, init_database
from app.core.models import AgentRun, Conversation, Document, User
from app.security.auth import hash_password


async def bootstrap_admin():
    async with AsyncSessionLocal() as session:
        if (await session.execute(select(User.id).where(User.is_admin.is_(True)))).first():
            print('Administrator already exists; no account changed.')
            return
        password = secrets.token_urlsafe(24)
        session.add(User(username='career_admin', password_hash=hash_password(password), is_admin=True))
        directory = BACKEND / '.local'
        directory.mkdir(mode=0o700, exist_ok=True)
        path = directory / 'initial_login.json'
        # Fail before creating the account if its local credential file exists.
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w') as output:
            json.dump({'username': 'career_admin', 'password': password}, output, indent=2)
        try:
            await session.commit()
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        print('Administrator created. Credentials: backend/.local/initial_login.json (0600).')


async def purge_legacy():
    from app.services.milvus_service import milvus_service
    from app.services.minio_service import minio_service
    from app.services.bm25_service import bm25_service
    await minio_service.initialize()
    await milvus_service.initialize()
    async with AsyncSessionLocal() as session:
        owned_ids = set(
            map(str, (await session.execute(select(Document.id).where(Document.owner_id.is_not(None)))).scalars()))
        legacy_documents = (await session.execute(
            select(Document.id, Document.minio_object_key).where(Document.owner_id.is_(None)))).all()
        legacy_runs = list((await session.execute(select(AgentRun.id).where(AgentRun.owner_id.is_(None)))).scalars())
        legacy_conversations = list(
            (await session.execute(select(Conversation.id).where(Conversation.owner_id.is_(None)))).scalars())
    chunks = await milvus_service.list_chunks()
    vector_ids = {str(chunk['document_id']) for chunk in chunks if str(chunk['document_id']) not in owned_ids}
    vector_ids.update(str(row.id) for row in legacy_documents)
    for identifier in vector_ids:
        await milvus_service.delete_document(identifier)
        await bm25_service.delete_document(identifier)
    objects = await asyncio.to_thread(
        lambda: list(minio_service.client.list_objects(minio_service.bucket, prefix='documents/', recursive=True)))
    keys = {row.minio_object_key for row in legacy_documents}
    for entry in objects:
        parts = entry.object_name.split('/')
        if len(parts) >= 3 and parts[1] not in owned_ids:
            keys.add(entry.object_name)
    for key in keys:
        await minio_service.delete(key)
    checkpoint_rows = 0
    async with engine.begin() as connection:
        # Only the project's checkpoint tables; preserve all owned run threads.
        owned_threads = list(
            map(str, (await connection.execute(select(AgentRun.id).where(AgentRun.owner_id.is_not(None)))).scalars()))
        for table in ('checkpoint_writes', 'checkpoint_blobs', 'checkpoints'):
            if (await connection.execute(text('SELECT to_regclass(:table)'), {'table': 'public.' + table})).scalar():
                from sqlalchemy import bindparam
                query = text(f'DELETE FROM {table}' + (' WHERE thread_id NOT IN :owned' if owned_threads else ''))
                if owned_threads:
                    query = query.bindparams(bindparam('owned', expanding=True))
                result = await connection.execute(query, {'owned': owned_threads} if owned_threads else {})
                checkpoint_rows += result.rowcount
        await connection.execute(delete(Conversation).where(Conversation.owner_id.is_(None)))
        await connection.execute(delete(AgentRun).where(AgentRun.owner_id.is_(None)))
        await connection.execute(delete(Document).where(Document.owner_id.is_(None)))
    await bm25_service.rebuild(await milvus_service.list_chunks())
    # Confirm original objects and vector records are actually gone.
    for key in keys:
        if await minio_service.exists(key):
            raise RuntimeError('Legacy object cleanup verification failed')
    remaining = await milvus_service.list_chunks()
    if any(str(row['document_id']) not in owned_ids for row in remaining):
        raise RuntimeError('Legacy vector cleanup verification failed')
    summary = {'documents': len(legacy_documents), 'runs': len(legacy_runs), 'conversations': len(legacy_conversations),
               'objects': len(keys), 'vector_documents': len(vector_ids), 'checkpoint_rows': checkpoint_rows,
               'completed_at': datetime.now(timezone.utc).isoformat()}
    directory = BACKEND / '.local'
    directory.mkdir(mode=0o700, exist_ok=True)
    (directory / 'legacy_cleanup.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    await milvus_service.close()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap-admin', action='store_true')
    parser.add_argument('--delete-unowned', action='store_true',
                        help='Destructive; requires explicit operator authorization')
    args = parser.parse_args()
    try:
        await init_database()
        if args.delete_unowned:
            await purge_legacy()
        if args.bootstrap_admin:
            await bootstrap_admin()
    finally:
        await close_database()


if __name__ == '__main__':
    asyncio.run(main())
