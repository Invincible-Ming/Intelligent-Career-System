"""Provision a fresh MCP reader without changing grants of existing users."""
import argparse
import json
import os
import secrets
import sys
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row

from db_guard import ACCESS_QUERY, ROLE_QUERY, check_identity
from policy import DATABASE_ROLE, VIEWS

ROOT = Path(__file__).resolve().parents[2]


def provision(admin_dsn, output, container_host=None):
    password = secrets.token_urlsafe(36)
    admin = conninfo_to_dict(admin_dsn.replace("postgresql+asyncpg://", "postgresql://", 1))
    host = container_host or admin.get("host", "localhost")
    if host in ("localhost", "127.0.0.1", "::1"):
        host = "host.docker.internal"
    if output.exists():
        raise ValueError("只读凭据文件已存在，不自动覆盖或轮换账号")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".pending")
    # Prepare the private file before committing SQL; a failed transaction
    # removes it, and an existing role is never silently reused or modified.
    with psycopg.connect(admin_dsn.replace("postgresql+asyncpg://", "postgresql://", 1), connect_timeout=5) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (DATABASE_ROLE,))
            if cursor.fetchone():
                raise ValueError("专用账号已存在；请审核其权限，不自动复用或修改已有账号")
            cursor.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {} NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS NOINHERIT CONNECTION LIMIT 4").format(sql.Identifier(DATABASE_ROLE), sql.Literal(password)))
            cursor.execute(sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(sql.Identifier(DATABASE_ROLE)))
            cursor.execute(sql.SQL("ALTER ROLE {} SET statement_timeout = '3s'").format(sql.Identifier(DATABASE_ROLE)))
            cursor.execute(sql.SQL("ALTER ROLE {} SET idle_in_transaction_session_timeout = '5s'").format(sql.Identifier(DATABASE_ROLE)))
            cursor.execute(sql.SQL("ALTER ROLE {} SET search_path = mcp_safe, pg_catalog").format(sql.Identifier(DATABASE_ROLE)))
            cursor.execute("CREATE SCHEMA IF NOT EXISTS mcp_safe")
            cursor.execute("REVOKE ALL ON SCHEMA mcp_safe FROM PUBLIC")
            cursor.execute("CREATE OR REPLACE VIEW mcp_safe.knowledge_inventory WITH (security_barrier=true) AS SELECT status, count(*)::integer AS document_count, COALESCE(sum(chunk_count),0)::bigint AS chunk_count FROM public.documents WHERE document_type = 'knowledge' GROUP BY status")
            cursor.execute("CREATE OR REPLACE VIEW mcp_safe.evaluation_summary WITH (security_barrier=true) AS SELECT count(*)::integer AS experiment_count, COALESCE(sum(test_count),0)::bigint AS test_count, COALESCE(sum(success_count),0)::bigint AS success_count, COALESCE(sum(failure_count),0)::bigint AS failure_count FROM public.evaluation_records")
            cursor.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(admin.get("dbname", connection.info.dbname)), sql.Identifier(DATABASE_ROLE)))
            cursor.execute(sql.SQL("GRANT USAGE ON SCHEMA mcp_safe TO {}").format(sql.Identifier(DATABASE_ROLE)))
            for view in VIEWS:
                cursor.execute(sql.SQL("REVOKE ALL ON mcp_safe.{} FROM PUBLIC").format(sql.Identifier(view)))
                cursor.execute(sql.SQL("GRANT SELECT ON mcp_safe.{} TO {}").format(sql.Identifier(view), sql.Identifier(DATABASE_ROLE)))
            # SET ROLE verifies inherited PUBLIC grants while preserving the
            # administrative connection for an atomic rollback on failure.
            cursor.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(DATABASE_ROLE)))
            cursor.execute("SET LOCAL transaction_read_only = on")
            cursor.row_factory = dict_row
            cursor.execute(ROLE_QUERY)
            role = cursor.fetchone()
            cursor.execute(ACCESS_QUERY)
            access = cursor.fetchone()
            cursor.execute("SHOW transaction_read_only")
            check_identity(role, access, cursor.fetchone()["transaction_read_only"])
            for view in VIEWS:
                cursor.execute(sql.SQL("SELECT * FROM mcp_safe.{} LIMIT 1").format(sql.Identifier(view)))
            reader_dsn = make_conninfo(host=host, port=admin.get("port", "5432"),
                                      dbname=connection.info.dbname, user=DATABASE_ROLE, password=password,
                                      sslmode=admin.get("sslmode", "prefer"))
            try:
                fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w") as stream:
                    json.dump({"dsn": reader_dsn}, stream)
                connection.commit()
                temporary.rename(output)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "deploy/mcp/secrets/database.json")
    parser.add_argument("--container-host", help="容器可连接的数据库主机，仅用于数据库工具的单一出口")
    args = parser.parse_args()
    dsn = os.environ.get("MCP_ADMIN_DATABASE_URL")
    if not dsn:
        from dotenv import dotenv_values
        dsn = dotenv_values(ROOT / "backend/.env").get("DATABASE_URL")
    if not dsn:
        parser.error("需要 MCP_ADMIN_DATABASE_URL 或 backend/.env 的数据库配置")
    try:
        provision(dsn, args.output, args.container_host)
    except (psycopg.Error, ValueError, PermissionError, OSError):
        # Exception strings can contain passwords/DSNs. Do not print them.
        print("初始化失败，未扩大现有用户权限。请确认数据库已启动、业务表已创建、管理员具有 CREATEROLE，且 PUBLIC 未授予业务表读取/用户 schema CREATE 权限。已有账号/凭据不会被覆盖。", file=sys.stderr)
        return 1
    print("MCP 专用只读账号与聚合视图已创建，凭据已保存在权限为 0600 的文件中。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
