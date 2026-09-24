"""Verify effective permissions, including grants inherited through PUBLIC."""
from policy import DATABASE_ROLE

ROLE_QUERY = """
SELECT rolname, rolsuper, rolcreaterole, rolcreatedb, rolreplication, rolbypassrls
FROM pg_roles WHERE rolname = current_user
"""
ACCESS_QUERY = """
SELECT EXISTS (
  SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
  WHERE n.nspname !~ '^pg_' AND n.nspname <> 'information_schema'
    AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
    AND NOT (n.nspname = 'mcp_safe' AND c.relname IN ('knowledge_inventory', 'evaluation_summary') AND c.relkind = 'v')
    AND (has_table_privilege(current_user, c.oid, 'SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER')
         OR has_any_column_privilege(current_user, c.oid, 'SELECT,INSERT,UPDATE,REFERENCES'))
) AS extra_access,
EXISTS (SELECT 1 FROM pg_auth_members m JOIN pg_roles r ON r.oid = m.member WHERE r.rolname = current_user) AS membership,
EXISTS (SELECT 1 FROM pg_namespace WHERE nspname !~ '^pg_' AND has_schema_privilege(current_user, oid, 'CREATE')) AS schema_create,
has_database_privilege(current_user, current_database(), 'CREATE') AS database_create
"""


def check_identity(role, access, readonly):
    if role["rolname"] != DATABASE_ROLE or any(role[k] for k in ("rolsuper", "rolcreaterole", "rolcreatedb", "rolreplication", "rolbypassrls")):
        raise PermissionError("MCP 数据库账号权限过高或身份不正确")
    if any(access.values()) or readonly != "on":
        raise PermissionError("MCP 账号存在视图外权限、角色继承、建表权限或事务不是只读")


async def verify_connection(connection):
    async with connection.cursor() as cursor:
        await cursor.execute(ROLE_QUERY)
        role = await cursor.fetchone()
        await cursor.execute(ACCESS_QUERY)
        access = await cursor.fetchone()
        await cursor.execute("SHOW transaction_read_only")
        readonly = (await cursor.fetchone())["transaction_read_only"]
    check_identity(role, access, readonly)
