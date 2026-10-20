# Milvus 运维脚本

三个脚本配合使用，**都不调用 `milvus_service.initialize()`**——那个守卫会在缺少
`owner_id` 字段时抛错，恰好挡住运维操作。

## 1. `migrate_milvus_owner.py` —— 保留文档的迁移

存量 Collection 需要加 `owner_id` 字段时使用。因 `enable_dynamic_field=False`，
无法给存量表补字段，只能重建并重新向量化。

```bash
# 预检（不写入，只报要处理多少文档）
python scripts/migrate_milvus_owner.py --dry-run

# 正式执行
python scripts/migrate_milvus_owner.py
```

- 旧 Collection **重命名**为 `_backup_<时间戳>`（不删除，可回滚）
- 按 PostgreSQL 的 `document_parse.text` 重新切块并向量化回填
- **会重新调用 Embedding**，1734 个 Chunk 量级会产生可观 Token 消耗

## 2. `rebuild_milvus_collection.py` —— 不保留文档的重建

存量文档无保留价值时用这个，**零 Token 成本**。

```bash
python scripts/rebuild_milvus_collection.py         # 预演
python scripts/rebuild_milvus_collection.py --yes   # 确认执行
```

- 备份旧 Collection 后重建**空** Collection（含 `owner_id` + INVERTED 标量索引）
- 建完**回读校验**字段与索引确实存在
- 原文档仍在 PostgreSQL，需要时重新上传即可

## 3. `cleanup_milvus_backups.py` —— 清理备份

```bash
python scripts/cleanup_milvus_backups.py         # 预演，列出将删除什么
python scripts/cleanup_milvus_backups.py --yes   # 确认删除
```

**只删**名字形如 `<collection>_backup_<14位时间戳>` 的备份，业务 Collection
和其他非备份 Collection 一律不动，并会在同名前二次确认。

---

## 用哪个

| 情况 | 用哪个 |
|------|--------|
| 文档要留着 | `migrate_milvus_owner.py` |
| 文档不要了（测试数据） | `rebuild_milvus_collection.py` |
| 迁移后确认没问题 | `cleanup_milvus_backups.py` |

## 为什么这些脚本不走 `initialize()`

`milvus_service.initialize()` 有一道保护：Collection 缺 `owner_id` 字段时
**拒绝启动**，避免在无隔离的索引上对外提供检索。但迁移脚本要处理的正是
"缺 `owner_id`" 这个状态——直接调用会形成死锁：

> 没有 `owner_id` 就跑不了迁移，不跑迁移就没有 `owner_id`

所以三个脚本都直连 `milvus_service.client`，绕过守卫。这套约束由
`tests/test_milvus_migration_safety.py` 固化。
