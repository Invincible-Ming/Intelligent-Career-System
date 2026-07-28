from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="BAAI/bge-reranker-v2-m3",
    local_dir="../models/bge-reranker-v2-m3",
)

print("模型下载完成：", path)