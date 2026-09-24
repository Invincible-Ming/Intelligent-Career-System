"""Bound query/document pairs by the actual CrossEncoder tokenizer budget."""


def make_rerank_pairs(tokenizer, query, contents, max_length):
    special = tokenizer.num_special_tokens_to_add(pair=True)
    if max_length < special + 8:
        raise ValueError("Reranker token budget is too small")
    query_ids = tokenizer.encode(query, add_special_tokens=False)
    query_ids = query_ids[: min(96, (max_length - special) // 3)]
    query_text = tokenizer.decode(query_ids, skip_special_tokens=True)
    budget = max_length - len(query_ids) - special - 4
    pairs, owners = [], []
    for owner, content in enumerate(contents):
        ids = tokenizer.encode(content, add_special_tokens=False)
        start = 0
        while True:
            end = min(len(ids), start + budget)
            text = tokenizer.decode(ids[start:end], skip_special_tokens=True)
            # Decoding/re-encoding can change token counts: verify the actual pair.
            while end > start + 1 and len(tokenizer.encode(query_text, text, add_special_tokens=True)) > max_length:
                end -= 1
                text = tokenizer.decode(ids[start:end], skip_special_tokens=True)
            pairs.append([query_text, text])
            owners.append(owner)
            if end >= len(ids):
                break
            start = max(start + 1, end - min(32, budget // 4))
    return pairs, owners
