from langchain_core.documents import Document


def chunk_post(post: dict) -> list[Document]:
    title = post.get("title", "")
    body = post.get("body", "")
    full_text = f"{title}\n\n{body}".strip()
    metadata = {
        "id": post.get("id", ""),
        "type": "post",
        "author": post.get("author", ""),
        "source": post.get("subreddit_or_source", ""),
        "url": post.get("url", ""),
        "created_utc": post.get("created_utc", 0),
        "time_window": post.get("time_window", ""),
        "sentiment": post.get("sentiment_label", "neutral"),
    }

    if len(full_text) <= 800:
        return [Document(page_content=full_text, metadata={**metadata, "chunk_index": 0})]

    paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    idx = 0
    for para in paragraphs:
        if len(current) + len(para) < 600:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(Document(page_content=current, metadata={**metadata, "chunk_index": idx}))
                idx += 1
            current = para
    if current:
        chunks.append(Document(page_content=current, metadata={**metadata, "chunk_index": idx}))
    return chunks


def chunk_comment(comment: dict) -> Document:
    body = comment.get("body", "").strip()
    return Document(
        page_content=body,
        metadata={
            "id": comment.get("id", ""),
            "type": "comment",
            "author": comment.get("author", ""),
            "source": comment.get("subreddit_or_source", ""),
            "url": comment.get("url", ""),
            "created_utc": comment.get("created_utc", 0),
            "time_window": comment.get("time_window", ""),
            "sentiment": comment.get("sentiment_label", "neutral"),
            "chunk_index": 0,
        },
    )
