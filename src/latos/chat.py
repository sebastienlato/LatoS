"""One versioned token serialization for SFT and held-out assistant continuations."""

from latos.tokenization import LatoTokenizer

CHAT_CONTRACT = {
    "version": 1,
    "start": "BOS (ID 1), once per conversation",
    "header": "encode newline + title-cased role + colon + newline as a separate segment",
    "body": "encode content as a separate segment, then explicit EOS (ID 2) per turn",
    "targets": "assistant content and its EOS only; headers and all other turns masked",
    "prompt": "same segments through last user EOS, then assistant header; no final EOS",
    "overflow": "reject entire conversation or prompt; never truncate or split",
    "roles": "optional initial system, then alternating user and assistant",
    "special_tokens": "unchanged vocabulary; literal control spellings encode as ordinary text",
}


def format_chat(
    codec: LatoTokenizer,
    messages: list[dict],
    *,
    max_length: int,
    generation_prompt: bool = False,
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    """Return unshifted IDs and target eligibility; model shifts labels exactly once."""
    if type(max_length) is not int or not 2 <= max_length <= 4096:
        raise ValueError("Invalid chat context limit")
    if type(generation_prompt) is not bool or not isinstance(messages, list) or not messages:
        raise ValueError("Chat requires a nonempty message list")
    ids, mask = [1], [False]
    expected = "user"
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("Messages require exactly role and content")
        role, content = message["role"], message["content"]
        if not isinstance(content, str) or not content.strip() or len(content) > 100_000:
            raise ValueError("Message content must be nonempty, bounded text")
        if role == "system" and index == 0:
            pass
        elif role == expected:
            expected = "assistant" if role == "user" else "user"
        else:
            raise ValueError("Invalid role order")
        header = codec.encode(f"\n{role.title()}:\n")
        body = codec.encode(content) + [2]
        ids.extend(header + body)
        mask.extend([False] * len(header) + [role == "assistant"] * len(body))
    if generation_prompt:
        if messages[-1]["role"] != "user":
            raise ValueError("Generation prompt must end with user")
        header = codec.encode("\nAssistant:\n")
        ids.extend(header)
        mask.extend([False] * len(header))
    elif messages[-1]["role"] != "assistant" or not any(mask):
        raise ValueError("Training conversation needs a final assistant with usable targets")
    if len(ids) > max_length:
        raise ValueError("Chat exceeds context limit; truncation is not supported")
    return tuple(ids), tuple(mask)
