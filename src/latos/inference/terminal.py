"""Streaming terminal interface; never opens a listener or writes model artifacts."""

import json
import signal
import sys
from pathlib import Path

import torch

from latos.chat import format_chat
from latos.doctor import available_backends, select_device
from latos.inference.stream import stream_generate
from latos.model.storage import bind_tokenizer, file_hash, load_model

SFT_SHA256 = "62b890214e5544292cc54b725a4e505730fca0d501b3337cdd3f051a622d09ae"
QUALITY_NOTE = (
    "Experimental local inference. Useful instruction following is not established. "
    "Phase 6 SFT scored 0/32 exact replies and worsened English loss; it is not an improved base."
)


class ChatSession:
    """Commit only nonempty EOS-completed turns; rejection/cancellation is transactional."""

    def __init__(self, model, codec, *, system=None, context_limit=None, **sampling):
        self.model, self.codec = model, codec
        self.context_limit = model.config.context_length if context_limit is None else context_limit
        if type(self.context_limit) is not int or not 2 <= self.context_limit <= min(
            4096, model.config.context_length
        ):
            raise ValueError("Chat context limit must fit the model capacity")
        self.system = system
        self.sampling = sampling
        self.reset()
        # Validate the optional system turn without changing the shared formatter.
        format_chat(
            codec,
            self.messages + [{"role": "user", "content": "Hello"}],
            max_length=4096,
            generation_prompt=True,
        )

    def reset(self):
        self.messages = [] if self.system is None else [{"role": "system", "content": self.system}]

    def reply(self, content, *, cancelled=None):
        candidate = self.messages + [{"role": "user", "content": content}]
        prompt, _ = format_chat(
            self.codec, candidate, max_length=self.context_limit, generation_prompt=True
        )
        budget = self.sampling.get("max_new_tokens", 32)
        if type(budget) is not int or not 1 <= budget <= self.context_limit:
            raise ValueError("Chat requires a positive token budget within its context limit")
        # Reserve the entire requested reply budget, including EOS. No partial messages
        # are silently discarded to make room, and no turn starts with zero capacity.
        if len(prompt) + budget > self.context_limit:
            raise ValueError(
                "Chat plus reply budget exceeds context limit; use /reset or less text"
            )
        for event in stream_generate(
            self.model, self.codec, list(prompt), cancelled=cancelled, **self.sampling
        ):
            if event["event"] == "done":
                committed = False
                if event["stop_reason"] == "eos" and event["text"].strip():
                    completed = candidate + [{"role": "assistant", "content": event["text"]}]
                    # History is stored as displayed text, then encoded in exact Phase 6
                    # segments next turn. Arbitrary sampled BPE IDs need not be canonical.
                    try:
                        format_chat(self.codec, completed, max_length=self.context_limit)
                    except ValueError:
                        pass
                    else:
                        self.messages = completed
                        committed = True
                event = {**event, "history_committed": committed}
            yield event


def terminal_text(text):
    """Keep generated terminal control characters inert; JSON preserves exact text."""
    return "".join(
        ch if ch in "\n\t" or (ord(ch) >= 32 and not 127 <= ord(ch) <= 159) else f"\\x{ord(ch):02x}"
        for ch in text
    )


def run_chat(args):
    if not 1 <= args.threads <= 64:
        raise ValueError("threads must be between 1 and 64")
    if args.json and args.prompt is None:
        raise ValueError("--json requires --prompt")
    torch.set_num_threads(args.threads)
    model = load_model(args.model_dir)
    codec = bind_tokenizer(model.config, args.tokenizer_dir)
    model.to(select_device(args.device, available_backends()))
    identity = {
        "event": "model",
        "model_dir": str(Path(args.model_dir).resolve()),
        "weights_sha256": file_hash(args.model_dir / "model.safetensors"),
        "tokenizer_sha256": model.config.tokenizer_sha256,
        "device": str(model.embedding.weight.device),
        "quality_note": QUALITY_NOTE,
    }
    identity["artifact"] = (
        "Phase 6 experimental SFT"
        if identity["weights_sha256"] == SFT_SHA256
        else "Explicitly selected checkpoint; see its provenance"
    )
    session = ChatSession(
        model,
        codec,
        system=args.system,
        context_limit=args.context_limit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        cached=not args.no_cache,
    )
    if args.json:
        print(json.dumps(identity), flush=True)
    else:
        print(
            f"{identity['artifact']}\nModel: {identity['model_dir']}\n{QUALITY_NOTE}",
            file=sys.stderr,
        )
        if args.prompt is None:
            print(
                "/reset clears history; /exit quits. Ctrl-C cancels a reply. EOF exits.",
                file=sys.stderr,
            )

    def respond(prompt):
        interrupted = False

        def cancel_reply(signum, frame):
            nonlocal interrupted
            interrupted = True

        previous = signal.signal(signal.SIGINT, cancel_reply)
        try:
            render_reply(prompt, lambda: interrupted)
        finally:
            signal.signal(signal.SIGINT, previous)

    def render_reply(prompt, cancelled):
        for event in session.reply(prompt, cancelled=cancelled):
            if args.json:
                print(json.dumps(event, ensure_ascii=True), flush=True)
            elif event["event"] != "done":
                print(terminal_text(event["text"]), end="", flush=True)
            else:
                print(flush=True)
                print(
                    f"[{event['stop_reason']}; {len(event['generated_ids'])} tokens; "
                    f"history {'saved' if event['history_committed'] else 'unchanged'}]",
                    file=sys.stderr,
                    flush=True,
                )

    if args.prompt is not None:
        respond(args.prompt)
        return 0
    while True:
        try:
            prompt = input("You> ")
        except EOFError, KeyboardInterrupt:
            print()
            return 0
        if prompt == "/exit":
            return 0
        if prompt == "/reset":
            session.reset()
            print("History cleared.")
            continue
        try:
            respond(prompt)
        except (ValueError, RuntimeError) as exc:
            print(f"Chat error: {exc}", file=sys.stderr)
        except KeyboardInterrupt:
            print("\nReply cancelled; history unchanged.", file=sys.stderr)
