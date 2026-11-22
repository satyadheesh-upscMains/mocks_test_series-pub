"""Gemma 4 12B on the GitHub runner, and the one way every stage talks to it.

Logs show the work without the content: model, stage, check, attempt, seconds, tokens and pass/reject with a fixed
reason code. Prompts, answers and model output are never printed.
"""
import ctypes
import json
import os
import time

from .text import Rejected

MODEL_REPO = "unsloth/gemma-4-12b-it-GGUF"
MODEL_FILE = "gemma-4-12b-it-Q4_K_M.gguf"
PROJECTOR = "mmproj-F32.gguf"


def log(**meta):
    print("gemma " + json.dumps(meta, ensure_ascii=False, separators=(",", ":")), flush=True)


def load_model(n_ctx=8192, vision=False):
    from huggingface_hub import hf_hub_download
    import llama_cpp as lc

    started = time.monotonic()
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, local_dir="models")
    handler = None
    if vision:
        from llama_cpp.llama_chat_format import Gemma4ChatHandler
        projector = hf_hub_download(repo_id=MODEL_REPO, filename=PROJECTOR, local_dir="models")
        handler = Gemma4ChatHandler(clip_model_path=projector, verbose=False, use_gpu=False)
    llm = lc.Llama(model_path=path, n_ctx=n_ctx, n_threads=max(2, os.cpu_count() or 2), n_gpu_layers=0,
                   verbose=False, chat_handler=handler)
    arch = llm.metadata.get("general.architecture", "")
    if os.path.basename(llm.model_path) != MODEL_FILE or not arch.startswith("gemma"):
        raise RuntimeError("Unexpected model file")
    # Long Devanagari token pieces need a larger buffer than llama-cpp-python's default (same fix as the study kit).
    model = llm._model

    def detokenize(tokens, special=False):
        buf = ctypes.create_string_buffer(128)
        out = bytearray()
        for token in tokens:
            n = lc.llama_token_to_piece(model.vocab, lc.llama_token(token), buf, len(buf), 0, special)
            if n < 0:
                buf = ctypes.create_string_buffer(-n)
                n = lc.llama_token_to_piece(model.vocab, lc.llama_token(token), buf, len(buf), 0, special)
            out.extend(buf.raw[:max(0, n)])
        if tokens and tokens[0] == model.token_bos() and out[:1] == b" ":
            del out[:1]
        return bytes(out)

    model.detokenize = detokenize
    log(event="loaded", model=MODEL_FILE, arch=arch, vision=vision, n_ctx=n_ctx,
        threads=llm.n_threads if hasattr(llm, "n_threads") else None, seconds=round(time.monotonic() - started, 1))
    return llm


class Examiner:
    """Wraps the model: JSON-schema output, validation, one corrected retry, content-free logs."""

    def __init__(self, llm, name=MODEL_FILE):
        self.llm = llm
        self.name = name
        self.calls = []

    def ask(self, *, stage, check, system, task, schema, max_tokens, validate, attempts=2):
        reason = None
        for attempt in range(1, attempts + 1):
            user = task
            if reason:
                # Our own fixed wording plus the rule's code: tells the model what to fix without echoing content.
                user += (f"\n\nYour previous output was rejected by an automatic quality check ({reason}). "
                         "Return corrected JSON that follows every rule.")
            started = time.monotonic()
            response = self.llm.create_chat_completion(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_object", "schema": schema},
                temperature=0.0, max_tokens=max_tokens,
            )
            choice = response["choices"][0]
            usage = response.get("usage") or {}
            meta = dict(stage=stage, check=check, attempt=attempt, seconds=round(time.monotonic() - started, 1),
                        prompt_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"))
            try:
                if choice.get("finish_reason") == "length":
                    raise Rejected("truncated")
                try:
                    raw = json.loads(choice["message"]["content"])
                except (TypeError, ValueError, KeyError):
                    raise Rejected("json") from None
                value = validate(raw)
            except Rejected as e:
                reason = e.code
                self.calls.append(dict(meta, status="rejected", reason=reason))
                log(**meta, status="rejected", reason=reason)
                continue
            self.calls.append(dict(meta, status="ok"))
            log(**meta, status="ok")
            return value
        raise Rejected(reason or "failed")
