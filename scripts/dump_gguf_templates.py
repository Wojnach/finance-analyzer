import json, sys
from gguf import GGUFReader

MODELS = {
    "phi4_mini": r"Q:\models\phi4-mini-reasoning-gguf\microsoft_Phi-4-mini-reasoning-Q4_K_M.gguf",
    "phi4_instruct": r"Q:\models\phi4-mini-instruct-gguf\microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
    "qwen3": r"Q:\models\qwen3-8b-gguf\Qwen3-8B-Q4_K_M.gguf",
    "fin_r1": r"Q:\models\fin-r1-gguf\Fin-R1-q5_k_m.gguf",
    "ministral3": r"Q:\models\ministral-3-8b-gguf\Ministral-3-8B-Instruct-2512-Q5_K_M.gguf",
    "ministral8_lora_base": r"Q:\models\ministral-8b-gguf\Ministral-8B-Instruct-2410-Q4_K_M.gguf",
    "finance-llama-8b": r"Q:\models\finance-llama-8b-gguf\wiroai-finance-llama-8b-q4_k_m.gguf",
}

FIELDS = ["tokenizer.chat_template", "tokenizer.ggml.bos_token_id",
          "tokenizer.ggml.eos_token_id", "general.architecture", "general.name"]

out = {}
for name, path in MODELS.items():
    entry = {}
    try:
        r = GGUFReader(path)
        for f in FIELDS:
            fld = r.fields.get(f)
            if fld is None:
                entry[f] = None
                continue
            try:
                v = fld.contents()
            except Exception:
                parts = fld.parts[fld.data[0]]
                v = bytes(parts).decode("utf-8", "replace") if hasattr(parts, "__iter__") else str(parts)
            entry[f] = v if not isinstance(v, (bytes, bytearray)) else v.decode("utf-8", "replace")
    except Exception as e:
        entry["error"] = repr(e)
    out[name] = entry

json.dump(out, open(r"Q:\finance-analyzer\data\gguf_templates.json", "w"), indent=1)
print("dumped", len(out))
