#!/usr/bin/env python
"""Konvertiert train_chat.jsonl (messages-Format) -> train_sharegpt.jsonl (Axolotl 0.4.x sharegpt-Format)."""
import json
from pathlib import Path

BASE = Path("/mnt/data_mysql/qlora")
SRC = BASE / "data" / "train_chat.jsonl"
DST = BASE / "data" / "train_sharegpt.jsonl"

ROLE_MAP = {"system": "system", "user": "human", "assistant": "gpt"}

n = 0
with open(SRC) as f, open(DST, "w") as out:
    for line in f:
        d = json.loads(line)
        msgs = d.get("messages", [])
        convs = [{"from": ROLE_MAP.get(m["role"], m["role"]), "value": m["content"]}
                 for m in msgs if m.get("content", "").strip()]
        if not convs:
            continue
        out.write(json.dumps({"conversations": convs}, ensure_ascii=False) + "\n")
        n += 1
print(f"konvertiert: {n} Samples -> {DST}")
