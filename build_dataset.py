#!/usr/bin/env python
"""QLoRA-Trainingsdaten-Builder für Qwen2.5-3B (Axolotl).

Quellen:
  1. llm-benchmark sft_daten.jsonl      (84 QA, Ground Truth)
  2. code-search-net (python)           (docstring -> code, QA)
  3. the-stack-smol (python, c++)       (Code-Completion)
  4. german.stackexchange (API)         (Deutsch/Grammatik QA)
  5. tldr-pages (GitHub)                (Linux-Befehle QA)
  6. docker/docs (GitHub)               (Docker-Doku, Completion)
  7. RFCs (rfc-editor.org)              (Netzwerk-Doku, Completion)
  8. GitHub-Issues (API)                (Titel+Body -> Top-Kommentar)
  9. reciTAL/mlsum (de)                 (News-Zusammenfassung QA)

Ausgabe:
  data/train_chat.jsonl       -> {"messages": [...]}  (ChatML)
  data/train_completion.jsonl -> {"text": "..."}      (Completion)
  data/dataset_stats.txt
"""
import json
import os
import re
import html
import random
import time
from pathlib import Path

import requests

BASE = Path("/mnt/data_mysql/qlora")
RAW = BASE / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
OUT_CHAT = BASE / "data" / "train_chat.jsonl"
OUT_COMPLETION = BASE / "data" / "train_completion.jsonl"
STATS = BASE / "data" / "dataset_stats.txt"

SYSTEM = "Du beantwortest technische Fragen korrekt und ohne zu erfinden. Wenn du unsicher bist, sage es ehrlich."

# GitHub-Token aus ~/.hermes/.env lesen (Memory: .env-Lesen erlaubt)
def load_gh_token():
    env_path = Path.home() / ".hermes" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("GITHUB_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None

GH_TOKEN = load_gh_token()
GH_HEADERS = {"Authorization": f"token {GH_TOKEN}"} if GH_TOKEN else {}
random.seed(42)

chat_rows, completion_rows = [], []
errors = []

def add_chat(messages, source):
    if messages and all(m.get("content", "").strip() for m in messages):
        chat_rows.append({"messages": messages, "meta": {"quelle": source}})

def add_completion(text, source):
    t = text.strip()
    if len(t) > 200:
        completion_rows.append({"text": t, "meta": {"quelle": source}})

def strip_html(s):
    s = re.sub(r"<pre[^>]*>.*?</pre>", " ", s, flags=re.S)
    s = re.sub(r"<[^>]+>", " ", s)
    return html.unescape(re.sub(r"\s+", " ", s)).strip()

def truncate(s, n=3500):
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " …"

# ---------------------------------------------------------------- 1) Benchmark
def load_benchmark():
    p = Path.home() / "llm-benchmark" / "llm_test" / "sft_daten.jsonl"
    n = 0
    for line in p.read_text().splitlines():
        d = json.loads(line)
        msgs = d.get("messages", [])
        if msgs and not any(m["role"] == "system" for m in msgs):
            msgs = [{"role": "system", "content": SYSTEM}] + msgs
        add_chat(msgs, "benchmark")
        n += 1
    print(f"[1] benchmark: {n} gelesen")

# ---------------------------------------------------------------- 2) code_search_net
def load_codesearchnet(n_max=800):
    from datasets import load_dataset
    ds = load_dataset("code-search-net/code_search_net", "python",
                      split="train", streaming=True)
    n = 0
    for ex in ds:
        doc = (ex.get("func_documentation_string") or "").strip()
        code = (ex.get("func_code_string") or "").strip()
        if len(doc) < 40 or len(code) < 40 or len(code) > 2500:
            continue
        # docstring kann mehrzeilig/kommentiert sein -> vereinfachen
        q = "Implementiere die folgende Funktion in Python:\n" + truncate(doc, 1200)
        add_chat([{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": q},
                  {"role": "assistant", "content": truncate(code, 2500)}], "codesearchnet")
        n += 1
        if n >= n_max:
            break
    print(f"[2] code_search_net: {n} QA-Paare")

# ---------------------------------------------------------------- 3) the-stack-smol-xs (nicht gated)
def load_stack_smol(n_py=700, n_cpp=600, n_docker=200):
    from huggingface_hub import hf_hub_download
    for lang, n_max in (("python", n_py), ("c++", n_cpp), ("dockerfile", n_docker)):
        try:
            path = hf_hub_download("bigcode/the-stack-smol-xs",
                                   f"data/{lang}/data.json",
                                   repo_type="dataset",
                                   cache_dir=str(BASE / "hf_cache"))
            raw = Path(path).read_text()
            try:
                data = json.loads(raw)  # manche Sprach-Subsets: ein JSON-Objekt
            except json.JSONDecodeError:
                data = [json.loads(l) for l in raw.splitlines() if l.strip()]  # meist: JSONL
        except Exception as e:
            errors.append(f"the-stack-smol/{lang}: {e}")
            print(f"[3] the-stack-smol/{lang}: FEHLER {e}")
            continue
        # Format unbekannt: Liste von Objekten oder dict
        if isinstance(data, dict):
            items = data.get("data", data.get("content", []))
        else:
            items = data
        random.shuffle(items)
        n = 0
        for it in items:
            if isinstance(it, str):
                text = it
            elif isinstance(it, dict):
                text = it.get("content") or it.get("text") or ""
            else:
                continue
            if len(text) < 300 or len(text) > 4000:
                continue
            add_completion(text, f"stack-smol-{lang}")
            n += 1
            if n >= n_max:
                break
        print(f"[3] the-stack-smol/{lang}: {n} Completion-Samples")

# ---------------------------------------------------------------- 4) german.stackexchange
def load_german_se(n_max=300, min_score=4):
    base = "https://api.stackexchange.com/2.3/questions"
    params = {"site": "german", "sort": "votes", "order": "desc",
              "pagesize": 100, "filter": "withbody"}
    page = 1
    qs = []
    while page <= 4:
        params["page"] = page
        r = requests.get(base, params=params, timeout=30)
        d = r.json()
        if "items" not in d:
            errors.append(f"german_se: {d.get('error_message', d)}")
            break
        qs.extend(d["items"])
        if not d.get("has_more"):
            break
        page += 1
        time.sleep(0.4)
    # Fragen mit akzeptierter Antwort -> Antworten gebündelt laden
    ids = [str(q["question_id"]) for q in qs if q.get("accepted_answer_id") and q.get("score", 0) >= min_score]
    n = 0
    for i in range(0, len(ids), 100):
        batch = ";".join(ids[i:i+100])
        r = requests.get(f"https://api.stackexchange.com/2.3/questions/{batch}/answers",
                         params={"site": "german", "filter": "withbody"}, timeout=30)
        d = r.json()
        ans_by_q = {}
        for a in d.get("items", []):
            if a.get("is_accepted"):
                ans_by_q[a["question_id"]] = a
        for q in qs:
            if n >= n_max:
                break
            if q["question_id"] not in ans_by_q:
                continue
            ans = ans_by_q[q["question_id"]]
            qbody = strip_html(q.get("body", ""))
            abody = strip_html(ans.get("body", ""))
            if len(qbody) < 40 or len(abody) < 80:
                continue
            add_chat([{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": truncate(q["title"] + "\n\n" + qbody, 1500)},
                      {"role": "assistant", "content": truncate(abody, 2500)}], "german-se")
            n += 1
        time.sleep(0.4)
    print(f"[4] german.stackexchange: {n} QA-Paare")

# ---------------------------------------------------------------- 5) tldr-pages
def load_tldr(n_max=300):
    dirs = ["pages/common", "pages/linux"]
    got = []
    for d in dirs:
        r = requests.get(f"https://api.github.com/repos/tldr-pages/tldr/contents/{d}",
                         headers=GH_HEADERS, timeout=30)
        if r.status_code != 200:
            errors.append(f"tldr {d}: HTTP {r.status_code}")
            continue
        got.extend([f["name"] for f in r.json() if f["name"].endswith(".md")])
    random.shuffle(got)
    n = 0
    for name in got:
        if n >= n_max:
            break
        cmd = name[:-3]
        r = requests.get(f"https://raw.githubusercontent.com/tldr-pages/tldr/main/pages/common/{name}",
                         timeout=30)
        if r.status_code != 200:
            r = requests.get(f"https://raw.githubusercontent.com/tldr-pages/tldr/main/pages/linux/{name}",
                             timeout=30)
        if r.status_code != 200 or len(r.text) < 80:
            continue
        add_chat([{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"Was ist der Befehl `{cmd}` und wie verwendet man ihn?"},
                  {"role": "assistant", "content": truncate(r.text, 1200)}], "tldr")
        n += 1
    print(f"[5] tldr-pages: {n} QA-Paare")

# ---------------------------------------------------------------- 6) docker/docs
def load_docker_docs(n_max=40):
    # docker/docs Repo: content/get-started/*.md (ohne /en!)
    r = requests.get("https://api.github.com/repos/docker/docs/contents/content/get-started",
                     headers=GH_HEADERS, timeout=30)
    if r.status_code != 200:
        errors.append(f"docker-docs: HTTP {r.status_code}")
        print(f"[6] docker-docs: FEHLER HTTP {r.status_code}")
        return
    files = [f["name"] for f in r.json() if f["name"].endswith(".md")]
    n = 0
    for name in files[:n_max]:
        r = requests.get(f"https://raw.githubusercontent.com/docker/docs/main/content/get-started/{name}",
                         timeout=30)
        if r.status_code != 200:
            continue
        text = re.sub(r"^---.*?---", "", r.text, flags=re.S).strip()
        add_completion(f"Docker-Dokumentation:\n{truncate(text, 3500)}", "docker-docs")
        n += 1
    print(f"[6] docker-docs: {n} Completion-Samples")

# ---------------------------------------------------------------- 7) RFCs (Netzwerk-Doku)
def load_rfcs():
    rfcs = {"rfc791.txt": "IP", "rfc793.txt": "TCP", "rfc768.txt": "UDP",
            "rfc1035.txt": "DNS", "rfc2131.txt": "DHCP", "rfc9293.txt": "TCPv2",
            "rfc2616.txt": "HTTP/1.1", "rfc8446.txt": "TLS1.3"}
    n = 0
    for fn, label in rfcs.items():
        try:
            r = requests.get(f"https://www.rfc-editor.org/rfc/{fn}", timeout=60)
            if r.status_code != 200:
                errors.append(f"rfc {fn}: HTTP {r.status_code}")
                continue
            # Kopfzeilen abschneiden, nur Kern
            text = r.text[r.text.find("Abstract"):]
            text = re.sub(r"\n{3,}", "\n\n", text)
            for chunk in [text[i:i+3500] for i in range(0, min(len(text), 70000), 3500)]:
                add_completion(f"RFC-Dokumentation ({label}):\n{chunk}", f"rfc-{label}")
                n += 1
        except Exception as e:
            errors.append(f"rfc {fn}: {e}")
    print(f"[7] RFCs: {n} Completion-Chunks")

# ---------------------------------------------------------------- 8) GitHub-Issues
def load_github_issues(n_max=400):
    repos = ["docker/cli", "docker/compose", "moby/moby", "ollama/ollama"]
    n = 0
    for repo in repos:
        if n >= n_max:
            break
        r = requests.get(f"https://api.github.com/repos/{repo}/issues",
                         headers=GH_HEADERS,
                         params={"state": "closed", "per_page": 100,
                                 "sort": "comments", "direction": "desc"},
                         timeout=30)
        if r.status_code != 200:
            errors.append(f"issues {repo}: HTTP {r.status_code}")
            continue
        for iss in r.json():
            if n >= n_max:
                break
            if "pull_request" in iss:  # keine PRs
                continue
            title = (iss.get("title") or "").strip()
            body = (iss.get("body") or "").strip()
            if len(title) < 10 or len(body) < 40:
                continue
            comments = iss.get("comments", 0)
            if comments < 1:
                continue
            cr = requests.get(iss["comments_url"], headers=GH_HEADERS,
                              params={"per_page": 1}, timeout=30)
            if cr.status_code != 200 or not cr.json():
                continue
            top = cr.json()[0].get("body") or ""
            if len(top) < 60:
                continue
            add_chat([{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": truncate(f"{title}\n\n{body}", 1800)},
                      {"role": "assistant", "content": truncate(top, 2500)}],
                     f"github-{repo}")
            n += 1
            time.sleep(0.2)
    print(f"[8] GitHub-Issues: {n} QA-Paare")

# ---------------------------------------------------------------- 9) Second Brain Wiki (deutsch, fachlich)
def load_wiki(n_max=60):
    """Second Brain + homelab-doku als deutsche Fachtexte (Completion)."""
    import glob
    files = []
    for pat in ["/home/reclaimer/wiki/*.md", "/home/reclaimer/wiki/entities/*.md",
                "/home/reclaimer/wiki/concepts/*.md", "/home/reclaimer/wiki/comparisons/*.md",
                "/home/reclaimer/homelab-doku/*.md"]:
        files.extend(glob.glob(pat))
    n = 0
    for p in sorted(set(files)):
        if n >= n_max:
            break
        try:
            text = Path(p).read_text()
        except Exception:
            continue
        # Frontmatter raus, Code-Bloecke behalten, nur sinnvolle Laenge
        text = re.sub(r"^---.*?---", "", text, flags=re.S).strip()
        if len(text) < 400:
            continue
        add_completion(f"Deutsche IT-Dokumentation:\n{truncate(text, 3500)}", "wiki-de")
        n += 1
    print(f"[9] wiki-de: {n} Completion-Samples")

# ---------------------------------------------------------------- Main
def main():
    random.shuffle  # noqa (seed gesetzt)
    load_benchmark()
    load_codesearchnet()
    load_stack_smol()
    load_german_se()
    load_tldr()
    load_docker_docs()
    load_rfcs()
    load_github_issues()
    load_wiki()

    # Dedupe
    seen_chat, seen_comp = set(), []
    for r in chat_rows:
        k = json.dumps(r["messages"], ensure_ascii=False)
        if k not in seen_chat:
            seen_chat.add(k)
            seen_comp.append(r)
    chat_rows[:] = seen_comp
    seen = set()
    comp_dedup = []
    for r in completion_rows:
        k = r["text"][:200]
        if k not in seen:
            seen.add(k)
            comp_dedup.append(r)
    completion_rows[:] = comp_dedup

    with open(OUT_CHAT, "w") as f:
        for r in chat_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT_COMPLETION, "w") as f:
        for r in completion_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = []
    stats.append(f"Chat-Samples: {len(chat_rows)}")
    stats.append(f"Completion-Samples: {len(completion_rows)}")
    from collections import Counter
    stats.append("Chat nach Quelle: " + str(dict(Counter(r["meta"]["quelle"] for r in chat_rows))))
    stats.append("Completion nach Quelle: " + str(dict(Counter(r["meta"]["quelle"] for r in completion_rows))))
    if errors:
        stats.append("Fehler: " + "; ".join(errors[:10]))
    Path(STATS).write_text("\n".join(stats) + "\n")
    print("\n".join(stats))

if __name__ == "__main__":
    main()
