#!/usr/bin/env python
"""Trainingsdaten-Erweiterung (Runde 2) für QLoRA.

Neue Quellen:
  1. code_search_net: java, javascript, go        (Code-QA, Chat)
  2. the-stack-smol-xs: c, go, rust                (Completion)
  3. german.stackexchange: mehr Seiten              (Deutsch/Grammatik, Chat)
  4. StackOverflow-EN (API): python/c++/linux/docker/networking (Chat)
  5. tldr-pages.de (deutsche Übersetzungen)         (Deutsch+Linux, Chat)
  6. Wikipedia-de (Streaming)                       (Completion)
  7. GitHub-Issues Homelab: n8n, paperless, immich, nextcloud (Chat)
  8. Docker-Doku: guides + manuals                  (Completion)

Ausgabe: data/train_chat2.jsonl + data/train_completion2.jsonl
"""
import json
import re
import html
import random
import time
from pathlib import Path

import requests

BASE = Path("/mnt/data_mysql/qlora")
RAW = BASE / "data" / "raw2"
RAW.mkdir(parents=True, exist_ok=True)
OUT_CHAT = BASE / "data" / "train_chat2.jsonl"
OUT_COMPLETION = BASE / "data" / "train_completion2.jsonl"
STATS = BASE / "data" / "dataset_stats2.txt"

SYSTEM = "Du beantwortest technische Fragen korrekt und ohne zu erfinden. Wenn du unsicher bist, sage es ehrlich."

def load_gh_token():
    env_path = Path.home() / ".hermes" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("GITHUB_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None

GH_TOKEN = load_gh_token()
GH_HEADERS = {"Authorization": f"token {GH_TOKEN}"} if GH_TOKEN else {}
random.seed(7)

chat_rows, completion_rows, errors = [], [], []

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

# ---------------------------------------------------------------- 1) code_search_net multi
def load_codesearchnet_multi(langs=("java", "javascript", "go"), n_each=300):
    from datasets import load_dataset
    for lang in langs:
        try:
            ds = load_dataset("code-search-net/code_search_net", lang,
                              split="train", streaming=True)
        except Exception as e:
            errors.append(f"csn/{lang}: {e}")
            continue
        n = 0
        for ex in ds:
            doc = (ex.get("func_documentation_string") or "").strip()
            code = (ex.get("func_code_string") or "").strip()
            if len(doc) < 40 or len(code) < 40 or len(code) > 2500:
                continue
            add_chat([{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": f"Implementiere die folgende Funktion in {lang.title()}:\n" + truncate(doc, 1200)},
                      {"role": "assistant", "content": truncate(code, 2500)}], f"csn-{lang}")
            n += 1
            if n >= n_each:
                break
        print(f"[1] code_search_net/{lang}: {n} QA")

# ---------------------------------------------------------------- 2) the-stack-smol-xs mehr
def load_stack_smol_more(langs=("c", "go", "rust"), n_each=200):
    from huggingface_hub import hf_hub_download
    for lang in langs:
        try:
            path = hf_hub_download("bigcode/the-stack-smol-xs", f"data/{lang}/data.json",
                                   repo_type="dataset", cache_dir=str(BASE / "hf_cache"))
            raw = Path(path).read_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = [json.loads(l) for l in raw.splitlines() if l.strip()]
        except Exception as e:
            errors.append(f"smol/{lang}: {e}")
            continue
        random.shuffle(data)
        n = 0
        for it in data:
            text = it if isinstance(it, str) else (it.get("content") or it.get("text") or "")
            if len(text) < 300 or len(text) > 4000:
                continue
            add_completion(text, f"smol-{lang}")
            n += 1
            if n >= n_each:
                break
        print(f"[2] the-stack-smol/{lang}: {n} Completion")

# ---------------------------------------------------------------- 3) german.stackexchange mehr
def load_german_se_more(n_max=250, min_score=3, max_pages=8):
    base = "https://api.stackexchange.com/2.3/questions"
    params = {"site": "german", "sort": "votes", "order": "desc",
              "pagesize": 100, "filter": "withbody"}
    qs = []
    for page in range(1, max_pages + 1):
        params["page"] = page
        r = requests.get(base, params=params, timeout=30)
        d = r.json()
        if "items" not in d:
            errors.append(f"german_se: {d.get('error_message', d)}")
            break
        qs.extend(d["items"])
        if not d.get("has_more"):
            break
        time.sleep(0.4)
    ids = [str(q["question_id"]) for q in qs if q.get("accepted_answer_id") and q.get("score", 0) >= min_score]
    n = 0
    for i in range(0, len(ids), 100):
        batch = ";".join(ids[i:i+100])
        r = requests.get(f"https://api.stackexchange.com/2.3/questions/{batch}/answers",
                         params={"site": "german", "filter": "withbody"}, timeout=30)
        d = r.json()
        ans_by_q = {a["question_id"]: a for a in d.get("items", []) if a.get("is_accepted")}
        for q in qs:
            if n >= n_max:
                break
            if q["question_id"] not in ans_by_q:
                continue
            qbody = strip_html(q.get("body", ""))
            abody = strip_html(ans_by_q[q["question_id"]].get("body", ""))
            if len(qbody) < 40 or len(abody) < 80:
                continue
            add_chat([{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": truncate(q["title"] + "\n\n" + qbody, 1500)},
                      {"role": "assistant", "content": truncate(abody, 2500)}], "german-se")
            n += 1
        time.sleep(0.4)
    print(f"[3] german.stackexchange: {n} QA")

# ---------------------------------------------------------------- 4) StackOverflow-EN (ein Tag pro Call!)
def load_stackoverflow_en(n_max=350, tags=("python", "c++", "linux", "docker", "networking")):
    base = "https://api.stackexchange.com/2.3/questions"
    qs = []
    for tag in tags:
        params = {"site": "stackoverflow", "sort": "votes", "order": "desc",
                  "pagesize": 100, "filter": "withbody", "tagged": tag}
        for page in (1, 2):
            params["page"] = page
            r = requests.get(base, params=params, timeout=30)
            d = r.json()
            if "items" not in d:
                errors.append(f"so-en/{tag}: {d.get('error_message', d)}")
                break
            qs.extend(d["items"])
            if not d.get("has_more"):
                break
            time.sleep(0.3)
        time.sleep(0.3)
    # Dedupe nach ID (mehrere Tags -> gleiche Fragen)
    seen_q = {}
    for q in qs:
        seen_q.setdefault(q["question_id"], q)
    ids = [str(q["question_id"]) for q in seen_q.values()
           if q.get("accepted_answer_id") and q.get("score", 0) >= 3]
    n = 0
    for i in range(0, len(ids), 100):
        batch = ";".join(ids[i:i+100])
        r = requests.get(f"https://api.stackexchange.com/2.3/questions/{batch}/answers",
                         params={"site": "stackoverflow", "filter": "withbody"}, timeout=30)
        d = r.json()
        ans_by_q = {a["question_id"]: a for a in d.get("items", []) if a.get("is_accepted")}
        for q in seen_q.values():
            if n >= n_max:
                break
            if q["question_id"] not in ans_by_q:
                continue
            qbody = strip_html(q.get("body", ""))
            abody = strip_html(ans_by_q[q["question_id"]].get("body", ""))
            if len(qbody) < 40 or len(abody) < 80:
                continue
            add_chat([{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": truncate(q["title"] + "\n\n" + qbody, 1500)},
                      {"role": "assistant", "content": truncate(abody, 2500)}], "so-en")
            n += 1
        time.sleep(0.3)
    print(f"[4] StackOverflow-EN: {n} QA")

# ---------------------------------------------------------------- 5) tldr-pages.de
def load_tldr_de(n_max=250):
    dirs = ["pages.de/common", "pages.de/linux"]
    got = []
    for d in dirs:
        r = requests.get(f"https://api.github.com/repos/tldr-pages/tldr/contents/{d}",
                         headers=GH_HEADERS, timeout=30)
        if r.status_code != 200:
            errors.append(f"tldr-de {d}: HTTP {r.status_code}")
            continue
        got.extend([f["name"] for f in r.json() if f["name"].endswith(".md")])
    random.shuffle(got)
    n = 0
    for name in got:
        if n >= n_max:
            break
        cmd = name[:-3]
        r = requests.get(f"https://raw.githubusercontent.com/tldr-pages/tldr/main/{dirs[0]}/{name}", timeout=30)
        if r.status_code != 200:
            r = requests.get(f"https://raw.githubusercontent.com/tldr-pages/tldr/main/{dirs[1]}/{name}", timeout=30)
        if r.status_code != 200 or len(r.text) < 60:
            continue
        add_chat([{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"Was ist der Befehl `{cmd}` und wie verwendet man ihn?"},
                  {"role": "assistant", "content": truncate(r.text, 1200)}], "tldr-de")
        n += 1
    print(f"[5] tldr-pages.de: {n} QA")

# ---------------------------------------------------------------- 6) FineWeb-2 de (deu_Latn, parquet-basiert)
def load_fineweb_de(n_max=600, min_len=800):
    from datasets import load_dataset
    try:
        ds = load_dataset("HuggingFaceFW/fineweb-2", "deu_Latn", split="train", streaming=True)
    except Exception as e:
        errors.append(f"fineweb-de: {e}")
        print(f"[6] fineweb-de: FEHLER {e}")
        return
    n = 0
    for ex in ds:
        text = (ex.get("text") or "").strip()
        if len(text) < min_len:
            continue
        add_completion("Deutscher Web-Text:\n" + truncate(text, 2500), "fineweb-de")
        n += 1
        if n >= n_max:
            break
    print(f"[6] fineweb-de: {n} Completion")

# ---------------------------------------------------------------- 7) GitHub-Issues Homelab
def load_homelab_issues(n_max=300):
    repos = ["n8n-io/n8n", "paperless-ngx/paperless-ngx", "immich-app/immich", "nextcloud/server"]
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
            if "pull_request" in iss:
                continue
            title = (iss.get("title") or "").strip()
            body = (iss.get("body") or "").strip()
            if len(title) < 10 or len(body) < 40 or iss.get("comments", 0) < 1:
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
                      {"role": "assistant", "content": truncate(top, 2500)}], f"github-{repo.split('/')[1]}")
            n += 1
            time.sleep(0.15)
    print(f"[7] GitHub-Issues Homelab: {n} QA")

# ---------------------------------------------------------------- 8) Docker-Doku mehr
def load_docker_more(n_max=40):
    dirs = ["content/guides", "content/manuals"]
    n = 0
    for d in dirs:
        r = requests.get(f"https://api.github.com/repos/docker/docs/contents/{d}",
                         headers=GH_HEADERS, timeout=30)
        if r.status_code != 200:
            errors.append(f"docker {d}: HTTP {r.status_code}")
            continue
        files = [f["name"] for f in r.json() if f["name"].endswith(".md")]
        for name in files[:20]:
            if n >= n_max:
                break
            r = requests.get(f"https://raw.githubusercontent.com/docker/docs/main/{d}/{name}", timeout=30)
            if r.status_code != 200:
                continue
            text = re.sub(r"^---.*?---", "", r.text, flags=re.S).strip()
            add_completion(f"Docker-Dokumentation:\n{truncate(text, 3500)}", "docker-docs")
            n += 1
    print(f"[8] docker-docs: {n} Completion")

# ---------------------------------------------------------------- Main
def main():
    load_codesearchnet_multi()
    load_stack_smol_more()
    load_german_se_more()
    load_stackoverflow_en()
    load_tldr_de()
    load_fineweb_de()
    load_homelab_issues()
    load_docker_more()

    # Dedupe
    seen, dedup = set(), []
    for r in chat_rows:
        k = json.dumps(r["messages"], ensure_ascii=False)
        if k not in seen:
            seen.add(k)
            dedup.append(r)
    chat_rows[:] = dedup
    seen, dedup = set(), []
    for r in completion_rows:
        k = r["text"][:200]
        if k not in seen:
            seen.add(k)
            dedup.append(r)
    completion_rows[:] = dedup

    with open(OUT_CHAT, "w") as f:
        for r in chat_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT_COMPLETION, "w") as f:
        for r in completion_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"NEU - Chat: {len(chat_rows)} | Completion: {len(completion_rows)}")
    print("Chat nach Quelle:", dict(Counter(r['meta']['quelle'] for r in chat_rows)))
    print("Completion nach Quelle:", dict(Counter(r['meta']['quelle'] for r in completion_rows)))
    if errors:
        print("Fehler:", "; ".join(errors[:8]))

if __name__ == "__main__":
    main()
