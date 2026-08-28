#!/usr/bin/env python
"""Trainingsdaten Runde 3 - gezielt für Neo 1.1: Docker, Python, GitHub.

Quellen:
  1. Docker-Doku: reference/commandline (CLI-Referenz), compose, engine, network, storage, build
  2. Docker GitHub-Issues: docker/compose, docker/docs (mehr)
  3. StackOverflow docker-Tag (3 Seiten, sort=votes, accepted)
  4. code_search_net python: Filter try/except + mehr Samples (Fehlerbehandlung)
  5. StackOverflow python-Tag (3 Seiten, sort=votes, accepted)
  6. GitHub-Issues Python-Projekte: flask, requests, django (Bugs)
  7. GitHub-Workflows: github/docs, cli/cli Issues + Doku
"""
import json, os, re, time, sys
from pathlib import Path
import requests

OUT = Path("/mnt/data_mysql/qlora/data")
chat = []
completion = []
errors = []

# GitHub-Token aus ~/.hermes/.env lesen (Rate-Limit 60/h -> 5000/h)
GH_TOKEN = None
env_path = Path(os.path.expanduser("~/.hermes/.env"))
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if line.startswith("GH_TOKEN=") or line.startswith("GITHUB_TOKEN="):
            GH_TOKEN = line.split("=", 1)[1].strip().strip('"').strip("'")
GH_HEADERS = {"Authorization": f"token {GH_TOKEN}"} if GH_TOKEN else {}
if GH_TOKEN:
    print("GitHub-Token gefunden (Rate-Limit hoch)")
else:
    print("KEIN GitHub-Token - nur 60 req/h")

def clean(t):
    return re.sub(r"\s+", " ", t).strip()

def add_chat(src, q, a):
    if not q or not a or len(q) < 40 or len(a) < 40:
        return
    chat.append({"system": "Du beantwortest technische Fragen korrekt, präzise und ohne zu erfinden.",
                 "user": q, "assistant": a, "quelle": src})

def add_completion(src, text):
    if text and len(text) > 300:
        completion.append({"text": text, "quelle": src})

def truncate(t, n=3500):
    return t[:n]

# ---------------------------------------------------------------- 1) Docker-Doku mehr
def load_docker_docs():
    base = "https://raw.githubusercontent.com/docker/docs/main"
    dirs = ["content/reference/cli", "content/reference/compose-file",
            "content/reference/api", "content/manuals/build", "content/manuals/compose",
            "content/manuals/engine"]
    for d in dirs:
        url = f"https://api.github.com/repos/docker/docs/contents/{d}"
        r = requests.get(url, headers=GH_HEADERS, timeout=30)
        if r.status_code != 200:
            errors.append(f"docker-docs/{d}: HTTP {r.status_code}")
            continue
        for item in r.json():
            if not item.get("name", "").endswith(".md"):
                continue
            raw = requests.get(item["download_url"], timeout=30)
            if raw.status_code != 200:
                continue
            text = clean(raw.text)
            # Nur wenn substanzieller Inhalt (nicht nur Index)
            if len(text) > 1500:
                add_completion(f"docker-docs/{item['name']}", truncate(text))
    print(f"[1] docker-docs: {len(completion)} Completion (kumulativ)")

# ---------------------------------------------------------------- 2) Docker GitHub-Issues
def load_docker_issues():
    repos = ["docker/compose", "docker/docs"]
    for repo in repos:
        for state in ("open", "closed"):
            url = ("https://api.github.com/repos/%s/issues"
                   "?state=%s&per_page=100&sort=comments&direction=desc") % (repo, state)
            r = requests.get(url, headers=GH_HEADERS, timeout=30)
            if r.status_code != 200:
                errors.append(f"github-{repo}: HTTP {r.status_code}")
                continue
            for iss in r.json():
                if "pull_request" in iss:
                    continue
                title = clean(iss.get("title", ""))
                body = clean(iss.get("body") or "")
                if len(body) < 100:
                    continue
                # Kommentare als "Lösung" holen (erster Kommentar mit Code)
                c_url = iss["comments_url"]
                cr = requests.get(c_url + "?per_page=5", headers=GH_HEADERS, timeout=30)
                answers = []
                if cr.status_code == 200:
                    for c in cr.json():
                        b = clean(c.get("body") or "")
                        if len(b) > 80 and not b.lower().startswith(("+1", "👍", "me too")):
                            answers.append(b)
                answer = answers[0] if answers else ""
                if answer:
                    add_chat(f"github-{repo}", f"GitHub-Issue: {title}\n\n{body}", answer)
            time.sleep(0.3)
    print(f"[2] docker-issues: {sum(1 for c in chat if 'github-docker' in c['quelle'])} Chat")

# ---------------------------------------------------------------- 3) SO docker-Tag
def load_so_docker():
    base = "https://api.stackexchange.com/2.3/questions"
    params = {"site": "stackoverflow", "sort": "votes", "order": "desc",
              "pagesize": 100, "filter": "withbody", "tagged": "docker"}
    n = 0
    for page in (1, 2, 3):
        params["page"] = page
        r = requests.get(base, params=params, timeout=30)
        d = r.json()
        if "items" not in d:
            errors.append(f"so-docker: {d.get('error_message', d)}")
            break
        ids = [str(q["question_id"]) for q in d["items"]
               if q.get("accepted_answer_id") and q.get("score", 0) >= 3]
        if not ids:
            continue
        ar = requests.get("https://api.stackexchange.com/2.3/answers/%s"
                          "?site=stackoverflow&filter=withbody&pagesize=100" % ";".join(ids[:100]),
                          timeout=30)
        if ar.status_code != 200:
            continue
        answers = {a["answer_id"]: a for a in ar.json().get("items", [])}
        for q in d["items"]:
            if q.get("accepted_answer_id") in answers:
                a = answers[q["accepted_answer_id"]]
                if a.get("score", 0) >= 3:
                    qb = clean(q.get("body") or "")
                    ab = clean(a.get("body") or "")
                    if len(qb) > 80 and len(ab) > 80:
                        add_chat("so-docker", qb[:2000], truncate(ab))
                        n += 1
        time.sleep(0.4)
    print(f"[3] so-docker: {n} Chat")

# ---------------------------------------------------------------- 4) code_search_net python (try/except-Filter)
def load_csn_python():
    import json as j
    from huggingface_hub import hf_hub_download
    try:
        from datasets import load_dataset
        ds = load_dataset("code-search-net/code_search_net", "python",
                          split="train", streaming=True)
        n_chat = 0
        for d in ds:
            doc = clean(d.get("docstring") or "")
            code = clean(d.get("code") or "")
            if not doc or not code:
                continue
            if "try" in code and ("except" in code or "raise" in code):
                if 60 < len(doc) < 500:
                    add_chat("csn-python-err", f"Schreibe Python-Code für: {doc}",
                             truncate(code, 1200))
                    n_chat += 1
                    if n_chat >= 250:
                        break
        print(f"[4] csn-python (try/except): {n_chat} Chat")
    except Exception as e:
        errors.append(f"csn-python: {e}")

# ---------------------------------------------------------------- 5) SO python-Tag
def load_so_python():
    base = "https://api.stackexchange.com/2.3/questions"
    params = {"site": "stackoverflow", "sort": "votes", "order": "desc",
              "pagesize": 100, "filter": "withbody", "tagged": "python"}
    n = 0
    for page in (1, 2, 3):
        params["page"] = page
        r = requests.get(base, params=params, timeout=30)
        d = r.json()
        if "items" not in d:
            errors.append(f"so-python: {d.get('error_message', d)}")
            break
        ids = [str(q["question_id"]) for q in d["items"]
               if q.get("accepted_answer_id") and q.get("score", 0) >= 3]
        if not ids:
            continue
        ar = requests.get("https://api.stackexchange.com/2.3/answers/%s"
                          "?site=stackoverflow&filter=withbody&pagesize=100" % ";".join(ids[:100]),
                          timeout=30)
        if ar.status_code != 200:
            continue
        answers = {a["answer_id"]: a for a in ar.json().get("items", [])}
        for q in d["items"]:
            if q.get("accepted_answer_id") in answers:
                a = answers[q["accepted_answer_id"]]
                if a.get("score", 0) >= 3:
                    qb = clean(q.get("body") or "")
                    ab = clean(a.get("body") or "")
                    if len(qb) > 80 and len(ab) > 80:
                        add_chat("so-python", qb[:2000], truncate(ab))
                        n += 1
        time.sleep(0.4)
    print(f"[5] so-python: {n} Chat")

# ---------------------------------------------------------------- 6) Python-Projekt-Issues
def load_py_issues():
    repos = ["pallets/flask", "psf/requests", "django/django"]
    for repo in repos:
        url = ("https://api.github.com/repos/%s/issues"
               "?state=closed&per_page=100&sort=comments&direction=desc&labels=bug") % repo
        r = requests.get(url, headers=GH_HEADERS, timeout=30)
        if r.status_code != 200:
            errors.append(f"github-{repo}: HTTP {r.status_code}")
            continue
        for iss in r.json():
            if "pull_request" in iss:
                continue
            title = clean(iss.get("title", ""))
            body = clean(iss.get("body") or "")
            if len(body) < 100:
                continue
            c_url = iss["comments_url"]
            cr = requests.get(c_url + "?per_page=5", headers=GH_HEADERS, timeout=30)
            answers = []
            if cr.status_code == 200:
                for c in cr.json():
                    b = clean(c.get("body") or "")
                    if len(b) > 80 and not b.lower().startswith(("+1", "👍")):
                        answers.append(b)
            answer = answers[0] if answers else ""
            if answer:
                add_chat(f"github-{repo}", f"Bug-Issue: {title}\n\n{body}", answer)
            time.sleep(0.2)
    print(f"[6] py-issues: {sum(1 for c in chat if 'github-' in c['quelle'] and ('flask' in c['quelle'] or 'requests' in c['quelle'] or 'django' in c['quelle']))} Chat")

# ---------------------------------------------------------------- 7) GitHub-Workflow-Doku + Issues
def load_github_workflows():
    # github/docs: Pages über API sammeln
    base = "https://raw.githubusercontent.com/github/docs/main/content"
    dirs = ["get-started/learning-about-github", "get-started/using-git",
            "pull-requests/collaborating-with-pull-requests",
            "issues/tracking-your-work-with-issues", "github-cli"]
    n = 0
    for d in dirs:
        url = f"https://api.github.com/repos/github/docs/contents/content/{d}"
        r = requests.get(url, headers=GH_HEADERS, timeout=30)
        if r.status_code != 200:
            errors.append(f"github-docs/{d}: HTTP {r.status_code}")
            continue
        for item in r.json():
            if not item.get("name", "").endswith(".md"):
                continue
            raw = requests.get(item["download_url"], timeout=30)
            if raw.status_code != 200:
                continue
            text = clean(raw.text)
            if len(text) > 1500:
                add_completion(f"github-docs/{item['name']}", truncate(text))
                n += 1
        time.sleep(0.3)
    # cli/cli Issues
    url = "https://api.github.com/repos/cli/cli/issues?state=closed&per_page=100&sort=comments&direction=desc"
    r = requests.get(url, headers=GH_HEADERS, timeout=30)
    if r.status_code == 200:
        for iss in r.json():
            if "pull_request" in iss:
                continue
            title = clean(iss.get("title", ""))
            body = clean(iss.get("body") or "")
            if len(body) < 100:
                continue
            c_url = iss["comments_url"]
            cr = requests.get(c_url + "?per_page=5", headers=GH_HEADERS, timeout=30)
            answers = []
            if cr.status_code == 200:
                for c in cr.json():
                    b = clean(c.get("body") or "")
                    if len(b) > 80 and not b.lower().startswith(("+1", "👍")):
                        answers.append(b)
            answer = answers[0] if answers else ""
            if answer:
                add_chat("github-cli/cli", f"GitHub-CLI-Issue: {title}\n\n{body}", answer)
            time.sleep(0.2)
    print(f"[7] github-workflows: {n} Completion-Doku")

# ---------------------------------------------------------------- Main
if __name__ == "__main__":
    load_docker_docs()
    load_docker_issues()
    load_so_docker()
    load_csn_python()
    load_so_python()
    load_py_issues()
    load_github_workflows()

    print(f"\nRUNDE 3 - Chat: {len(chat)} | Completion: {len(completion)}")
    from collections import Counter
    print("Chat nach Quelle:", dict(Counter(c["quelle"] for c in chat)))
    print("Completion nach Quelle:", dict(Counter(c["quelle"] for c in completion)))
    if errors:
        print("Fehler:", errors[:8])

    (OUT / "train_chat3.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chat) + "\n")
    (OUT / "train_completion3.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in completion) + "\n")
    print("Gespeichert: data/train_chat3.jsonl + data/train_completion3.jsonl")
