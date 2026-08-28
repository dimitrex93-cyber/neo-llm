#!/usr/bin/env python3
"""
evaluate_llm.py — LLM-Halluzinations-Benchmark (Ollama)

Lädt alle fragen_<thema>.json aus llm_test/, stellt jede Frage einem Modell
(Ollama /v1/chat/completions) und wertet aus:
  - MC:  Antwort-Zahl (0-3) wird per Regex extrahiert → volle Punkte bei Treffer
  - Open: Stichwort-Matching (wörtlich / Präfix ab 6 Zeichen, Umlaut-normalisiert)
         → anteilige Punkte (wie der _kws_abgleich der Lernapp)

Report: Gesamt-Score je Modell, je Thema, je Schwierigkeit, je
Halluzinations-Kategorie + Liste der Halluzinationen (falsch beantwortete Fragen).

Nutzung:
  python3 evaluate_llm.py --model llama3.2:3b-64k
  python3 evaluate_llm.py --model qwen2.5:3b-128k --thema linux --limit 6
  python3 evaluate_llm.py --model X --export-sft sft_daten.jsonl

Nur Stdlib (keine Abhängigkeiten). Ground Truth = die Fragenbanken selbst;
die Dateien dienen zugleich als SFT-Trainingsdaten.
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

BASE_URL = "http://127.0.0.1:11434/v1"
HERE = Path(__file__).resolve().parent

# Umlaut-Normalisierung (wie Lernapp _kws_abgleich)
UMLAUT = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}


def norm(s: str) -> str:
    s = s.lower()
    for a, b in UMLAUT.items():
        s = s.replace(a, b)
    return re.sub(r"[^a-z0-9 ]", " ", s)


def stichwort_treffer(antwort: str, stichworte: list[str]) -> tuple[int, int]:
    """Gefunden/Fehlt für eine offene Antwort. Wörtlich, sonst längstes Wort
    des Stichworts (>=6 Zeichen) als Präfix-Match."""
    ant = norm(antwort)
    gefunden, fehlt = 0, 0
    for sw in stichworte:
        sw_n = norm(sw)
        if sw_n in ant:
            gefunden += 1
            continue
        # längstes Wort >= 6 Zeichen als Präfix-Match
        woerter = [w for w in sw_n.split() if len(w) >= 6]
        if woerter and any(w in ant for w in woerter):
            gefunden += 1
            continue
        fehlt += 1
    return gefunden, fehlt


def llm_antwort(model: str, frage: str, system: str = "", timeout: int = 120,
                temperature: float = 0.0) -> tuple[str, float]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": frage})
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
        "max_tokens": 300,
    }
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=timeout))
        text = r["choices"][0]["message"].get("content", "").strip()
        # Reasoning-Tokens (falls vorhanden) abtrennen
        if "<｜end▁of▁thinking｜>" in text:
            text = text.split("<｜end▁of▁thinking｜>", 1)[1].strip()
        elif "<｜end▁of▁thinking｜>" in text:
            text = text.split(" response", 1)[1].strip()
        return text, time.time() - t0
    except Exception as e:
        return f"<FEHLER: {e}>", 0.0


def bewerte_mc(frage: dict, antwort_text: str) -> int:
    m = re.search(r"\b([0-3])\b", antwort_text)
    if not m:
        return 0
    return frage["punkte"] if int(m.group(1)) == frage["antwort"] else 0


def bewerte_open(frage: dict, antwort_text: str) -> int:
    gefunden, fehlt = stichwort_treffer(antwort_text, frage.get("stichworte", []))
    if gefunden + fehlt == 0:
        return 0
    return round(frage["punkte"] * gefunden / (gefunden + fehlt))


def frage_als_prompt(f: dict) -> str:
    if f["typ"] == "mc":
        opts = "\n".join(f"{i}. {o}" for i, o in enumerate(f["optionen"]))
        return (f"{f['frage']}\n{opts}\n\n"
                "Antworte NUR mit der Zahl der richtigen Option (0, 1, 2 oder 3).")
    return f"{f['frage']}\n\nAntworte in eigenen Worten auf Deutsch (1-3 Sätze)."


def lade_fragen(ordner: Path) -> list[dict]:
    fragen = []
    for pfad in sorted(ordner.glob("fragen_*.json")):
        d = json.loads(pfad.read_text(encoding="utf-8"))
        fragen.extend(d)
    return fragen


def export_sft(fragen: list[dict], out: Path) -> None:
    """Fragen als SFT-Trainingsdaten exportieren (JSONL, Chat-Format)."""
    with out.open("w", encoding="utf-8") as fh:
        for f in fragen:
            if f["typ"] == "mc":
                opts = "\n".join(f"{i}. {o}" for i, o in enumerate(f["optionen"]))
                loesung = f"Die richtige Antwort ist Option {f['antwort']}: {f['optionen'][f['antwort']]}"
            else:
                loesung = f["antwort"]
            eintrag = {
                "messages": [
                    {"role": "system", "content": "Du beantwortest technische Fragen korrekt und ohne zu erfinden."},
                    {"role": "user", "content": f["frage"]},
                    {"role": "assistant", "content": loesung + " " + f["erklaerung"]},
                ],
                "meta": {
                    "thema": f["id"].split("_")[0],
                    "schwierigkeit": f["schwierigkeit"],
                    "halluzination_kategorie": f.get("halluzination_kategorie", ""),
                },
            }
            fh.write(json.dumps(eintrag, ensure_ascii=False) + "\n")
    print(f"SFT-Daten: {len(fragen)} Eintraege -> {out}")


def main():
    ap = argparse.ArgumentParser(description="LLM-Halluzinations-Benchmark")
    ap.add_argument("--model", required=True, help="Ollama-Modellname")
    ap.add_argument("--ordner", type=Path, default=HERE,
                    help="Ordner mit fragen_*.json (default: llm_test/)")
    ap.add_argument("--thema", help="Nur ein Thema (z.B. linux)")
    ap.add_argument("--limit", type=int, help="Nur die ersten N Fragen")
    ap.add_argument("--export-sft", help="Pfad für SFT-JSONL-Export (statt Benchmark)")
    ap.add_argument("--samples", type=int, default=1,
                    help="Self-Consistency: N Antworten pro Frage, Mehrheitsentscheid (MC) bzw. "
                         "beste Stichwort-Abdeckung (open). Temperatur 0.7 für Diversität.")
    args = ap.parse_args()

    fragen = lade_fragen(args.ordner)
    if args.thema:
        fragen = [f for f in fragen if f["id"].startswith(args.thema + "_")]
    if args.limit:
        fragen = fragen[: args.limit]
    if not fragen:
        print("Keine Fragen gefunden (llm_test/fragen_*.json).")
        sys.exit(1)

    if args.export_sft:
        export_sft(fragen, Path(args.export_sft))
        return

    samples = max(1, args.samples)
    print(f"Benchmark: {args.model} | {len(fragen)} Fragen | Samples: {samples}\n")
    ergebnisse = []
    max_punkte = sum(f["punkte"] for f in fragen)

    for i, f in enumerate(fragen, 1):
        if samples > 1:
            # Self-Consistency: N Antworten, dann Mehrheits-/Best-Abdeckung
            antworten = [llm_antwort(args.model, frage_als_prompt(f), temperature=0.7)
                         for _ in range(samples)]
            texte = [a[0] for a in antworten]
            dt = sum(a[1] for a in antworten)
            if f["typ"] == "mc":
                stimmen = []
                for t in texte:
                    m = re.search(r"\b([0-3])\b", t)
                    if m:
                        stimmen.append(m.group(1))
                if stimmen:
                    mehrheit = max(set(stimmen), key=stimmen.count)
                    text = mehrheit
                else:
                    text = texte[0]
            else:
                text = max(texte, key=lambda t: stichwort_treffer(t, f.get("stichworte", []))[0])
        else:
            text, dt = llm_antwort(args.model, frage_als_prompt(f))
        if f["typ"] == "mc":
            erzielt = bewerte_mc(f, text)
        else:
            erzielt = bewerte_open(f, text)
        ergebnisse.append((f, text, erzielt, dt))
        status = "OK " if erzielt == f["punkte"] else ("teil" if erzielt > 0 else "FALSCH")
        print(f"[{i:02d}/{len(fragen)}] {f['id']:14} {status:5} {erzielt}/{f['punkte']} "
              f"({dt:4.1f}s)")

    # ---- Report ----
    gesamt = sum(e[2] for e in ergebnisse)
    print("\n" + "=" * 60)
    print(f"GESAMT: {gesamt}/{max_punkte} Punkte ({100*gesamt/max_punkte:.0f}%)")
    print(f"Zeit gesamt: {sum(e[3] for e in ergebnisse):.0f}s")

    # je Thema
    print("\n-- je Thema --")
    themen = {}
    for f, _, erzielt, _ in ergebnisse:
        themen.setdefault(f["id"].split("_")[0], [0, 0])
        themen[f["id"].split("_")[0]][0] += erzielt
        themen[f["id"].split("_")[0]][1] += f["punkte"]
    for t, (e, m) in sorted(themen.items()):
        print(f"  {t:12} {e:3}/{m:3} ({100*e/m:.0f}%)")

    # je Schwierigkeit
    print("\n-- je Schwierigkeit --")
    for sw in ["leicht", "mittel", "schwer"]:
        e = sum(x[2] for x in ergebnisse if x[0]["schwierigkeit"] == sw)
        m = sum(x[0]["punkte"] for x in ergebnisse if x[0]["schwierigkeit"] == sw)
        if m:
            print(f"  {sw:8} {e:3}/{m:3} ({100*e/m:.0f}%)")

    # je Halluzinations-Kategorie
    print("\n-- je Halluzinations-Kategorie --")
    kat = {}
    for f, _, erzielt, _ in ergebnisse:
        k = f.get("halluzination_kategorie", "?")
        kat.setdefault(k, [0, 0])
        kat[k][0] += erzielt
        kat[k][1] += f["punkte"]
    for k, (e, m) in sorted(kat.items()):
        print(f"  {k:14} {e:3}/{m:3} ({100*e/m:.0f}%)")

    # Halluzinationen auflisten
    print("\n-- Halluzinationen (0 Punkte) --")
    for f, text, erzielt, _ in ergebnisse:
        if erzielt == 0:
            print(f"\n  [{f['id']}] {f['frage'][:110]}")
            print(f"    LLM: {text[:140]}")

    print("\nFertig.")


if __name__ == "__main__":
    main()
