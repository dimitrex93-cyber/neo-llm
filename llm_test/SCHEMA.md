# LLM-Benchmark — Schema (llm_test/)

Zweck: Halluzinations-Benchmark für kleine lokale LLMs (Ollama). Die Fragen sind so gebaut,
dass sie typische LLM-Halluzinationen triggern. Die Dateien dienen zugleich als
SFT-Trainingsdaten (frage + korrekte Antwort + erklärung = Ground Truth).

## Dateien

- `fragen_<thema>.json` — 12 Fragen je Thema (4 leicht / 4 mittel / 4 schwer)
- Themen: python, cpp, linux, docker, netzwerk, github, grammatik
- `evaluate_llm.py` — Benchmark-Harness (Fragen an Ollama, Auswertung, Report)

## JSON-Format (Liste von Frage-Objekten, UTF-8)

MC-Frage:
```json
{
  "id": "python_01",
  "typ": "mc",
  "frage": "…",
  "optionen": ["…", "…", "…", "…"],
  "antwort": 2,
  "punkte": 1,
  "schwierigkeit": "leicht",
  "erklaerung": "Ground-Truth-Begründung (kurz, für SFT nutzbar).",
  "halluzination_kategorie": "zahlen-falle"
}
```

Offene Frage:
```json
{
  "id": "python_04",
  "typ": "open",
  "frage": "…",
  "antwort": "Musterantwort als Text",
  "punkte": 3,
  "schwierigkeit": "schwer",
  "erklaerung": "…",
  "stichworte": ["begriff1", "begriff2"],
  "halluzination_kategorie": "detail"
}
```

## Halluzinations-Kategorien (halluzination_kategorie)

| Kategorie | Idee |
|---|---|
| `zahlen-falle` | Exakte Zahlen (Versionen, Ports, Exit-Codes, Defaults) — LLMs erfinden plausible Zahlen |
| `verwechslung` | Ähnliche Konzepte/Flags/Befehle (merge/rebase, -f/-F, TCP/UDP) |
| `falsch-freund` | Richtig klingende, aber falsche Aussage — „Ist X korrekt?" |
| `detail` | Exakte Syntax-/Flag-Details, Default-Verhalten, Reihenfolgen |
| `negativ` | „Welche Aussage ist FALSCH?" — lockt zur vorschnellen Zustimmung |
| `code-bug` | Code-Snippet mit subtilem Bug finden |

## Regeln

- Genau 1 korrekte MC-Option; Distraktoren plausibel (typische LLM-Halluzinationen)
- Fachlich 100 % korrekt (Ground Truth)
- Fragen und Erklärungen auf Deutsch; Code korrekt (englische Keywords)
- Punktwerte: leicht = 1, mittel = 2, schwer = 3
