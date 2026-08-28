# Neo — Feingetuntes Qwen2.5-3B für Homelab & DevOps 🇩🇪

[![HuggingFace](https://img.shields.io/badge/HuggingFace-Neo%201.0%3A3b-yellow)](https://huggingface.co/Dimitrex93/neo1.0-3b)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-green.svg)](https://www.python.org/)

**Neo** ist ein QLoRA-Fine-Tune von **Qwen2.5-3B-Instruct** — trainiert für den realen Homelab-/DevOps-Einsatz mit Schwerpunkt **Deutsch**. Das Besondere: Das komplette Training läuft auf einer **NVIDIA P2000 (5 GB, Pascal)** — ohne A100, ohne Cloud.

> *„Neo" (griech. νέος = neu) — ein Modell, das aus dem Basismodell neu geboren wurde: Linux, Docker, Netzwerktechnik, Python/C++ und deutsche Grammatik.*

## 🥊 Ergebnisse

### Benchmark (84 domänenspezifische Fragen)

| Modell | Punkte |
|---|---|
| Qwen2.5-3B (Basis) | 92/168 (55 %) |
| **Neo 1.0:3b** | **104/168 (62 %)** |

### Generalisierungstest (50 unbekannte Fragen — nicht im Training)

| Modell | Punkte |
|---|---|
| Qwen2.5-3B (Basis) | 57/104 (55 %) |
| **Neo 1.0:3b** | **74/104 (71 %)** |

Stärkste Verbesserungen: **Linux +33 pp**, **C++ +29 pp**, **zahlen-falle +57 pp** (14 % → 71 %).

## 🧠 Training

| Eigenschaft | Wert |
|---|---|
| Basismodell | `Qwen/Qwen2.5-3B-Instruct` (fp16) |
| Methode | QLoRA (r=16, α=32, NF4 4-bit) |
| Epochen / Sequenzlänge | 3 / 512 |
| Trainingsdaten | 1.644 Samples (84 → 20×) |
| Hardware | **NVIDIA P2000 5 GB** (~6,5 h) |
| Stack | Axolotl 0.4.0 · torch 2.2.2+cu118 · transformers 4.37.0 · bitsandbytes 0.43.3 |

**Datenquellen:** eigener LLM-Benchmark · the-stack-smol · code_search_net · tldr-pages · docker-docs · RFCs (DHCP/TCP/HTTP/TLS) · german.stackexchange · GitHub-Issues (docker, ollama, moby) · deutsche Wiki

## 📦 Reproduktion

```bash
# 1. venv
python3.11 -m venv venv
pip install --no-deps torch==2.2.2+cu118 --index-url https://download.pytorch.org/whl/cu118
# ... vollständige Pins siehe config.yml / Wiki

# 2. Trainingsdaten bauen (84 → 1.644 Samples)
python build_dataset.py     # Runde 1
python build_dataset2.py    # Runde 2
python build_dataset3.py    # Runde 3
python convert_to_sharegpt.py

# 3. Trainieren (P2000, GPU 0)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -m axolotl.cli.train config.yml

# 4. Export (Merge → GGUF → Ollama)
python merge_export.sh   # siehe Pipeline-Doku
ollama create neo1.0:3b -q Q4_K_M -f Modelfile
```

> ⚠️ **Pascal-Warnung:** Kein bf16, kein Flash-Attention, kein bitsandbytes > 0.43.3 (droppt Pascal-Support). Alle Pitfalls sind im Wiki dokumentiert.

## 🔬 Benchmark-Harness

Das Verzeichnis [`llm_test/`](llm_test/) enthält den vollständigen, eigenständigen Benchmark:

```bash
python3 llm_test/evaluate_llm.py --model neo1.0:3b
python3 llm_test/evaluate_llm.py --model neo1.0:3b --ordner llm_test_gen/   # Generalisierungstest
```

- 84 Fragen (MC + offen) in 7 Themen · 6 Halluzinations-Kategorien (falsch-freund, zahlen-falle, verwechslung …)
- 50 zusätzliche Generalisierungs-Fragen, die **nie** im Training waren
- Nur Python-Standardbibliothek — kein Framework nötig

## 🚀 Modell laden

```bash
ollama run hf.co/Dimitrex93/neo1.0-3b          # direkt von HuggingFace
```

## 💬 Feedback → Neo 1.2

Verbesserungen, Fehler und Themenwünsche bitte über die [HF-Discussions](https://huggingface.co/Dimitrex93/neo1.0-3b/discussions) — jeder Vorschlag fließt in die Daten für die nächste Version!

## 🗺️ Roadmap

- [x] Neo 1.0:3b — Basis-Fine-Tune (QLoRA, P2000)
- [ ] Neo 1.1:3b — +3.900 Samples (Docker, Python, GitHub-Workflows, Deutsch)
- [ ] Neo 1.2:3b — Community-Feedback einarbeiten

## ⚖️ Lizenz

Apache 2.0 (Modell), Basis-Modell unter [Qwen-Lizenz](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).
