# 🏛️ JurisSim: Neuro-Symbolic Legislative Stress-Testing Engine

> **Pre-enactment loophole detection for draft legislation using adversarial AI + formal mathematical proofs.**

Built for the **AMD Developer Hackathon 2026** | Running on **8× AMD Instinct MI300X**

---

## 📌 The Problem

Laws have bugs — just like software. Corporations routinely exploit loopholes that drafters never intended:
- Facebook shifted data oversight from Ireland to the US to minimize GDPR exposure
- Uber classified workers as contractors to bypass labor protections
- Shell companies use jurisdictional routing to avoid tax obligations

**These loopholes are only discovered after enactment, through years of costly litigation.**

## 💡 The Solution

JurisSim is a **Pre-Enactment Stress-Test**. A Judge or Parliamentary Drafter uploads a draft bill, and the system:

1. **Discovers** exploitable loopholes using adversarial AI reasoning
2. **Proves** them mathematically using Z3 SMT formal verification
3. **Suggests** precise amendments to close each loophole
4. **Scores** every clause with an Ambiguity Score (0.0 → 1.0)

The bill is **hardened before it is even passed**, preventing loopholes from ever existing.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        JUDGE / DRAFTER                         │
│                    Uploads Draft Bill via UI                    │
└──────────────────────────┬─────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    GRADIO WEB INTERFACE                       │
│              (app.py — port 7861)                             │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                 ANALYSIS PIPELINE (pipeline.py)               │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │   CLAUSE     │  │  AMBIGUITY   │  │   ADVERSARIAL       │ │
│  │ EXTRACTION   │──│  SCORING     │──│   RED-TEAMING       │ │
│  │              │  │  (RQ 0→1)    │  │   (Strategist)      │ │
│  └─────────────┘  └──────────────┘  └──────────┬──────────┘ │
│                                                  │            │
│                    ┌─────────────────────────────┘            │
│                    │                                          │
│  ┌─────────────────▼──────────────────────────────────────┐  │
│  │              Z3 FORMALIZATION                           │  │
│  │                                                        │  │
│  │  Path A: Template Match (7 patterns, instant)          │  │
│  │      ↓ if no match                                     │  │
│  │  Path B: Bespoke Z3 Generation (model writes code)     │  │
│  │                                                        │  │
│  └─────────────────┬──────────────────────────────────────┘  │
│                    │                                          │
│  ┌─────────────────▼──────────────────────────────────────┐  │
│  │           Z3 SOLVER (Sandbox Execution)                 │  │
│  │                                                        │  │
│  │  sat   = Loophole mathematically proven                │  │
│  │  unsat = Clause is secure against this hypothesis      │  │
│  │  error = Formalization failed (reported as inconclusive)│  │
│  └─────────────────┬──────────────────────────────────────┘  │
│                    │                                          │
│                  sat?──────────────── no ──▶ Mark Secure     │
│                    │                                          │
│                   yes                                         │
│                    │                                          │
│  ┌─────────────────▼──────────────────────────────────────┐  │
│  │         PATCH GENERATION (Auditor role)                 │  │
│  │    "Amend Section X to include..."                     │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    REPORT WITH:                               │
│  • Per-clause Ambiguity Score (RQ)                           │
│  • Adversarial hypotheses                                    │
│  • Z3 mathematical proofs                                    │
│  • Suggested amendments                                      │
└──────────────────────────────────────────────────────────────┘
```

### Infrastructure Layer

```
┌──────────────────────────────────────────────────────────────┐
│                  AMD INSTINCT MI300X × 8                      │
│                  1,536 GB HBM3 Total VRAM                    │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  vLLM Inference Server (port 8000)                     │  │
│  │  DeepSeek-R1-0528 (685B MoE, ~700GB FP8)              │  │
│  │  Tensor Parallel across 8 GPUs                         │  │
│  │  AITER MLA Optimizations for ROCm                      │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │ Qdrant       │  │ BGE-small    │  │ Z3 Solver          │ │
│  │ Vector DB    │  │ Embeddings   │  │ (sandboxed)        │ │
│  │ (in-memory)  │  │ (384-dim)    │  │                    │ │
│  └──────────────┘  └──────────────┘  └────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## 🔒 Safety & Fallback System

JurisSim is built to **never fail during a demo**:

| Layer | What Could Fail | Safety Net |
|:------|:----------------|:-----------|
| **Model Download** | DeepSeek-R1 is 700GB | `bash setup_server.sh 2` → Qwen3-235B (470GB) or `3` → Qwen3-32B (64GB) |
| **API Connection** | Network timeout mid-analysis | 3 retries with exponential backoff (1s, 2s, 4s) |
| **Z3 Formalization** | Model generates invalid code | Template system catches 7 common loophole patterns with 100% reliability |
| **Novel Patterns** | No template matches | Model generates bespoke Z3 code (works well with 685B model) |
| **Solver Timeout** | Z3 hangs on complex formula | 10-second timeout with graceful "inconclusive" fallback |
| **Security** | Malicious code injection | Sandbox blocks `os`, `subprocess`, `sys`, `eval`, `exec`, `open` |

### Model Cascade

```
Tier 1 (Default)         Tier 2 (Fallback)         Tier 3 (Guaranteed)
┌──────────────────┐    ┌──────────────────┐     ┌──────────────────┐
│ DeepSeek-R1-0528 │    │ Qwen3-235B-A22B  │     │ Qwen3-32B        │
│ 685B MoE         │───▶│ 235B MoE         │────▶│ 32B Dense        │
│ ~700GB FP8       │    │ ~470GB BF16      │     │ ~64GB BF16       │
│ Best reasoning   │    │ Fast + smart     │     │ Fastest          │
└──────────────────┘    └──────────────────┘     └──────────────────┘
```

---

## 📂 Project Structure

```
JurisSim/
├── app.py                    # Gradio web UI (port 7861)
├── mock_test.py              # End-to-end pipeline test
├── setup_server.sh           # One-command MI300X deployment
├── requirements.txt          # Python dependencies
├── .env                      # Configuration (auto-generated by setup_server.sh)
│
├── src/
│   ├── llm_inference.py      # LegalLLM (local) + LegalLLM_API (vLLM)
│   ├── pipeline.py           # Main analysis pipeline + report formatter
│   ├── prompts.py            # All prompt templates
│   ├── vector_db.py          # Qdrant vector DB + document chunker
│   ├── z3_solver.py          # Sandboxed Z3 code execution
│   └── z3_templates.py       # 7 hardcoded loophole pattern templates
│
├── mock_data/                # Indian Law Corpus (6 acts)
│   ├── dpdp_act_2023.md      # Digital Personal Data Protection Act
│   ├── it_act_2000.md        # Information Technology Act
│   ├── bns_2023.md           # Bharatiya Nyaya Sanhita
│   ├── companies_act_2013.md # Companies Act
│   ├── income_tax_act_1961.md# Income Tax Act
│   └── fdi_policy.md         # Foreign Direct Investment Policy
│
├── demo_bills/               # Demo bills with intentional loopholes
│   └── digital_privacy_act_draft.md  # Mirror Proxy loophole demo
│
├── tests/                    # Unit tests
│   ├── test_z3_golden.py     # Z3 template golden tests
│   ├── test_retrieval.py     # Vector DB retrieval tests
│   └── test_formalization.py # Formalization pipeline tests
│
└── training/                 # Training data (for future fine-tuning)
    ├── dataset.jsonl
    ├── legal_clauses.jsonl
    ├── generate_pairs.py
    └── train_qlora.py
```

---

## 🚀 Quick Start

### Local Development (CPU/small GPU)
```bash
pip install -r requirements.txt
python mock_test.py          # Run pipeline test
python app.py                # Launch Gradio UI at localhost:7861
```

### Server Deployment (8× MI300X)
```bash
git clone https://github.com/Mark-Joseph-42/JurisSim.git
cd JurisSim
bash setup_server.sh         # Installs deps, launches vLLM, waits for ready
python3 app.py               # Launch Gradio UI
```

**If DeepSeek-R1 fails to load:**
```bash
bash setup_server.sh 2       # Fallback: Qwen3-235B-A22B
bash setup_server.sh 3       # Emergency: Qwen3-32B (always works)
```

---

## 🎯 Demo: The Mirror Proxy Loophole

### The Draft Bill
The demo uses a "Digital Privacy and Accountability Act" (`demo_bills/digital_privacy_act_draft.md`) containing intentional vulnerabilities.

### The Key Loophole
**Section 2(a)** defines "Data Controller" as:
> *"any entity registered under the Companies Act, 2013, that collects, stores, or processes personal data of Indian citizens through servers physically located within the territory of India."*

**The exploit:** An offshore entity (not registered in India) sets up a "mirror proxy" server. Data flows through India but is processed abroad. This entity is **not a "Data Controller"** under this act and escapes all obligations.

### Expected Output
- **Ambiguity Score (RQ): ~0.85** — High vulnerability
- **Loophole type**: `jurisdiction_evasion`
- **Z3 result**: `sat` — mathematically proven exploitable
- **Suggested amendment**: *"Amend Section 2(a) to read: 'Data Controller means any entity exercising de facto control over the processing of personal data of Indian citizens, regardless of its place of registration or the physical location of its servers.'"*

---

## ⚖️ Indian Law Coverage

The vector database is pre-loaded with real provisions from:

| Act | Key Sections | Relevance |
|:----|:-------------|:----------|
| **DPDP Act, 2023** | Data Fiduciary, Consent, Cross-Border Transfer, Penalties | Data privacy loopholes |
| **IT Act, 2000** | Intermediary definition, Safe Harbour, Section 69 | Digital regulation gaps |
| **BNS, 2023** | Organised Crime, Cheating, Punishments | Criminal law definitions |
| **Companies Act, 2013** | Subsidiary, Related Party, CSR, Fraud | Corporate structuring exploits |
| **Income Tax Act, 1961** | Residency (182-day), Transfer Pricing | Tax avoidance patterns |
| **FDI Policy** | E-commerce rules, Beneficial Ownership | Foreign investment loopholes |

---

## 🔧 Z3 Template Patterns

7 pre-built loophole patterns for instant, reliable formalization:

| Pattern | Example | Z3 Strategy |
|:--------|:--------|:------------|
| `threshold_splitting` | Corp splits into subsidiaries to stay under emissions cap | Int constraints: per-entity ≤ cap, total > cap |
| `definition_gap` | Using "loyalty points" instead of "currency" to avoid tax | Bool: undefined_term ≠ defined_term, obligation bypassed |
| `jurisdiction_evasion` | Mirror proxy routes data through non-covered jurisdiction | Bool: foreign=true, domestic=false, obligation=false |
| `temporal_gap` | Exploiting "72 hours after becoming aware" notification window | Int: delay > deadline |
| `scope_limitation` | Company restructures to fall below ₹500cr CSR threshold | Int: entity_value < threshold, obligation=false |
| `aggregation_evasion` | Splitting ₹10L transaction into 100 × ₹10K to avoid reporting | Int: each < cap, sum > cap |
| `consent_loophole` | Arguing "continued use" = implicit consent | Bool: alternate_action ≠ required_consent |

---

## 🖥️ The AMD Advantage

| Feature | Benefit for JurisSim |
|:--------|:---------------------|
| **192GB HBM3 per GPU** | Universal Context Window — entire legal codes held in memory |
| **8× MI300X (1,536GB total)** | Run DeepSeek-R1 (685B) — the world's most powerful reasoning model |
| **vLLM + ROCm** | Production-grade inference with AITER MLA kernel optimizations |
| **Tensor Parallelism** | Model sharded across 8 GPUs for maximum throughput |

---

## 📄 License

MIT License. Built for the AMD Developer Hackathon 2026.
