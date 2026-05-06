# JurisSim — Final Implementation Plan (Safety-First)

## Key Clarification: No Training Required

**We are NOT training a model.** DeepSeek-R1-0528 is a pre-trained 685-billion parameter model released by DeepSeek under the MIT license. It has already been trained on trillions of tokens including legal text, code, and reasoning tasks.

We use it as-is for inference. The model's pre-trained knowledge of Indian law, Z3/SMT solving, and adversarial reasoning is sufficient for the demo.

---

## What Happens on the Server (Minimal Steps)

```
Step 1: git clone https://github.com/Mark-Joseph-42/JurisSim.git
Step 2: cd JurisSim
Step 3: bash setup_server.sh
Step 4: (wait for model to load — script monitors automatically)
Step 5: python3 app.py
Step 6: Open the Gradio URL in browser, run demo
```

**That's it. No coding on the server.** The setup_server.sh script handles:
- Environment variables (HSA_OVERRIDE, VLLM_ROCM_USE_AITER)
- Dependency installation
- .env file creation
- vLLM launch in tmux with correct flags
- Health-check polling until the model is ready

---

## Safety System: 5 Layers of Protection

### Layer 1: Model Cascade
If the primary model fails to download or load:
```bash
bash setup_server.sh 1   # DeepSeek-R1-0528 (685B) — best quality
bash setup_server.sh 2   # Qwen3-235B-A22B — fast + smart
bash setup_server.sh 3   # Qwen3-32B — smallest, always works
```

### Layer 2: API Retry (in llm_inference.py)
Every API call retries 3 times with exponential backoff (1s, 2s, 4s). A single network hiccup won't crash the demo.

### Layer 3: Template System (in z3_templates.py)
7 hardcoded Z3 patterns produce **correct code 100% of the time** for common loopholes. The model only needs to classify the pattern — not write Z3 from scratch.

For the Mirror Proxy demo: the model classifies it as `jurisdiction_evasion`, and the template renders perfect Z3 code instantly.

### Layer 4: Bespoke Z3 Generation
If no template matches, the model writes Z3 code using few-shot examples from the prompt. DeepSeek-R1 is excellent at code generation, so this works for novel patterns.

### Layer 5: Graceful Degradation
If Z3 formalization fails entirely, the clause is marked "inconclusive" (not "error"). The red-team hypotheses and suggested patches still appear — just without mathematical proof.

---

## File Change Summary

| File | Change | Purpose |
|:-----|:-------|:--------|
| `mock_data/*.md` | Replaced 5 generic acts with 6 real Indian laws | DPDP, IT Act, BNS, Companies Act, Income Tax, FDI |
| `demo_bills/digital_privacy_act_draft.md` | New | Mirror Proxy demo bill with planted loopholes |
| `src/llm_inference.py` | Added `LegalLLM_API` class + retry logic | vLLM inference with safety |
| `src/prompts.py` | Added `AMBIGUITY_SCORING_PROMPT` | Per-clause RQ scoring |
| `src/pipeline.py` | Added ambiguity scoring step | RQ scores in report |
| `app.py` | Added `USE_API` toggle | Seamless local↔server switching |
| `mock_test.py` | Added demo bill loading + API support | Server-ready testing |
| `setup_server.sh` | Complete rewrite with model cascade + health check | One-command deployment |
| `README.md` | Full documentation with architecture diagrams | Submission-ready |
