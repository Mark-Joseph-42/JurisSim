# 🏛️ JurisSim: Neuro-Symbolic Legislative Stress-Testing

**JurisSim** is an advanced legislative audit engine designed to identify "legal bugs" or loopholes in draft laws before they are enacted. By combining frontier-grade LLM reasoning with formal mathematical verification, JurisSim provides a "Pre-Enactment Stress-Test" for lawmakers, judges, and parliamentary drafters.

## 🚀 The AMD Advantage
JurisSim is optimized for **AMD Instinct™ MI300X** accelerators.
- **DeepSeek-R1-0528 (685B MoE)**: Leveraging 8x MI300X GPUs to run the world's most powerful reasoning model.
- **1.5 TB Total VRAM**: Utilizing the massive HBM3 capacity for a "Universal Context Window," allowing the ingestion of entire legal codes without lossy chunking.
- **vLLM + ROCm™**: Optimized inference using AITER MLA kernels for high-throughput legislative auditing.

## 🧠 How It Works
1. **Clause Extraction**: Parses complex bills into testable legal mandates.
2. **Ambiguity Scoring (RQ)**: Calculates a Reliability Quotient (0.0-1.0) for every clause to identify high-risk phrasing.
3. **Adversarial Red-Teaming**: Simulates sophisticated corporate actors attempting to bypass the "Spirit of the Law" while following the "Letter of the Law."
4. **Formal Verification (Z3)**: Converts legal hypotheses into SMT formulas to mathematically prove if a loophole is satisfiable (`sat`).
5. **Legislative Hardening**: Automatically suggests precise amendments to close detected gaps.

## 🛠️ Quick Start (Local Prototype)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run mock test
python mock_test.py

# 3. Launch UI
python app.py
```

## ☁️ Server Deployment (AMD Developer Cloud - 8x MI300X)
1. **Clone the repo**: `git clone https://github.com/Mark-Joseph-42/JurisSim.git && cd JurisSim`
2. **Run setup**: `bash setup_server.sh`
3. **Monitor vLLM**: `tmux attach -t vllm` (Wait for it to load)
4. **Launch Demo**: `python3 app.py`

## ⚖️ Indian Law Corpus
JurisSim includes pre-indexed context for:
- Digital Personal Data Protection (DPDP) Act, 2023
- Information Technology (IT) Act, 2000
- Bharatiya Nyaya Sanhita (BNS), 2023
- Companies Act, 2013
- Income Tax Act, 1961
- Foreign Direct Investment (FDI) Policy

## 📄 License
MIT License. Built for the AMD Developer Hackathon 2026.
