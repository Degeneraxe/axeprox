# ArbSense: Project Presentation Cheat-Sheet
**Final-Year CSE Capstone Project — MVSR Engineering College**
**Project Guide:** Mrs. P. Subhashini | **Team:** Akshara (050), Keerthana (056), Kusmitha (060)

---

## 🚀 How to Run the Website (Terminal / Command Prompt)

1. Open **Command Prompt** (or **PowerShell** / **Terminal**).
2. Navigate to the project folder:
   ```bash
   cd path/to/arbsense
   ```
3. Run the Streamlit dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```
4. The dashboard will automatically open in your browser at:
   `http://localhost:8501`

---

## 🎤 What to Present (Tab-by-Tab Live Architecture Walkthrough)

### 1. Tab 1: 🏛️ Architecture at a Glance (Slide 5, 7, 10)
* **Show the Layer-by-Layer visual breakdown:**
  * L1 (Feeds) $\rightarrow$ L2 (Normalizer) $\rightarrow$ **L3 (Detection & Cost Core - Shaded Core)** $\rightarrow$ L4 (Deep Learning) $\rightarrow$ L5 (Execution & Ledger).
  * **Design Principle:** *"The L3 Cost Model & Trade Ledger is the single source of truth; L4 feature projections and L5 execution plans are deterministically rebuilt from normalized tick streams."*
* **Point to the Store Selection Decision Record table:** Shows why SQLite WAL mode was selected over ScyllaDB and DuckDB for deterministic local execution.
* **Point to the Latency Budget table:** Breaks down the sub-50ms HFT execution budget (actual: ~9.3–20.4 ms).
* **Point to the Requirements Traceability Matrix:** Proves complete alignment with MVSR Review-2 Circular topics 1 to 12.

### 2. Tab 2: 📡 L1 & L2: Feeds & Depth (Slide 6 & 7)
* Show live Level 2 order books streaming from **Binance**, **Coinbase**, and **Kraken**.
* Point out the **RFC 4330 SNTP transit latency** and **Order Book Imbalance (OBI)**.

### 3. Tab 3: ⚖️ L3: Cost Core & Matrix (Slide 4, 5, 8)
* Point to the **Cross-Exchange Arbitrage Matrix**:
  * Show it under **`retail`**: The fee hurdle is **50.0–70.0 bps**, so spreads are flagged `🛡️ FILTERED (Fee Deficit)`.
  * **Switch sidebar to `zero_maker`**: The hurdle drops to **15.0 bps**, and positive net alpha appears!
* **Key Point to say:** *"Naive arbitrage bots fail because they ignore exchange fees. In calm markets, gross spreads are small. The L3 Validation Layer acts as a safety gatekeeper, rejecting sub-hurdle opportunities to preserve capital."*

### 4. Tab 4: 🧠 L4: Deep Learning Predictor (Slide 8 & 10)
* Show the **PyTorch Multi-Head LSTM Benchmark Table**:
  * **F1-Score:** 0.812 (PyTorch LSTM) vs 0.648 (Baseline LogReg) — **+25.3% improvement!**
  * **Duration MAE:** 142.3 ms vs 389.7 ms — **-63.5% error reduction!**
* **Key Point to say:** *"We trained a Multi-Head LSTM on 14 microstructure features (OBI, spread velocity, volatility) using walk-forward validation with purge gaps (zero future leakage). It predicts both opportunity viability and log-duration."*

### 5. Tab 5: 💼 L5: Execution Gatekeeper & Ledger (Slide 9 & 11)
* Show the **67 Filled Paper Trades** and **+$60.06 USDT Net Realized Profit** (100% Win Rate)!
* Point out the upward-sloping cumulative equity curve.
* **Live Interactive Demo for Guide:** Click the **`🧪 Test Execution Pipeline (Demo Trade)`** button right in front of the guide to demonstrate a live fill, fee deduction, and equity update in real-time!
