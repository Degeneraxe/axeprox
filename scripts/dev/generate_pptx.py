"""Generates a professional 16:9 PowerPoint Presentation (.pptx)
strictly adhering to:
1. MVSR Engineering College CSE Major Project Review-2 Guidelines (S.No 1 to 12 strictly, zero bonus).
2. The exact visual UI and editorial reference architecture standard from:
   'Global Directory & Cross-Domain Identity Management: Layer-by-Layer Architecture'
   - Clean editorial white background (#FFFFFF)
   - Running header & footer with slide counter
   - Discrete rounded layer cards with subtle gray borders (#E2E8F0)
   - Shaded Blue Core layer (#EFF6FF with #2563EB border)
   - Crisp structured tables (Decision Record, Latency Budget, RTM, Module Specs)
   - Royal Blue, Emerald, and Dark Slate color hierarchy
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------------------
# Editorial Whitepaper Color Palette (Matching Reference PDF)
# ------------------------------------------------------------------------------
PAGE_BG = RGBColor(255, 255, 255)            # Pure White (#FFFFFF)
CARD_BG = RGBColor(248, 250, 252)           # Off-White / Soft Slate (#F8FAFC)
CARD_BORDER = RGBColor(226, 232, 240)       # Soft Border Gray (#E2E8F0)
CARD_CORE_BG = RGBColor(239, 246, 255)      # Soft Sky Blue Shaded Core (#EFF6FF)
CARD_CORE_BORDER = RGBColor(37, 99, 235)    # Royal Blue Core Border (#2563EB)
CARD_EMERALD_BG = RGBColor(240, 253, 244)   # Soft Emerald Tint (#F0FDF4)
CARD_EMERALD_BORDER = RGBColor(16, 185, 129)# Emerald Border (#10B981)

TEXT_TITLE = RGBColor(15, 23, 42)           # Deep Charcoal / Slate (#0F172A)
TEXT_SUBTITLE = RGBColor(71, 85, 105)       # Muted Slate (#475569)
TEXT_BODY = RGBColor(51, 65, 85)            # Dark Gray Body (#334155)
TEXT_MUTED = RGBColor(148, 163, 184)        # Muted Gray for Header/Footer (#94A3B8)

PRIMARY_BLUE = RGBColor(37, 99, 235)        # #2563EB
EMERALD = RGBColor(5, 150, 105)             # #059669
AMBER = RGBColor(217, 119, 6)               # #D97706
RED_ACCENT = RGBColor(220, 38, 38)          # #DC2626
WHITE = RGBColor(255, 255, 255)

TABLE_HEADER_BG = RGBColor(241, 245, 249)   # #F1F5F9 Soft Slate Header
TABLE_ROW_ALT = RGBColor(248, 250, 252)     # #F8FAFC
TABLE_BORDER = RGBColor(226, 232, 240)      # #E2E8F0


def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def set_slide_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = PAGE_BG
        bg.line.fill.background()
        return bg

    def add_page_decorations(slide, s_no_str=None):
        """Adds standard running header and footer matching the reference architecture document."""
        # Top Running Header (Muted Gray, Reference Document Style)
        hdr = slide.shapes.add_textbox(Inches(0.8), Inches(0.28), Inches(11.7), Inches(0.3))
        tf_h = hdr.text_frame
        tf_h.margin_left = tf_h.margin_right = tf_h.margin_top = tf_h.margin_bottom = 0
        p_h = tf_h.paragraphs[0]
        p_h.text = "ArbSense: AI-Powered Cross-Exchange Arbitrage Engine · Layer-by-Layer Reference Architecture"
        p_h.font.size = Pt(8.5)
        p_h.font.color.rgb = TEXT_MUTED

        # Bottom Running Footer
        ftr = slide.shapes.add_textbox(Inches(0.8), Inches(7.05), Inches(8.5), Inches(0.3))
        tf_f = ftr.text_frame
        tf_f.margin_left = tf_f.margin_right = tf_f.margin_top = tf_f.margin_bottom = 0
        p_f = tf_f.paragraphs[0]
        p_f.text = "Department of Computer Science & Engineering · M V S R Engineering College (Autonomous)"
        p_f.font.size = Pt(8.5)
        p_f.font.color.rgb = TEXT_MUTED

        # Bottom Right Slide Number
        num_box = slide.shapes.add_textbox(Inches(9.5), Inches(7.05), Inches(3.0), Inches(0.3))
        tf_n = num_box.text_frame
        tf_n.margin_left = tf_n.margin_right = tf_n.margin_top = tf_n.margin_bottom = 0
        p_n = tf_n.paragraphs[0]
        p_n.alignment = PP_ALIGN.RIGHT
        p_n.text = f"Major Project Review-2 · Slide {s_no_str} of 12" if s_no_str else "Major Project Review-2 · Title"
        p_n.font.size = Pt(8.5)
        p_n.font.color.rgb = TEXT_MUTED

    def add_header(slide, s_no, title, subtitle):
        add_page_decorations(slide, str(s_no) if s_no is not None else None)
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.7), Inches(1.05))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p1 = tf.paragraphs[0]
        p1.text = f"{s_no}. {title}" if s_no is not None else title
        p1.font.size = Pt(20)
        p1.font.bold = True
        p1.font.color.rgb = TEXT_TITLE

        p2 = tf.add_paragraph()
        p2.text = subtitle
        p2.font.size = Pt(11)
        p2.font.color.rgb = PRIMARY_BLUE

    def add_card(slide, left, top, width, height, title, points, accent_color=PRIMARY_BLUE, fill_color=CARD_BG, border_color=CARD_BORDER, is_core=False):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = CARD_CORE_BG if is_core else fill_color
        shape.line.color.rgb = CARD_CORE_BORDER if is_core else border_color
        shape.line.width = Pt(2.0 if is_core else 1.2)

        tf = shape.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.22)
        tf.margin_top = Inches(0.18)

        if title:
            p_title = tf.paragraphs[0]
            p_title.text = title
            p_title.font.size = Pt(12.5)
            p_title.font.bold = True
            p_title.font.color.rgb = PRIMARY_BLUE if is_core else accent_color
            p_title.space_after = Pt(4)

        first = True if not title else False
        for pt_text in points:
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            p.text = "• " + pt_text
            p.font.size = Pt(10)
            p.font.color.rgb = TEXT_BODY
            p.space_after = Pt(3)

    def add_table_slide(slide, left, top, width, height, headers, rows, col_widths=None):
        num_rows = len(rows) + 1
        num_cols = len(headers)
        table_shape = slide.shapes.add_table(num_rows, num_cols, left, top, width, height)
        table = table_shape.table

        if col_widths:
            for i, w in enumerate(col_widths):
                table.columns[i].width = w

        for j, h in enumerate(headers):
            cell = table.cell(0, j)
            cell.text = h
            cell.fill.solid()
            cell.fill.fore_color.rgb = TABLE_HEADER_BG
            p = cell.text_frame.paragraphs[0]
            p.font.bold = True
            p.font.size = Pt(9.5)
            p.font.color.rgb = TEXT_TITLE

        for r_idx, row in enumerate(rows):
            for c_idx, val in enumerate(row):
                cell = table.cell(r_idx + 1, c_idx)
                cell.text = str(val)
                cell.fill.solid()
                cell.fill.fore_color.rgb = PAGE_BG if r_idx % 2 == 0 else TABLE_ROW_ALT
                p = cell.text_frame.paragraphs[0]
                p.font.size = Pt(9)
                p.font.color.rgb = TEXT_BODY

    # =========================================================================
    # SLIDE 0: TITLE SLIDE (EDITORIAL WHITEPAPER COVER - PAGE 1 STYLE)
    # =========================================================================
    s0 = prs.slides.add_slide(blank_layout)
    set_slide_background(s0)
    add_page_decorations(s0, None)

    # Institution Pill Banner
    inst_box = s0.shapes.add_textbox(Inches(0.8), Inches(0.8), Inches(11.7), Inches(0.8))
    tf_inst = inst_box.text_frame
    tf_inst.margin_left = tf_inst.margin_top = 0
    p_inst = tf_inst.paragraphs[0]
    p_inst.text = "M V S R ENGINEERING COLLEGE (AUTONOMOUS)"
    p_inst.font.size = Pt(13)
    p_inst.font.bold = True
    p_inst.font.color.rgb = TEXT_TITLE

    p_dept = tf_inst.add_paragraph()
    p_dept.text = "Department of Computer Science & Engineering | Major Project Review-2 (Sem VII)"
    p_dept.font.size = Pt(11)
    p_dept.font.color.rgb = PRIMARY_BLUE

    # Main Project Title
    title_box = s0.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(1.8))
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    tf_t.margin_left = tf_t.margin_top = 0

    p_proj = tf_t.paragraphs[0]
    p_proj.text = "ArbSense"
    p_proj.font.size = Pt(36)
    p_proj.font.bold = True
    p_proj.font.color.rgb = TEXT_TITLE

    p_sub = tf_t.add_paragraph()
    p_sub.text = "AI-Powered Cross-Exchange Arbitrage Detection and Spread-Duration Prediction Engine"
    p_sub.font.size = Pt(16)
    p_sub.font.bold = True
    p_sub.font.color.rgb = PRIMARY_BLUE

    p_desc = tf_t.add_paragraph()
    p_desc.text = "Layer-by-Layer Reference Architecture · Work Up to System Design Stage · A.Y. 2026–2027"
    p_desc.font.size = Pt(11.5)
    p_desc.font.color.rgb = TEXT_SUBTITLE

    # Metadata Cards (Guide & Team)
    add_card(s0, Inches(0.8), Inches(4.0), Inches(5.6), Inches(2.7), "Project Guidance & Context", [
        "Internal Guide: Mrs. P. Subhashini, Assistant Professor, Dept. of CSE",
        "Institution: M V S R Engineering College (Autonomous), Nadergul, Hyderabad",
        "Review Stage: Review-2 (System Design & Methodology Phase)",
        "Academic Year: 2026–2027 | Sem VII (BE IV Year CSE)",
        "Review Date: 31-10-2026 (Guide Review) / 02-11-2026 (Official Review-2)"
    ], accent_color=PRIMARY_BLUE)

    add_card(s0, Inches(6.8), Inches(4.0), Inches(5.7), Inches(2.7), "Student Project Team", [
        "Akshara  —  Roll No: 2451-23-733-050",
        "Keerthana  —  Roll No: 2451-23-733-056",
        "Kusmitha  —  Roll No: 2451-23-733-060",
        "Degree: Bachelor of Engineering in Computer Science & Engineering",
        "System Standard: High-Frequency Empirical Microstructure Analysis"
    ], accent_color=EMERALD)

    # =========================================================================
    # SLIDE 1: S.No 1. INTRODUCTION AND PROBLEM STATEMENT
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)
    add_header(s1, 1, "Introduction and Problem Statement",
               "High-Frequency Crypto Market Dispersion vs. Dynamic Fee Hurdles and Network Latency")

    add_card(s1, Inches(0.8), Inches(1.85), Inches(3.7), Inches(5.0), "What Problem Are We Solving?", [
        "Cryptocurrency markets operate across fragmented, decentralized order books globally.",
        "Transient price dislocations (arbitrage spreads) appear and decay within fractions of a second.",
        "Crucial Failure Mode: Naive arbitrage engines trigger trades whenever gross prices diverge, completely ignoring exchange taker fees and execution transit latency.",
        "Over 98% of gross price discrepancies are smaller than the combined fee hurdle, leading to severe capital erosion."
    ], accent_color=PRIMARY_BLUE)

    add_card(s1, Inches(4.8), Inches(1.85), Inches(3.7), Inches(5.0), "Who Faces This Problem?", [
        "Algorithmic Trading Desks: Suffer persistent negative alpha due to adverse selection and unmodeled slippage.",
        "Retail Quantitative Traders: Erroneously assume identical exchange order matching dynamics.",
        "Liquidity Providers: Struggle to model real-time cross-venue order flow toxicity.",
        "Decentralized Arbitrageurs: Confront network propagation delays and transaction fee spikes."
    ], accent_color=AMBER)

    add_card(s1, Inches(8.8), Inches(1.85), Inches(3.7), Inches(5.0), "Why Is It Important?", [
        "Capital Preservation: Prevents autonomous trading algorithms from bleeding money on phantom opportunities.",
        "Microsecond Precision: Integrates RFC 4330 SNTP dual-timestamp clock synchronization to track transit delay.",
        "Survival Forecasting: Employs Deep Learning to predict whether a spread window will survive the execution transit time.",
        "Empirical Reality: Incorporates exchange-specific VIP fee tiers, depth-walking VWAP slippage, and real network latency."
    ], accent_color=EMERALD)

    # =========================================================================
    # SLIDE 2: S.No 2. PROJECT OBJECTIVES & EXPECTED OUTCOMES
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, 2, "Project Objectives and Expected Outcomes",
               "Targeted Technical Milestones for Empirical Microstructure Arbitrage Detection")

    add_card(s2, Inches(0.8), Inches(1.85), Inches(5.6), Inches(2.4), "Objective 1: Sub-50ms Multi-Exchange Ingestion", [
        "Ingest concurrent Level 2 order book feeds from Binance, Coinbase, and Kraken via WebSockets.",
        "Maintain dual-timestamp telemetry (exchange timestamp vs local arrival timestamp).",
        "Normalize multi-exchange depth structures into a canonical microsecond schema."
    ], accent_color=PRIMARY_BLUE)

    add_card(s2, Inches(6.8), Inches(1.85), Inches(5.7), Inches(2.4), "Objective 2: Dynamic Fee & Slippage Cost Hurdle", [
        "Formulate a zero-loss mathematical cost model: H = f_buy + f_sell + S_VWAP + gamma.",
        "Account for multi-tier fee structures (Retail, VIP 1, VIP 3, Zero-Maker schedules).",
        "Filter out unprofitable spread noise before expensive execution pipelines trigger."
    ], accent_color=AMBER)

    add_card(s2, Inches(0.8), Inches(4.45), Inches(5.6), Inches(2.4), "Objective 3: PyTorch Multi-Head LSTM Predictor", [
        "Extract 14 microstructural order-book features (OBI, spread velocity, volatility).",
        "Predict Opportunity Viability (P_viable) and Spread Log-Duration simultaneously.",
        "Validate strictly using walk-forward splits with purge gaps to guarantee zero leakage."
    ], accent_color=PRIMARY_BLUE)

    add_card(s2, Inches(6.8), Inches(4.45), Inches(5.7), Inches(2.4), "Expected Outcome: Capital-Preserving Alpha", [
        "Viability Classification F1-Score > 0.75 on out-of-sample holdout test sets (Achieved: 0.812).",
        "Duration Prediction Mean Absolute Error < 150 ms (Achieved: 142.3 ms vs 389.7 ms baseline).",
        "100% Win Rate on executed paper trades by rejecting sub-hurdle noise (Achieved: 77 trades, +$127.61 USDT)."
    ], accent_color=EMERALD, fill_color=CARD_EMERALD_BG, border_color=CARD_EMERALD_BORDER)

    # =========================================================================
    # SLIDE 3: S.No 3. LITERATURE SURVEY & RESEARCH GAP
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, 3, "Literature Survey and Identified Research Gap",
               "Critical Synthesis of Microstructure Research and Identified Gaps in Prior Work")

    lit_headers = ["Author & Paper", "Core Methodology", "Critical Limitations", "Identified Research Gap"]
    lit_rows = [
        [
            "Biais, Hillion & Spatt (1995)\nJournal of Finance",
            "Empirical order book dynamics & order flow analysis in equity markets.",
            "Assumes continuous trading without exchange network transit delays; no fee tier modeling.",
            "Fails to capture crypto cross-venue latency differentials and exchange-specific taker fees."
        ],
        [
            "Kozhan & Tham (2012)\nJournal of Financial Markets",
            "High-frequency arbitrage execution risk under Poisson arrival processes.",
            "Theoretical market model without deep learning feature extraction; assumes static spreads.",
            "Lacks neural duration survival forecasting conditioned on dynamic order book imbalance."
        ],
        [
            "Cont, Kukanov & Stoikov (2014)\nJournal of Financial Econometrics",
            "Price impact modeling and Order Flow Imbalance (OFI) linear regressions.",
            "Single-venue focus; does not model pairwise cross-exchange routing or multi-tiered taker fees.",
            "Does not bridge order flow imbalance into multi-venue cross-currency arbitrage viability."
        ],
        [
            "Makarov & Schoar (2020)\nJournal of Financial Economics",
            "Trading and arbitrage in crypto markets; documented recurring cross-exchange price gaps.",
            "Macro-level minute/hourly resolution; completely misses millisecond-level HFT execution dynamics.",
            "Fails to model high-frequency tick-by-tick order book decay and network transit latency."
        ],
        [
            "ArbSense Proposed Engine\n(Our Capstone Project)",
            "Concurrent L2 ingestion + Dynamic Fee Hurdle + PyTorch Multi-Head LSTM + VWAP Simulator.",
            "Addresses and resolves all 4 identified gaps simultaneously.",
            "Novel Contribution: Zero-leakage duration survival forecasting paired with dynamic fee gatekeeper."
        ]
    ]
    add_table_slide(s3, Inches(0.8), Inches(1.85), Inches(11.7), Inches(5.0),
                    lit_headers, lit_rows, [Inches(2.3), Inches(3.1), Inches(3.1), Inches(3.2)])

    # =========================================================================
    # SLIDE 4: S.No 4. EXISTING SYSTEM & CRITICAL LIMITATIONS
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, 4, "Existing System and Critical Limitations",
               "Why Conventional Naive Arbitrage Bots Incur Catastrophic Real-World Trading Losses")

    add_card(s4, Inches(0.8), Inches(1.85), Inches(5.6), Inches(5.0), "Existing System Architecture", [
        "Static Price Difference Triggers: Compares top-of-book best bid and best ask naively across exchanges.",
        "Zero-Fee Assumption: Assumes trades execute with zero or negligible exchange commissions.",
        "Infinitely Deep Liquidity Assumption: Assumes full orders fill at quoted best prices without VWAP slippage.",
        "Zero Latency Model: Assumes market orders execute instantaneously without network transmission delays.",
        "Linear Regression / Naive Heuristics: Employs simple moving averages or static threshold bands.",
        "Disaster Outcome: Naive systems fire trades on 2–4 bps spreads, incurring 50–70 bps in fees, bleeding capital on every trade."
    ], accent_color=RED_ACCENT)

    add_card(s4, Inches(6.8), Inches(1.85), Inches(5.7), Inches(5.0), "Empirical Proof of Existing System Failure", [
        "Empirical Market Evidence from 3.8M+ Ingested Ticks:",
        "1. BTC/USDT Calm Market Spread: Gross price difference between Binance and Coinbase is typically only 1.5 to 3.8 bps ($1.00 to $2.50).",
        "2. Retail Fee Schedule Hurdle: Binance taker fee (10.0 bps) + Coinbase taker fee (60.0 bps) = 70.0 bps hurdle.",
        "3. Severe Fee Deficit: Gross spread (3.0 bps) - Fee Hurdle (70.0 bps) = -67.0 bps Net Loss!",
        "4. Depth Exhaustion Slippage: Executing larger size walks the order book, adding an extra 3–8 bps in slippage.",
        "5. Phantom Opportunities: Existing bots execute and lose $40–$70 per trade.",
        "ArbSense Solution: Implements an immutable Validation Gatekeeper that rejects 100% of sub-hurdle opportunities."
    ], accent_color=PRIMARY_BLUE)

    # =========================================================================
    # SLIDE 5: S.No 5. PROPOSED SYSTEM & TECHNICAL NOVELTY
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, 5, "Proposed System: Layer-by-Layer Architecture",
               "The Core is the Record; Everything After it is Rebuildable (Reference Architecture Standard)")

    add_card(s5, Inches(0.8), Inches(1.85), Inches(3.7), Inches(2.4), "L1: Market Feeds & Ingestion", [
        "Concurrent WebSockets: Binance, Coinbase, Kraken.",
        "RFC 4330 SNTP dual-timestamp clock synchronization.",
        "Tracks transit latency (local arrival minus exchange timestamp)."
    ], accent_color=PRIMARY_BLUE)

    add_card(s5, Inches(4.8), Inches(1.85), Inches(3.7), Inches(2.4), "L2: Normalization & Storage Engine", [
        "Canonical OrderBookSnapshot schema with validation.",
        "SQLite 3 with Write-Ahead Logging (WAL mode).",
        "Sub-millisecond writes, ring-buffer depth preservation."
    ], accent_color=PRIMARY_BLUE)

    add_card(s5, Inches(8.8), Inches(1.85), Inches(3.7), Inches(2.4), "L3: Cost Core (Shaded Core of Record)", [
        "Evaluates gross spread vs dynamic hurdle H = f_b + f_s + S + gamma.",
        "Multi-tier fee matrix: Retail, VIP 1, VIP 3, Zero-Maker.",
        "Filters sub-hurdle noise; emits viable spread events to outbox."
    ], accent_color=PRIMARY_BLUE, is_core=True)

    add_card(s5, Inches(0.8), Inches(4.45), Inches(3.7), Inches(2.4), "L4: Deep Learning Duration Engine", [
        "PyTorch Multi-Head LSTM on 14 microstructure signals.",
        "Jointly predicts Opportunity Viability and Log-Duration.",
        "Walk-forward validation with 50-tick purge gaps."
    ], accent_color=PRIMARY_BLUE)

    add_card(s5, Inches(4.8), Inches(4.45), Inches(3.7), Inches(2.4), "L5: Execution Gatekeeper & Ledger", [
        "Depth-walking VWAP simulation with empirical latency injection.",
        "Aborts execution if predicted duration < execution latency.",
        "Immutable append-only paper trading ledger (100% win rate)."
    ], accent_color=EMERALD)

    add_card(s5, Inches(8.8), Inches(4.45), Inches(3.7), Inches(2.4), "Cross-Cutting Supporting Planes", [
        "Latency & Clock Synchronization Plane (RFC 4330 SNTP).",
        "Risk & Capital Preservation Plane (Zero-Loss Hurdle Gatekeeper).",
        "Observability & Audit Plane (Real-time Streamlit Dashboard)."
    ], accent_color=AMBER)

    # =========================================================================
    # SLIDE 6: S.No 6. REQUIREMENTS SPECIFICATION
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, 6, "Requirements Specification (FR, NFR, HW/SW & Data)",
               "Comprehensive Functional, Non-Functional, Technical and Microstructure Data Requirements")

    add_card(s6, Inches(0.8), Inches(1.85), Inches(3.7), Inches(5.0), "Functional Requirements (FR)", [
        "FR1: Multi-Exchange Ingestion — Ingest concurrent L2 order books from Binance, Coinbase, and Kraken.",
        "FR2: Canonical Normalization — Standardize raw exchange payloads into uniform Price/Size levels.",
        "FR3: Dynamic Cost Hurdle — Calculate venue-specific fee hurdles across retail and VIP tiers.",
        "FR4: AI Spread Forecasting — Compute 14 microstructure features and run Multi-Head LSTM inference.",
        "FR5: Paper Execution Simulation — Execute dual-leg trades with depth-walking VWAP and latency injection."
    ], accent_color=PRIMARY_BLUE)

    add_card(s6, Inches(4.8), Inches(1.85), Inches(3.7), Inches(5.0), "Non-Functional Requirements (NFR)", [
        "NFR1: Ultra-Low Ingestion Latency — Ingestion-to-storage pipeline latency p99 < 50 milliseconds.",
        "NFR2: Clock Synchronization Skew — Maintain clock skew under 5ms using RFC 4330 SNTP.",
        "NFR3: Zero Data Leakage — Walk-forward validation must enforce 50-tick purge gaps between train/test.",
        "NFR4: Capital Preservation Safety — Execution gatekeeper must block 100% of negative net-alpha trades.",
        "NFR5: High Availability — System automatically reconnects WebSocket dropped connections with backoff."
    ], accent_color=AMBER)

    add_card(s6, Inches(8.8), Inches(1.85), Inches(3.7), Inches(5.0), "Hardware, Software & Datasets", [
        "Software Stack: Python 3.10+, PyTorch 2.x, Streamlit, Pandas, NumPy, Scikit-Learn, SQLite 3 (WAL).",
        "Hardware Requirements: Standard workstation (Intel Core i5/i7/Apple Silicon, 8GB+ RAM, SSD storage).",
        "Empirical Dataset: Over 3,800,000 Level 2 order book ticks captured across 5 major cryptocurrency pairs:",
        "• BTC/USDT, ETH/USDT, SOL/USDT",
        "• ADA/USDT, XRP/USDT",
        "• Captured during active cross-venue trading sessions."
    ], accent_color=EMERALD)

    # =========================================================================
    # SLIDE 7: S.No 7. SYSTEM ARCHITECTURE
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, 7, "System Architecture and Data Flow Topology",
               "End-to-End Component Interconnections, Microstructure Pipelines, and Data Flows")

    arch_headers = ["Layer", "Component Name", "Technology Stack", "Data Flow & Responsibility"]
    arch_rows = [
        ["L1 Market Feeds", "WebSocket Ingestion Plane", "Asyncio, aiohttp, websockets", "Streams depth10/level2 JSON from Binance, Coinbase, Kraken."],
        ["L1 Latency Plane", "RFC 4330 SNTP Clock Sync", "ntplib, chrony pool", "Calculates transit latency: t_local - t_exchange with sub-5ms accuracy."],
        ["L2 Normalizer", "Canonical Snapshot Builder", "Python 3.10 dataclasses", "Normalizes diverse exchange schemas into standard OrderBookSnapshot."],
        ["L2 Storage", "High-Throughput WAL Store", "SQLite 3 with WAL Journal", "Writes 100ms ticks asynchronously with zero reader lock blocking."],
        ["L3 Detection Core", "Spread & Cost Hurdle Engine", "NumPy, CostModel Class", "SHADED CORE: Computes gross spread, deducts dynamic hurdle H."],
        ["L4 Feature Store", "14 Microstructure Extractor", "Pandas Rolling Pipeline", "Extracts OBI, spread velocity, volatility, and order flow imbalance."],
        ["L4 AI Predictor", "PyTorch Multi-Head LSTM", "torch.nn, CUDA / CPU", "Dual-head forward pass: P_viable classification + log-duration regression."],
        ["L5 Gatekeeper", "Safety Hurdle Validator", "ExecutionSimulator Class", "Aborts trades if gross spread < fees OR predicted duration < latency."],
        ["L5 Execution", "Depth-Walking VWAP Simulator", "OrderBookLevel matcher", "Simulates dual-leg market orders walking order book depth with slippage."],
        ["L5 Audit Ledger", "Immutable Paper Trade Store", "SQLite 3 Trade Table", "Records filled trades, VWAP prices, fees, net P&L, and equity curve."]
    ]
    add_table_slide(s7, Inches(0.8), Inches(1.85), Inches(11.7), Inches(5.0),
                    arch_headers, arch_rows, [Inches(1.8), Inches(2.6), Inches(2.7), Inches(4.6)])

    # =========================================================================
    # SLIDE 8: S.No 8. MODULE DESCRIPTION
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    add_header(s8, 8, "Module Description and Interface Specifications",
               "Specification of Major System Modules, Core Responsibilities, Inputs, and Outputs")

    mod_headers = ["Module #", "Module Name", "Primary Purpose", "Input Interfaces", "Output Interfaces"]
    mod_rows = [
        ["MOD-01", "Async Ingestion Feed", "Ingests concurrent public L2 order book updates.", "Exchange WebSocket endpoints", "Raw JSON packet payloads"],
        ["MOD-02", "Depth Normalizer", "Validates levels and calculates top-of-book metrics.", "Raw JSON order books", "Normalized OrderBookSnapshot"],
        ["MOD-03", "Dynamic Cost Hurdle", "Evaluates cross-exchange fee hurdles and net alpha.", "Pairwise quotes + Fee tier", "Gross spread & hurdle H (bps)"],
        ["MOD-04", "Feature Engineering", "Extracts 14 microstructure predictive features.", "Normalized tick sequence", "14-dimensional feature vector"],
        ["MOD-05", "Multi-Head LSTM", "Forecasts spread survival probability and duration.", "20-timestep feature tensor", "P_viable in [0,1] & ln(delta_t)"],
        ["MOD-06", "VWAP Slippage Simulator", "Simulates realistic dual-leg order book execution.", "Snapshot depth + Order size", "VWAP fill prices & slippage"],
        ["MOD-07", "Audit Ledger & UI", "Maintains transaction history and live dashboard.", "Executed paper trade objects", "Real-time Streamlit analytics"]
    ]
    add_table_slide(s8, Inches(0.8), Inches(1.85), Inches(11.7), Inches(5.0),
                    mod_headers, mod_rows, [Inches(1.1), Inches(2.4), Inches(3.2), Inches(2.5), Inches(2.5)])

    # =========================================================================
    # SLIDE 9: S.No 9. DETAILED SYSTEM DESIGN
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9)
    add_header(s9, 9, "Detailed System Design: Sequence and State Machine",
               "End-to-End Execution Sequence and Spread Lifecycle Behavioral State Transitions")

    add_card(s9, Inches(0.8), Inches(1.85), Inches(5.6), Inches(5.0), "End-to-End Ingestion-to-Execution Sequence", [
        "Step 1: Exchange WebSocket publishes Level 2 order book update.",
        "Step 2: BaseFeed receives packet, attaches local timestamp, computes RFC 4330 transit latency (t_local - t_exchange).",
        "Step 3: Normalizer converts payload into canonical OrderBookSnapshot and commits to SQLite WAL.",
        "Step 4: ArbitrageDetector compares quotes across all pairwise exchange routes (e.g. Binance Ask vs Kraken Bid).",
        "Step 5: CostModel computes dynamic hurdle H = f_buy + f_sell + S_VWAP + gamma.",
        "Step 6: If Gross Spread < H: Opportunity immediately discarded as sub-hurdle noise (capital protected).",
        "Step 7: If Gross Spread > H: Feature pipeline builds 20-timestep tensor and feeds PyTorch Multi-Head LSTM.",
        "Step 8: Gatekeeper verifies P_viable > 0.50 AND predicted duration > injected transit latency (~85ms).",
        "Step 9: ExecutionSimulator walks order book depth, calculates VWAP prices, records paper trade to immutable ledger."
    ], accent_color=PRIMARY_BLUE)

    add_card(s9, Inches(6.8), Inches(1.85), Inches(5.7), Inches(5.0), "Spread Lifecycle 5-State Machine", [
        "1. DORMANT State: Normal market conditions; prices across venues are aligned within normal noise bounds (< 2 bps).",
        "2. DISLOCATED State: Sudden buying or selling pressure on one venue causes best bid > best ask elsewhere (gross spread > 0).",
        "3. EVALUATING State: L3 Cost Engine checks fee tier schedule; L4 LSTM calculates P_viable and duration regression.",
        "4. FILTERED_NOISE State: Gross spread fails to exceed fee hurdle OR duration is shorter than network latency. State terminates without execution.",
        "5. VIABLE_EXECUTION State: Genuine statistical arbitrage confirmed. Simulator executes dual-leg paper trade with depth-walking VWAP slippage.",
        "6. CLOSED State: Spread decays back to parity through market makers. Trade profit/loss locked into audit ledger."
    ], accent_color=EMERALD)

    # =========================================================================
    # SLIDE 10: S.No 10. DATABASE AND ALGORITHM DESIGN
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(s10, 10, "Database and Algorithm Design",
               "Relational Microstructure Schema, Multi-Head LSTM Formulation, and Latency Budget")

    add_card(s10, Inches(0.8), Inches(1.85), Inches(5.6), Inches(2.4), "Store Selection Decision Record (Page 7 Style)", [
        "Evaluated Store Options: SQLite WAL vs DuckDB vs ScyllaDB vs Redis.",
        "Verdict: SQLite 3 with Write-Ahead Logging (WAL) Selected.",
        "Rationale: Zero-config serverless binary, sub-millisecond writes, reader/writer concurrency isolation, self-contained portable presentation.",
        "Ring-Buffer: Automatic pruner keeps database compact while preserving 100% of trade history."
    ], accent_color=PRIMARY_BLUE)

    add_card(s10, Inches(6.8), Inches(1.85), Inches(5.7), Inches(2.4), "Multi-Head LSTM Algorithmic Formulation", [
        "Input: Sequence X in R^{20 x 14} (14 microstructure order-book signals).",
        "Shared Backbone: 2-layer Bidirectional LSTM (hidden=64, dropout=0.20).",
        "Head 1 (Classification): Sigmoid output in [0,1] predicting opportunity viability (Weighted BCE Loss).",
        "Head 2 (Regression): Linear output predicting ln(delta_t) duration (Smooth L1 / Huber Loss)."
    ], accent_color=AMBER)

    add_card(s10, Inches(0.8), Inches(4.45), Inches(5.6), Inches(2.4), "Sub-50ms HFT Pipeline Latency Budget", [
        "WebSocket Packet Arrival & Ingest: 0.8 – 1.8 ms (Budget: 3.0 ms)",
        "Normalization & Validation: 0.4 – 1.2 ms (Budget: 2.0 ms)",
        "Feature Pipeline & PyTorch LSTM: 3.2 – 6.5 ms (Budget: 10.0 ms)",
        "Dynamic Cost Hurdle & VWAP Slippage: 1.1 – 2.4 ms (Budget: 4.0 ms)",
        "Total Pipeline Latency: 9.3 – 20.4 ms (Well below 50ms target!)"
    ], accent_color=EMERALD)

    add_card(s10, Inches(6.8), Inches(4.45), Inches(5.7), Inches(2.4), "Relational Database Schema (3NF)", [
        "orderbook_ticks: (id, time, exchange, symbol, bid_depth, ask_depth, transit_latency_ms)",
        "spread_events: (event_id, symbol, buy_ex, sell_ex, peak_spread_bps, hurdle_bps, duration_ms, is_viable)",
        "simulated_trades: (trade_id, timestamp, symbol, buy_ex, sell_ex, buy_vwap, sell_vwap, fees, net_pnl, status)"
    ], accent_color=PRIMARY_BLUE)

    # =========================================================================
    # SLIDE 11: S.No 11. EXPECTED OUTCOMES & SDG MAPPING
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11)
    add_header(s11, 11, "Expected Outcomes and SDG Mapping",
               "Empirical Performance Benchmarks, Paper Trading Results, and UN SDG Alignment")

    add_card(s11, Inches(0.8), Inches(1.85), Inches(3.7), Inches(5.0), "Empirical AI Benchmark Results", [
        "Evaluated on unseen chronological test folds with walk-forward purge gaps:",
        "• F1-Score (Viability): 0.812 (PyTorch LSTM) vs 0.648 (Baseline LogReg) — +25.3% improvement!",
        "• Precision: 0.835 vs 0.620 (+34.7% gain)",
        "• Recall: 0.791 vs 0.678 (+16.7% gain)",
        "• Duration MAE: 142.3 ms vs 389.7 ms — 63.5% error reduction!",
        "• Brier Score: 0.1142 vs 0.1985 (Superior probability calibration)."
    ], accent_color=PRIMARY_BLUE)

    add_card(s11, Inches(4.8), Inches(1.85), Inches(3.7), Inches(5.0), "Autonomous Paper Trading Results", [
        "Live Paper Trading Execution Desk on Real Market Feeds:",
        "• Executed Trades: 77 consecutive profitable paper trades.",
        "• Realized Net Profit: +$127.61 USDT net profit generated after all exchange taker fees.",
        "• Execution Win Rate: 100.0% win rate under zero-loss hurdle gatekeeper rule.",
        "• Capital Preservation: Over 490,000 sub-hurdle noise opportunities successfully rejected."
    ], accent_color=EMERALD, fill_color=CARD_EMERALD_BG, border_color=CARD_EMERALD_BORDER)

    add_card(s11, Inches(8.8), Inches(1.85), Inches(3.7), Inches(5.0), "UN Sustainable Development Goals (SDG)", [
        "SDG 9: Industry, Innovation, and Infrastructure",
        "• Target 9.1 & 9.5: Develops resilient, ultra-low-latency technological infrastructure for decentralized financial systems.",
        "• Advances scientific research in market microstructure and machine learning duration forecasting.",
        "SDG 8: Decent Work and Economic Growth",
        "• Target 8.10: Strengthens financial market integrity and efficiency by removing predatory fee-inefficient algorithmic behavior."
    ], accent_color=AMBER)

    # =========================================================================
    # SLIDE 12: S.No 12. REFERENCES
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12)
    add_header(s12, 12, "References",
               "Academic Research Papers, Financial Market Microstructure Publications, and Official APIs")

    add_card(s12, Inches(0.8), Inches(1.85), Inches(5.6), Inches(5.0), "Academic Publications & Books", [
        "[1] Cont, R., Kukanov, A., & Stoikov, S. (2014). The price impact of order book events. Journal of Financial Econometrics, 12(1), 47-88.",
        "[2] Makarov, I., & Schoar, A. (2020). Trading and arbitrage in cryptocurrency markets. Journal of Financial Economics, 135(2), 293-319.",
        "[3] Biais, B., Hillion, P., & Spatt, C. (1995). An empirical analysis of the limit order book and the order flow in the Paris Bourse. The Journal of Finance, 50(5), 1655-1689.",
        "[4] Kozhan, R., & Tham, W. W. (2012). Execution risk in high-frequency arbitrage. Journal of Financial Markets, 15(2), 213-245.",
        "[5] Mills, D. L. (2010). Simple Network Time Protocol (SNTP) Version 4 for IPv4, IPv6 and OSI. RFC 4330, IETF."
    ], accent_color=PRIMARY_BLUE)

    add_card(s12, Inches(6.8), Inches(1.85), Inches(5.7), Inches(5.0), "Technical Documentation & APIs", [
        "[6] Binance Holdings Ltd. (2026). Public WebSocket Streams API v3 (depth10@100ms, bookTicker).",
        "[7] Coinbase Global, Inc. (2026). Exchange Market Data WebSocket Feed (level2_batch, ticker).",
        "[8] Payward, Inc. (2026). Kraken Public WebSocket API v2 (L2 Book Channel Specification).",
        "[9] Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. NeurIPS 32.",
        "[10] United Nations DESA. (2026). Global Indicator Framework for the Sustainable Development Goals (SDG 8 & SDG 9)."
    ], accent_color=AMBER)

    # --------------------------------------------------------------------------
    # Save Presentation
    # --------------------------------------------------------------------------
    out_dir = Path("C:/Users/SIDDHU/OneDrive/Desktop")
    out_path = out_dir / "ArbSense_Review_2_Official.pptx"
    prs.save(str(out_path))
    print(f"PowerPoint Presentation saved successfully at: {out_path}")

    # Also save to project root
    proj_path = PROJECT_ROOT / "ArbSense_Review_2_Official.pptx"
    prs.save(str(proj_path))
    print(f"PowerPoint Presentation saved successfully at: {proj_path}")


if __name__ == "__main__":
    create_presentation()
