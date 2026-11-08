#!/usr/bin/env python3
"""
Sector-Specific Quantitative & Analytical Frameworks
===================================================
Tailors LLM analytical reasoning, market-driving factor considerations,
valuation metric mandates, and sampling temperatures on a sectorial basis.

Prevents cross-sector valuation failures:
  - Banking/Insurance: Eliminates EV/EBITDA & Net Debt distortions; prioritizes P/TBV, NIM, ROTCE.
  - Materials & Mining: Counters Peter Lynch cyclical P/E trap; links to spot commodity curves & AISC.
  - Real Estate: Substitutes GAAP EPS with FFO/AFFO multiples and Cap Rate spreads.
  - Technology: Focuses on hyperscaler CapEx, wafer fab utilization, and gross margin resilience.
  - Energy & Tankers: Focuses on spot TCE day-rates, refinery crack spreads, and FCF yields.

Calibrates model temperature by sector risk profile:
  - Financials / REITs / Utilities: 0.10 (strict balance-sheet and arithmetic discipline)
  - Materials / Energy: 0.15 (commodity cost curve alignment)
  - Industrials / Consumer Staples: 0.18 (order backlog & margin defense)
  - Consumer Discretionary / Communications: 0.22 (consumer sentiment & digital ad auctions)
  - Information Technology / Biotech: 0.25 (complex multi-catalyst product & clinical roadmaps)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# Canonical GICS Sector Reference and In-Depth Sector Blueprints
SECTOR_FRAMEWORKS: Dict[str, Dict[str, Any]] = {
    "Financials": {
        "sector": "Financials",
        "benchmark_etf": "XLF",
        "recommended_temperature": 0.10,
        "market_driving_factors": [
            "Yield curve slope (10Y-2Y & 10Y-3M spreads): steepening curve expands Net Interest Margin (NIM); inversion compresses loan spreads and induces deposit flight.",
            "Central bank policy rates (Fed/ECB) & terminal rate path: impacts deposit beta repricing speed vs loan asset yields.",
            "Credit cycle & asset quality: Provisions for Credit Losses (PCL), non-performing asset (NPA) ratios, and Commercial Real Estate (CRE) / retail loan charge-offs.",
            "Regulatory capital adequacy & capital return: Basel III Endgame Tier-1 CET1 ratios vs minimums, annual stress test performance, and share buyback / dividend sustainability.",
            "Fee revenue diversification: investment banking M&A advisory, wealth management AUM inflows, and trading desk market volatility capture."
        ],
        "valuation_mandate": (
            "CRITICAL VALUATION RULE FOR FINANCIALS: NEVER use EV/EBITDA, Free Cash Flow, Enterprise Value, or Debt/Equity "
            "(customer deposits are operational liabilities, not corporate debt; capital is the firm's inventory). "
            "Exclusively evaluate: Price-to-Tangible-Book (P/TBV), Return on Tangible Common Equity (ROTCE), Return on Assets (ROA), "
            "Net Interest Margin (NIM), Efficiency Ratio (<55% is high quality), and CET1 Capital Buffer."
        ),
        "cyclical_and_event_traps": (
            "Avoid penalizing sound banks for high gross leverage. Watch for unrealized mark-to-market losses on Held-to-Maturity (HTM) "
            "bond portfolios, sudden deposit concentration runs, and surging CRE refinancing defaults."
        )
    },

    "Materials": {
        "sector": "Materials",
        "benchmark_etf": "XLB",
        "recommended_temperature": 0.15,
        "market_driving_factors": [
            "Underlying commodity spot & futures curves: Gold ($XAU/USD), Silver, Copper ($HG), Iron Ore, Lithium, and Containerboard pricing.",
            "US Dollar Index (DXY) inverse transmission: Dollar strength suppresses global commodity realizations; dollar weakness fuels spot commodity rallies.",
            "All-In Sustaining Costs (AISC): per ounce/tonne operating costs (diesel fuel, grid power, explosives, labor, cyanide/chemical reagents).",
            "Global manufacturing & infrastructure demand: China industrial PMI, stimulus measures, and Western reshoring/defense infrastructure buildouts.",
            "Resource nationalism & regulatory permits: mining royalty tax hikes, environmental permitting delays, and sovereign jurisdiction risk."
        ],
        "valuation_mandate": (
            "PETER LYNCH CYCLICAL P/E TRAP: Do NOT treat optically low P/E (5x-8x) as cheap when commodity spot prices are at cyclical peaks "
            "— this often signals peak earnings right before a supply glut. Conversely, elevated or negative P/E at the trough of a commodity "
            "cycle often represents the generational entry setup. Base valuation on spot commodity cash margins (Spot Price minus AISC), "
            "proven & probable reserve replacement life, and Free Cash Flow yield at current strip prices."
        ),
        "cyclical_and_event_traps": (
            "Beware of massive Greenfield CapEx announcements at cyclical highs (capital destruction). Watch for cost inflation compressing "
            "cash margins even as headline revenues rise, and mine grade degradation."
        )
    },

    "Information Technology": {
        "sector": "Information Technology",
        "benchmark_etf": "XLK",
        "recommended_temperature": 0.25,
        "market_driving_factors": [
            "Hyperscaler CapEx guidance: capital spending allocations by Microsoft, Alphabet, Amazon, and Meta into AI accelerators, data centers, and optical interconnects.",
            "Semiconductor supply chain health: foundry wafer fab utilization (TSMC), advanced packaging availability (CoWoS, HBM memory supply), and Wafer Fab Equipment (WFE) billings.",
            "Enterprise software spending cycles: corporate IT budget growth, deal closing velocity, cloud migration runway, and software seat expansion.",
            "AI monetization proof points: measurable corporate productivity gains and ROI on AI software vs speculative compute deployment.",
            "Discount rate duration sensitivity: high-multiple secular tech has long-duration cash flows, vulnerable to spikes in 10-Year Real Treasury Yields."
        ],
        "valuation_mandate": (
            "High forward P/E or EV/Sales multiples are permissible ONLY if validated by gross margin expansion (>60% indicates durable pricing power), "
            "secular TAM growth, Rule of 40 performance (Revenue Growth % + Free Cash Flow Margin % >= 40%), and Net Revenue Retention (NRR > 115%). "
            "Differentiate between cyclical hardware/semiconductor vendors and mission-critical recurring software compounders."
        ),
        "cyclical_and_event_traps": (
            "Beware of customer inventory digestion following multi-quarter chip double-ordering. Watch for decelerating ARR growth masked by "
            "one-off professional services revenue, and multiple compression triggered by real rate spikes."
        )
    },

    "Energy": {
        "sector": "Energy",
        "benchmark_etf": "XLE",
        "recommended_temperature": 0.15,
        "market_driving_factors": [
            "OPEC+ production quotas & spare capacity cushion: disciplined supply quotas vs quota cheating and baseline re-allocations.",
            "Global product crack spreads (3-2-1): refinery processing margins, distillate (diesel/jet fuel) inventory deficits, and gasoline seasonal crack margins.",
            "Marine freight logistics & charter rates: VLCC, Suezmax, LNG, and LPG spot Time-Charter Equivalent (TCE) day-rates, and canal transit choke-points (Red Sea, Hormuz, Panama).",
            "Natural gas seasonal demand: Henry Hub / European TTF spot prices, Heating Degree Days (HDD) / Cooling Degree Days (CDD), and LNG export terminal uptime.",
            "Strategic Petroleum Reserve (SPR) policy: government reserve releases or replenishment tenders."
        ],
        "valuation_mandate": (
            "Evaluate operational break-even price per barrel/BOE vs current strip pricing. Prioritize Free Cash Flow Yield (>10% is strong), "
            "balance-sheet net debt to EBITDA (<1.0x), disciplined low reinvestment rates (CapEx < 50% of Operating Cash Flow), and "
            "base dividend safety buffer against sharp spot oil drops."
        ),
        "cyclical_and_event_traps": (
            "Do NOT extrapolate seasonal spikes in spot tanker day-rates or winter gas prices into perpetual forward cash flows. "
            "Watch for rapid non-OPEC shale supply responses capping multi-year crude upside."
        )
    },

    "Health Care": {
        "sector": "Health Care",
        "benchmark_etf": "XLV",
        "recommended_temperature": 0.25,
        "market_driving_factors": [
            "Regulatory action dates: FDA Prescription Drug User Fee Act (PDUFA) decisions, Advisory Committee (AdCom) panels, and EMA European approvals.",
            "Clinical trial pipeline readouts: Phase 2/3 primary endpoint statistical significance (p-values), hazard ratios, and adverse event tolerability profiles.",
            "Patent cliff & Loss of Exclusivity (LOE): expiration dates on flagship blockbuster therapies vs replacement revenue from newly approved indications.",
            "Legislative pricing & reimbursement: US Medicare drug price negotiations (Inflation Reduction Act), PBM formulary coverage exclusions, and biosimilar erosion.",
            "M&A and strategic licensing: large-cap pharma cash deployment to replenish depleted pipelines before patent cliffs."
        ],
        "valuation_mandate": (
            "Segment commercialized cash-generating biopharma (evaluate pipeline risk-adjusted NPV, patent cliff timeline, and ROIC) from "
            "pre-commercial clinical biotech (evaluate net cash runway in months, burn rate, and Probability-of-Success [PoS] adjusted DCF). "
            "For medical devices and managed care, prioritize hospital surgical procedure volume, medical loss ratios (MLR), and pricing power."
        ),
        "cyclical_and_event_traps": (
            "Beware of single-asset clinical biotech binary gap risk (failure can cause 60-80% overnight drawdowns). Watch for sudden "
            "dilutive secondary equity offerings following positive data headlines."
        )
    },

    "Real Estate": {
        "sector": "Real Estate",
        "benchmark_etf": "XLRE",
        "recommended_temperature": 0.10,
        "market_driving_factors": [
            "Cost of capital vs Cap Rates: spread between property capitalization rates and the 10-Year Treasury Yield / BBB credit spreads.",
            "Commercial Real Estate (CRE) debt refinancing wall: maturing low-coupon mortgages having to refinance at elevated benchmark interest rates.",
            "Sub-sector demand divergence: AI data centers, cell towers, and modern multi-tenant industrial logistics vs secularly challenged legacy commodity office spaces.",
            "Occupancy rates & leasing spreads: blended re-leasing spreads (cash rent growth on expiring leases) and tenant lease term (WALT).",
            "Inflation hedge structure: lease structures with annual CPI indexation or fixed contractual escalators vs fixed-income leases."
        ],
        "valuation_mandate": (
            "CRITICAL VALUATION RULE FOR REITS: NEVER evaluate standard GAAP EPS or GAAP P/E multiples (depreciation of buildings is a non-cash "
            "accounting charge that distorts economic cash generation). Evaluate exclusively: Funds From Operations (FFO) / Adjusted FFO (AFFO) "
            "multiples, Net Asset Value (NAV) premium/discount, Debt-to-EBITDA (<6.0x), and AFFO Dividend Payout Ratio (<85% for safety)."
        ),
        "cyclical_and_event_traps": (
            "Avoid high headline dividend yields that exceed 100% of AFFO (optical yield traps preceding dividend cuts). "
            "Watch for tenant bankruptcies and non-renewals eroding cash NOI."
        )
    },

    "Industrials": {
        "sector": "Industrials",
        "benchmark_etf": "XLI",
        "recommended_temperature": 0.18,
        "market_driving_factors": [
            "Global defense spending commitments: NATO 2% GDP targets, US DoD modernization procurement backlogs, and Foreign Military Sales (FMS).",
            "Commercial aerospace super-cycle: multi-year commercial airliner backlogs (Boeing / Airbus), aftermarket commercial spares & maintenance (MRO) demand, and supply chain casting/forging bottlenecks.",
            "Fixed non-residential investment & reshoring: federal stimulus execution (CHIPS Act, Inflation Reduction Act, Infrastructure Bill) and domestic factory construction.",
            "Freight & goods transportation volume: Cass Freight Index, rail carloads, intermodal container volumes, and trucking spot rates."
        ],
        "valuation_mandate": (
            "Prioritize Book-to-Bill ratio (>1.0 indicates expanding forward demand), multi-year funded order backlog length (years of revenue covered), "
            "operating margin conversion, and Free Cash Flow conversion (>90% of Net Income). Differentiate high-moat aftermarket monopolies from low-margin sub-contractors."
        ),
        "cyclical_and_event_traps": (
            "Beware of fixed-price defense or construction contracts with inflation exposure (cost overruns absorbed by the contractor). "
            "Watch for supply chain component de-stocking triggering sharp cyclical inventory cuts."
        )
    },

    "Consumer Discretionary": {
        "sector": "Consumer Discretionary",
        "benchmark_etf": "XLY",
        "recommended_temperature": 0.22,
        "market_driving_factors": [
            "Real disposable household income vs core inflation: real wage purchasing power, personal savings rates, and consumer confidence.",
            "Consumer credit health: credit card balances, personal loan charge-offs, and auto loan 30+ day delinquencies.",
            "Store traffic & comp sales: comparable store sales growth (decomposed into foot traffic vs average transaction ticket size).",
            "Inventory-to-sales ratios: excessive inventory prompting gross-margin destroying promotional markdowns vs lean inventory preserving full-price sell-through.",
            "Tariff and freight exposure: ocean shipping container spot rates and potential import tariffs on Asian manufacturing."
        ],
        "valuation_mandate": (
            "Evaluate same-store sales growth, gross margin stability against promotional discounting, Customer Lifetime Value (LTV) vs "
            "Customer Acquisition Cost (CAC), and inventory turns. Differentiate luxury/aspirational goods (inelastic high-income buyer) "
            "from mass-market discretionary (highly elastic, prone to trade-down)."
        ),
        "cyclical_and_event_traps": (
            "Beware of rapid fashion/trend obsolescence and inventory build-ups heading into economic slowdowns. "
            "Watch for consumer credit exhaustion compressing high-ticket big-box purchases."
        )
    },

    "Consumer Staples": {
        "sector": "Consumer Staples",
        "benchmark_etf": "XLP",
        "recommended_temperature": 0.18,
        "market_driving_factors": [
            "Pricing power vs unit volume elasticity: ability to push retail price increases without triggering unit volume degradation.",
            "Private label (store brand) substitution: consumer trade-down rates during sticky food and household inflation.",
            "Agricultural & packaging commodity inputs: cost deflators/inflators across grain, edible oils, coffee, aluminum cans, and resin/plastics.",
            "Global retail shelf leverage: distribution shelf-space dominance and direct-store-delivery (DSD) network moats.",
            "Bond-proxy yield competition: defensive dividend yield attractiveness vs short-term risk-free Treasury yields."
        ],
        "valuation_mandate": (
            "Scrutinize organic revenue growth by decomposing into price/mix contribution vs pure unit volume growth (price-driven growth "
            "without volume signals eventual demand destruction). Prioritize Return on Invested Capital (ROIC > 15%), consistent Free Cash Flow "
            "conversion, and dividend coverage safety."
        ),
        "cyclical_and_event_traps": (
            "Beware of companies hiding unit volume declines with aggressive price increases ('shrinkflation' backlash). "
            "Watch for multiple contraction when bond yields surge, as staples are often traded as fixed-income proxies."
        )
    },

    "Communication Services": {
        "sector": "Communication Services",
        "benchmark_etf": "XLC",
        "recommended_temperature": 0.22,
        "market_driving_factors": [
            "Digital advertising macro cycle: brand advertising budgets vs direct-response performance ad-spend, and auction pricing (CPM / CPC trends).",
            "Streaming entertainment unit economics: subscriber additions/churn, Average Revenue Per User (ARPU), content production amortization vs cash spend.",
            "Wireless telecom competition: postpaid phone subscriber net adds, customer churn rates, and capital expenditure intensity on 5G/fiber networks.",
            "Regulatory & antitrust scrutiny: app store fee structures, ad-tech antitrust lawsuits, and AI content search disruption."
        ],
        "valuation_mandate": (
            "Differentiate high-margin digital ad platforms (high ROIC, dominant network effects, high FCF conversion) from capital-intensive "
            "telecom utilities (high leverage, high CapEx, modest return on capital). Evaluate ARPU growth, subscriber acquisition payback period, "
            "and content cash amortisation vs GAAP reporting."
        ),
        "cyclical_and_event_traps": (
            "Watch for streaming content cash burn masked by multi-year accounting amortization schedules. "
            "Beware of digital ad deceleration preceding broad economic downturns."
        )
    },

    "Utilities": {
        "sector": "Utilities",
        "benchmark_etf": "XLU",
        "recommended_temperature": 0.10,
        "market_driving_factors": [
            "Surging AI & industrial power demand: unprecedented electricity load growth driven by hyperscaler AI data centers, industrial onshoring, and vehicle electrification.",
            "Clean energy Power Purchase Agreements (PPAs): long-term contracted power agreements with tech hyperscalers at premium rates (nuclear, solar, gas).",
            "State Public Utility Commission (PUC) rate cases: allowed Return on Equity (ROE) approved on rate-base investments and cost recovery timeliness.",
            "Interest rate transmission: utilities operate with high balance-sheet debt; refinancing low-coupon bonds increases financing expense, while dividend yields compete directly with Treasuries.",
            "Weather volatility & storm risk: severe summer peak cooling or winter freezing events impacting spot generation costs and grid liability."
        ],
        "valuation_mandate": (
            "Analyze utilities as regulated bond-proxies with secular electricity demand call options. Prioritize Rate-Base Capital Growth (CAGR > 6-8%), "
            "regulatory allowed ROE (typically 9.5-10.5%), balance-sheet debt/EBITDA (<5.5x), and dividend payout ratio safety (60-75% of regulated earnings)."
        ),
        "cyclical_and_event_traps": (
            "Watch for adverse regulatory decisions (denial of rate base cost recovery), catastrophic wildfire / storm liability exposure, "
            "and spiking benchmark yields compressing utility valuation multiples."
        )
    }
}

# Alias mapping for diverse sector string inputs (from SEC, Danelfin, Gloomberb, Yahoo Finance)
SECTOR_ALIASES: Dict[str, str] = {
    # Tech
    "TECHNOLOGY": "Information Technology",
    "TECH": "Information Technology",
    "SEMICONDUCTOR": "Information Technology",
    "SEMICONDUCTORS": "Information Technology",
    "SOFTWARE": "Information Technology",
    "IT": "Information Technology",
    # Financials
    "BANK": "Financials",
    "BANKS": "Financials",
    "BANKING": "Financials",
    "FINANCIAL": "Financials",
    "FINANCIAL SERVICES": "Financials",
    "INSURANCE": "Financials",
    # Materials
    "BASIC MATERIALS": "Materials",
    "MINING": "Materials",
    "METALS & MINING": "Materials",
    "METALS": "Materials",
    "GOLD": "Materials",
    "CHEMICALS": "Materials",
    # Real Estate
    "REIT": "Real Estate",
    "REITS": "Real Estate",
    "REALESTATE": "Real Estate",
    # Energy
    "OIL & GAS": "Energy",
    "OIL": "Energy",
    "SHIPPING": "Energy",
    "MARINE": "Energy",
    # Health Care
    "HEALTHCARE": "Health Care",
    "BIOTECH": "Health Care",
    "BIOTECHNOLOGY": "Health Care",
    "PHARMACEUTICALS": "Health Care",
    "PHARMA": "Health Care",
    # Industrials
    "INDUSTRIAL": "Industrials",
    "AEROSPACE": "Industrials",
    "DEFENSE": "Industrials",
    "CAPITAL GOODS": "Industrials",
    # Consumer
    "CONSUMER CYCLICAL": "Consumer Discretionary",
    "DISCRETIONARY": "Consumer Discretionary",
    "CONSUMER NON-CYCLICAL": "Consumer Staples",
    "STAPLES": "Consumer Staples",
    # Communications
    "COMMUNICATION": "Communication Services",
    "TELECOM": "Communication Services",
    "TELECOMMUNICATIONS": "Communication Services",
    "MEDIA": "Communication Services",
    # Utilities
    "UTILITY": "Utilities",
    "POWER": "Utilities",
}


def normalize_sector_name(raw_sector: Optional[str]) -> str:
    """Normalizes any sector string variation into the canonical GICS sector."""
    if not raw_sector:
        return "Equities"
    clean = str(raw_sector).strip()
    upper = clean.upper()

    # 1. Exact match in canonical sectors
    for canon in SECTOR_FRAMEWORKS.keys():
        if canon.upper() == upper:
            return canon

    # 2. Exact match in aliases
    if upper in SECTOR_ALIASES:
        return SECTOR_ALIASES[upper]

    # 3. Whole-word / token match or longer substring match (>= 4 chars) to avoid false positives (e.g. "IT" in "UTILITIES")
    words = [w.strip() for w in upper.replace("-", " ").replace("/", " ").replace("&", " ").split()]
    for w in words:
        if w in SECTOR_ALIASES:
            return SECTOR_ALIASES[w]

    for alias_key, canon in SECTOR_ALIASES.items():
        if len(alias_key) >= 4 and (alias_key in upper or upper in alias_key):
            return canon

    return clean.title() if clean else "Equities"


def get_sector_framework(sector_name: Optional[str]) -> Dict[str, Any]:
    """
    Returns the comprehensive analytical framework for the given sector.
    Falls back to a robust general equities framework if unknown.
    """
    canon = normalize_sector_name(sector_name)
    if canon in SECTOR_FRAMEWORKS:
        return SECTOR_FRAMEWORKS[canon]

    # Clean default fallback for unclassified equities
    return {
        "sector": canon or "Equities",
        "benchmark_etf": "SPY",
        "recommended_temperature": 0.20,
        "market_driving_factors": [
            "Macro interest rate & real yield regime (Fed/ECB stance).",
            "Broad economic cycle phase (expansion, peak, slowdown, contraction).",
            "Sector relative strength vs S&P 500 / STOXX 600 benchmarks.",
            "Quarterly earnings delivery vs analyst consensus expectations.",
            "Balance-sheet solvency, free cash flow conversion, and debt maturity schedule."
        ],
        "valuation_mandate": (
            "Evaluate company valuation relative to historical 5-year averages and industry peers. "
            "Examine whether multiple expansion or compression is justified by structural return on capital (ROIC) and earnings durability."
        ),
        "cyclical_and_event_traps": (
            "Beware of value traps where low multiples reflect structural industry obsolescence or unmanageable balance-sheet debt. "
            "Verify event timing (earnings within 3 days introduces binary gap risk)."
        )
    }


def get_sector_temperature(sector_name: Optional[str], override_temp: Optional[float] = None) -> float:
    """
    Returns the calibrated model sampling temperature for the sector.
    If an explicit override temperature is provided (e.g. from CLI --temperature),
    the override is strictly respected.
    """
    if override_temp is not None:
        return float(override_temp)

    fw = get_sector_framework(sector_name)
    return float(fw.get("recommended_temperature", 0.20))


def format_sector_system_prompt_overlay(sector_name: Optional[str], compact: bool = False) -> str:
    """
    Builds the authoritative sector-specific analytical guidance overlay
    to be incorporated into the LLM system prompt / instruction payload.
    """
    fw = get_sector_framework(sector_name)
    canon = fw["sector"]
    factors = fw["market_driving_factors"]
    val_mandate = fw["valuation_mandate"]
    traps = fw["cyclical_and_event_traps"]

    factors_bulleted = "\n".join(f"  • {f}" for f in factors)

    if compact:
        # High-density compact overlay to protect token limits on 8k models (Qwen / local LLMs)
        return (
            f"\n\n═══════════════════════════════════════════════════════════════════════════════\n"
            f"🎯 SECTOR ANALYTICAL MANDATE: {canon.upper()} (Benchmark: {fw['benchmark_etf']})\n"
            f"═══════════════════════════════════════════════════════════════════════════════\n"
            f"1. PRIMARY SECTOR DRIVERS TO EVALUATE:\n{factors_bulleted}\n"
            f"2. VALUATION METRIC MANDATE:\n{val_mandate}\n"
            f"3. CYCLICAL & EVENT TRAPS TO AVOID:\n{traps}\n"
        )

    # Full institutional analytical overlay for Nemotron / Claude / Gemini / GPT-4o
    return (
        f"\n\n═══════════════════════════════════════════════════════════════════════════════\n"
        f"🏛️ INSTITUTIONAL SECTOR ANALYTICAL FRAMEWORK: {canon.upper()}\n"
        f"Benchmark ETF: {fw['benchmark_etf']} | Calibrated Sampling Temperature: {fw['recommended_temperature']:.2f}\n"
        f"═══════════════════════════════════════════════════════════════════════════════\n"
        f"A. CRITICAL SECTOR MARKET-DRIVING FACTORS:\n"
        f"You MUST explicitly evaluate how the following sector-specific drivers impact this company's forward 20-session outlook:\n"
        f"{factors_bulleted}\n\n"
        f"B. SECTOR-SPECIFIC VALUATION METRIC MANDATE:\n"
        f"{val_mandate}\n\n"
        f"C. SECTOR CYCLICAL TRAPS & FALSE-FLAG WARNINGS:\n"
        f"{traps}\n"
        f"═══════════════════════════════════════════════════════════════════════════════\n"
    )
