"""Financial Quality Evaluation, Compounder Bonus, and Value-Trap Gate."""

from typing import Any, Dict, List, Optional, Tuple
from quant.schemas import FinancialQualityResult, PillarStatus, InputTracking
from quant.factors import clean_float
from quant.normalization import clamp


def evaluate_financial_quality(
    gloomberb_payload: Optional[Dict[str, Any]],
    use_score_penalty: bool = False,
) -> FinancialQualityResult:
    """
    Evaluates comprehensive financial quality and solvency:
    - 1. Free Cash Flow (FCF) Margin
    - 2. Operating Margin & Pricing Power
    - 3. Net Debt / EBITDA Solvency
    - 4. Interest Coverage Ratio
    - 5. Earnings Quality (Operating Cash Flow / Net Income accrual test)
    - 6. Share Dilution Rate

    Design B (Default - Hard Eligibility Gate):
    - When a stock is certified as a Value Trap, it sets is_buy_eligible = False.
    - No arbitrary minus-15 score penalty is applied to avoid double-counting.

    Design A (Optional):
    - Applies a -15.0 score penalty to the composite score instead.
    """
    payload = gloomberb_payload or {}
    fin = payload.get("financials", {})
    fq = fin.get("financial_quality") or {}
    ratios = fin.get("key_ratios", {})
    peer_data = payload.get("peer_valuation", {})
    benchmarks = peer_data.get("direct_peer_benchmarks", [])

    fcf_margin = clean_float(fq.get("fcf_margin") if fq.get("fcf_margin") is not None else ratios.get("fcf_margin"), None)
    op_margin = clean_float(fq.get("operating_margin") if fq.get("operating_margin") is not None else ratios.get("operating_margins"), None)
    net_debt_ebitda = clean_float(fq.get("net_debt_to_ebitda") if fq.get("net_debt_to_ebitda") is not None else ratios.get("net_debt_to_ebitda"), None)
    interest_cov = clean_float(fq.get("interest_coverage") if fq.get("interest_coverage") is not None else ratios.get("interest_coverage"), None)
    earnings_quality = clean_float(fq.get("earnings_quality_ratio") if fq.get("earnings_quality_ratio") is not None else ratios.get("earnings_quality_ratio"), None)
    dilution_rate = clean_float(fq.get("share_dilution_rate"), None)
    net_debt = clean_float(fq.get("net_debt"), None)

    has_fcf = fcf_margin is not None
    has_op = op_margin is not None
    is_net_cash = (net_debt is not None and net_debt <= 0.0) or (net_debt_ebitda is not None and net_debt_ebitda <= 0.0)
    has_debt = net_debt_ebitda is not None or is_net_cash
    has_cov = interest_cov is not None or is_net_cash
    has_eq = earnings_quality is not None

    tracking = {
        "fcf_margin": InputTracking("fcf_margin", fcf_margin, has_fcf),
        "operating_margin": InputTracking("operating_margin", op_margin, has_op),
        "net_debt_to_ebitda": InputTracking("net_debt_to_ebitda", net_debt_ebitda, has_debt),
        "interest_coverage": InputTracking("interest_coverage", interest_cov, has_cov),
        "earnings_quality_ratio": InputTracking("earnings_quality_ratio", earnings_quality, has_eq),
        "share_dilution_rate": InputTracking("share_dilution_rate", dilution_rate, dilution_rate is not None),
    }

    valid_count = sum([has_fcf, has_op, has_debt, has_cov, has_eq])
    if valid_count == 0:
        return FinancialQualityResult(
            score=None,
            status=PillarStatus.UNAVAILABLE,
            metrics={},
            flags=[],
            is_value_trap=False,
            value_trap_reasons=[],
            is_buy_eligible=True,
            compounder_bonus=0.0,
            inputs=tracking,
        )

    score = 50.0
    flags = []

    # 1. FCF Margin
    if has_fcf:
        if fcf_margin >= 20.0:
            score += 15.0
        elif fcf_margin >= 10.0:
            score += 8.0
        elif fcf_margin >= 0.0:
            score += 3.0
        else:
            score -= 15.0
            flags.append(f"negative FCF margin ({fcf_margin:.1f}%) signals cash burn")

    # 2. Operating Margin
    if has_op:
        if op_margin >= 25.0:
            score += 10.0
        elif op_margin >= 12.0:
            score += 5.0
        elif op_margin < 0.0:
            score -= 10.0
            flags.append(f"negative operating margin ({op_margin:.1f}%)")

    # 3. Net Debt / EBITDA
    if is_net_cash:
        score += 15.0  # Fortress balance sheet
    elif net_debt_ebitda is not None:
        if net_debt_ebitda <= 1.5:
            score += 8.0
        elif net_debt_ebitda <= 3.0:
            score += 0.0
        elif net_debt_ebitda <= 4.5:
            score -= 10.0
            flags.append(f"elevated leverage (Net Debt/EBITDA {net_debt_ebitda:.1f}x)")
        else:
            score -= 20.0
            flags.append(f"distressed leverage (Net Debt/EBITDA {net_debt_ebitda:.1f}x)")

    # 4. Interest Coverage
    if is_net_cash:
        score += 10.0
    elif interest_cov is not None:
        if interest_cov >= 8.0:
            score += 10.0
        elif interest_cov >= 3.5:
            score += 5.0
        elif interest_cov >= 1.5:
            score -= 5.0
            flags.append(f"tight interest coverage ({interest_cov:.1f}x)")
        else:
            score -= 15.0
            flags.append(f"severe interest coverage strain ({interest_cov:.1f}x)")

    # 5. Earnings Quality (OCF / Net Income)
    if has_eq:
        if earnings_quality >= 1.1:
            score += 10.0
        elif earnings_quality >= 0.8:
            score += 4.0
        elif earnings_quality < 0.5:
            score -= 12.0
            flags.append(f"accrual divergence (OCF/Net Income {earnings_quality:.2f}x)")

    # 6. Share Dilution
    if dilution_rate is not None:
        if dilution_rate > 5.0:
            score -= 8.0
            flags.append(f"share dilution ({dilution_rate:.1f}%)")
        elif dilution_rate <= 0.0:
            score += 5.0

    final_score = clamp(score)
    status = PillarStatus.MEASURED if valid_count >= 4 else PillarStatus.PARTIAL

    # Value-Trap Detection
    fwd_pe = clean_float(ratios.get("forward_pe"), None)
    peer_pes = [clean_float(p.get("forward_pe"), None) for p in benchmarks if clean_float(p.get("forward_pe"), None) is not None]
    avg_pe = (sum(peer_pes) / len(peer_pes)) if peer_pes else None

    pe_discount = (avg_pe - fwd_pe) / avg_pe if (fwd_pe is not None and avg_pe is not None and avg_pe > 0) else 0.0
    is_cheap = (fwd_pe is not None and fwd_pe < 15.0) or (pe_discount >= 0.20)

    is_distressed_debt = (net_debt_ebitda is not None and net_debt_ebitda > 3.5)
    is_burning_cash = (fcf_margin is not None and fcf_margin < 0.0)
    is_insolvent = (interest_cov is not None and interest_cov < 2.0 and not is_net_cash)
    is_low_quality = final_score < 40.0

    is_value_trap = False
    value_trap_reasons = []

    if is_cheap and (is_distressed_debt or is_burning_cash or is_insolvent or is_low_quality):
        is_value_trap = True
        reasons = []
        if is_distressed_debt:
            reasons.append(f"heavy leverage (Net Debt/EBITDA {net_debt_ebitda:.1f}x)")
        if is_burning_cash:
            reasons.append(f"negative cash flow (FCF margin {fcf_margin:.1f}%)")
        if is_insolvent:
            reasons.append(f"strained debt coverage (interest coverage {interest_cov:.1f}x)")
        if is_low_quality and not (is_distressed_debt or is_burning_cash or is_insolvent):
            reasons.append(f"substandard financial quality ({final_score:.1f}/100)")
        value_trap_reasons.append(f"Value trap: cheap multiple masks structural distress ({', '.join(reasons)})")

    # Design B: Hard eligibility rule blocks BUY
    is_buy_eligible = not is_value_trap

    # Compounder bonus: Pristine financial quality (>= 80) earns +3.0 pts
    compounder_bonus = 3.0 if (final_score >= 80.0) else 0.0

    return FinancialQualityResult(
        score=final_score,
        status=status,
        metrics={
            "fcf_margin": fcf_margin,
            "operating_margin": op_margin,
            "net_debt_to_ebitda": net_debt_ebitda,
            "interest_coverage": interest_cov,
            "earnings_quality_ratio": earnings_quality,
            "share_dilution_rate": dilution_rate,
            "is_net_cash": is_net_cash,
        },
        flags=flags,
        is_value_trap=is_value_trap,
        value_trap_reasons=value_trap_reasons,
        is_buy_eligible=is_buy_eligible,
        compounder_bonus=compounder_bonus,
        inputs=tracking,
    )


def evaluate_value_trap(gloomberb_payload: Optional[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """Helper to quickly check value-trap status."""
    res = evaluate_financial_quality(gloomberb_payload)
    return res.is_value_trap, res.value_trap_reasons


def apply_quality_adjustment(
    raw_composite: float,
    fq_result: FinancialQualityResult,
    use_score_penalty: bool = False,
) -> float:
    """
    Applies quality adjustments to the raw composite:
    - If fq_result.compounder_bonus > 0: adds +3.0 pts.
    - If use_score_penalty (Design A) and is_value_trap: subtracts -15.0 pts.
    - Under Design B (Default): value trap does not subtract score; it acts as a Hard Gate.
    """
    score = raw_composite

    if use_score_penalty and fq_result.is_value_trap:
        score -= 15.0
    elif fq_result.compounder_bonus > 0.0:
        score += fq_result.compounder_bonus

    return clamp(score)
