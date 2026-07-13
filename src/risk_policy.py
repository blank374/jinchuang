from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


SAME_CUSTOMER_THRESHOLD = 0.92
CROSS_CUSTOMER_THRESHOLD = 0.75

RELATION_LABELS = {
    "self": "self match",
    "same_customer": "same customer / renewal",
    "cross_customer": "cross customer / suspected misuse",
    "unknown": "unknown business relation",
}

RISK_TYPE_LABELS = {
    "cross_customer_suspect": "cross-customer suspected misuse",
    "same_customer_repeat": "same-customer repeated use",
    "normal_low_risk": "normal low-risk candidate",
}

RECOMMENDED_ACTIONS = {
    "cross_customer_suspect": "Send to anti-fraud review; verify identity and loan context before approval.",
    "same_customer_repeat": "Send to standard operations/compliance review; confirm renewal or repeated submission.",
    "normal_low_risk": "Keep as low-priority audit evidence; no blocking action is suggested.",
}


@dataclass(frozen=True)
class ThresholdPolicy:
    enabled: bool = True
    same_customer: float = SAME_CUSTOMER_THRESHOLD
    cross_customer: float = CROSS_CUSTOMER_THRESHOLD
    default: float = 0.94

    @classmethod
    def from_config(cls, config: dict) -> "ThresholdPolicy":
        retrieval = config.get("retrieval", {})
        dynamic = retrieval.get("dynamic_threshold", {})
        return cls(
            enabled=bool(dynamic.get("enabled", False)),
            same_customer=float(dynamic.get("same_customer", SAME_CUSTOMER_THRESHOLD)),
            cross_customer=float(dynamic.get("fraud", CROSS_CUSTOMER_THRESHOLD)),
            default=float(retrieval.get("similarity_threshold", 0.94)),
        )


def infer_relation(query_loan_id: str, metadata: dict, loan_to_group: dict[str, str]) -> str:
    match_loan_id = str(metadata.get("loan_id", "") or metadata.get("biz_id", ""))
    if query_loan_id and query_loan_id == match_loan_id:
        return "self"

    query_group = loan_to_group.get(query_loan_id, "") if query_loan_id else ""
    match_group = str(metadata.get("similar_group", "") or "")
    if query_group and match_group and query_group == match_group:
        return "same_customer"

    return "cross_customer"


def assess_match(
    score: float,
    query_loan_id: str,
    metadata: dict,
    loan_to_group: dict[str, str],
    policy: ThresholdPolicy,
) -> dict:
    relation = infer_relation(query_loan_id, metadata, loan_to_group)
    if relation == "self":
        threshold = 1.0
    elif not policy.enabled:
        threshold = policy.default
    elif relation == "same_customer":
        threshold = policy.same_customer
    else:
        threshold = policy.cross_customer

    is_suspicious = bool(score >= threshold and relation != "self")

    if relation == "cross_customer" and is_suspicious:
        risk_type = "cross_customer_suspect"
    elif relation == "same_customer" and is_suspicious:
        risk_type = "same_customer_repeat"
    else:
        risk_type = "normal_low_risk"

    if risk_type == "cross_customer_suspect" and score >= policy.same_customer:
        risk_level = "high"
        review_priority = "urgent"
    elif risk_type in {"cross_customer_suspect", "same_customer_repeat"}:
        risk_level = "medium"
        review_priority = "standard"
    else:
        risk_level = "low"
        review_priority = "low"

    return {
        "relation": relation,
        "relation_label": RELATION_LABELS.get(relation, relation),
        "threshold_used": threshold,
        "is_suspicious": is_suspicious,
        "risk_type": risk_type,
        "risk_type_label": RISK_TYPE_LABELS[risk_type],
        "risk_level": risk_level,
        "review_priority": review_priority,
        "recommended_action": RECOMMENDED_ACTIONS[risk_type],
        "policy_version": "siglip2_stratified_v1",
    }


def summarize_risks(items: list[dict]) -> dict:
    counts = Counter(item.get("risk_type", "normal_low_risk") for item in items)
    return {
        "cross_customer_suspect": int(counts.get("cross_customer_suspect", 0)),
        "same_customer_repeat": int(counts.get("same_customer_repeat", 0)),
        "normal_low_risk": int(counts.get("normal_low_risk", 0)),
    }
