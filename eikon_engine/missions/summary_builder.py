from __future__ import annotations

from typing import Any, Dict, List


def build_executive_summary(structured_result: Dict[str, Any], source_analysis: Dict[str, Any]) -> Dict[str, Any]:
    schema = structured_result.get("schema") if isinstance(structured_result, dict) else {}
    if not isinstance(schema, dict):
        schema = {}

    founders = list(schema.get("founders") or [])
    description = str(schema.get("description") or "").strip()
    website = str(schema.get("website") or "").strip()
    mentions = list(schema.get("recent_mentions") or [])

    top_sources = source_analysis.get("top_sources") if isinstance(source_analysis, dict) else []
    if not isinstance(top_sources, list):
        top_sources = []
    high_count = int(source_analysis.get("high_credibility_count", 0) or 0)
    medium_count = int(source_analysis.get("medium_credibility_count", 0) or 0)
    avg_source = float(source_analysis.get("average_score", 0.0) or 0.0)

    completeness_score = 0.0
    completeness_score += 0.25 if description else 0.0
    completeness_score += 0.25 if founders else 0.0
    completeness_score += 0.25 if website else 0.0
    completeness_score += 0.25 if mentions else 0.0

    source_score = min(1.0, 0.45 + (0.15 * high_count) + (0.08 * medium_count) + (0.25 * avg_source))

    founder_counts = structured_result.get("founder_occurrences") if isinstance(structured_result, dict) else {}
    consistency_score = 0.6
    if isinstance(founder_counts, dict) and founder_counts:
        repeated = [name for name, count in founder_counts.items() if int(count) >= 2]
        if repeated:
            consistency_score = 0.9
        else:
            consistency_score = 0.7

    confidence = round(min(1.0, 0.45 * completeness_score + 0.40 * source_score + 0.15 * consistency_score), 2)

    recent_activity = ""
    if mentions:
        first = mentions[0]
        if isinstance(first, dict):
            recent_activity = str(first.get("title") or first.get("url") or "")

    overview_parts = [part for part in [description, f"Website: {website}" if website else ""] if part]
    overview = " ".join(overview_parts).strip() or "Summary generated from collected sources."

    return {
        "overview": overview,
        "key_founders": founders[:6],
        "business_focus": description or "Not confidently extracted",
        "recent_activity": recent_activity or "No recent mentions validated",
        "sources_used": top_sources,
        "confidence_score": confidence,
    }
