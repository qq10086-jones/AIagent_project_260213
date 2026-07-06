"""News-catalyzed theme-leader candidate engine (ADR-0009).

Hybrid layer over the price screener: news theme-heat × ticker themes →
per-candidate catalyst → theme-leader rank → hybrid rerank. Output stays an
uncalibrated_research_score (Rule 8.3 / 9.4); it never alters the P12-05
calibration track.
"""
