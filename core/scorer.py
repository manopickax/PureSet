"""
신뢰도 스코어 계산 모듈.
수식: 100 - (결측 비율 * 40) - (중복 비율 * 30) - (이상치 비율 * 30), 최소 0
"""

from __future__ import annotations

import pandas as pd


def compute_metrics(
    df: pd.DataFrame,
    anomaly_mask: pd.Series | None,
    ignored_indices: set[int] | None = None,
) -> dict:
    """
    현재 df와 IsolationForest 결과 마스크를 받아
    결측/중복/이상치 건수 및 신뢰도 스코어를 담은 dict를 반환한다.

    ignored_indices: 사용자가 '무시' 처리한 행 인덱스 집합.
    이상치 중 무시된 항목은 n_outlier_ignored로 분리되며,
    스코어 감점은 미처리(pending) 이상치만 반영한다.
    """
    if df is None or len(df) == 0:
        return _empty_metrics()

    n_total = len(df)

    # ── 결측치: 수치 컬럼 중 하나라도 NaN인 행 ──────────────────────
    numeric_cols = _numeric_cols(df)
    if numeric_cols:
        missing_mask = df[numeric_cols].isnull().any(axis=1)
    else:
        missing_mask = pd.Series(False, index=df.index)
    n_missing = int(missing_mask.sum())

    # ── 완전 중복: timestamp 포함 전체 컬럼 기준 ────────────────────
    n_duplicated = int(df.duplicated().sum())

    # ── 이상치: 전체 / 무시됨 / 미처리 분리 ─────────────────────────
    if anomaly_mask is not None and len(anomaly_mask) == n_total:
        n_outlier = int(anomaly_mask.sum())
        if ignored_indices:
            n_outlier_ignored = int(
                anomaly_mask[anomaly_mask.index.isin(ignored_indices)].sum()
            )
        else:
            n_outlier_ignored = 0
    else:
        n_outlier = 0
        n_outlier_ignored = 0

    n_outlier_pending = n_outlier - n_outlier_ignored

    # ── 비율 계산 (0~1) — 감점은 미처리 이상치 기준 ─────────────────
    r_missing = n_missing / n_total if n_total > 0 else 0.0
    r_dup = n_duplicated / n_total if n_total > 0 else 0.0
    r_outlier = n_outlier_pending / n_total if n_total > 0 else 0.0

    # ── 스코어 ───────────────────────────────────────────────────────
    score = 100 - (r_missing * 40) - (r_dup * 30) - (r_outlier * 30)
    score = max(0.0, round(score, 1))

    return {
        "score": score,
        "n_total": n_total,
        "n_missing": n_missing,
        "n_duplicated": n_duplicated,
        "n_outlier": n_outlier,
        "n_outlier_ignored": n_outlier_ignored,
        "n_outlier_pending": n_outlier_pending,
        "r_missing": r_missing,
        "r_dup": r_dup,
        "r_outlier": r_outlier,
    }


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def _empty_metrics() -> dict:
    return {
        "score": 0.0,
        "n_total": 0,
        "n_missing": 0,
        "n_duplicated": 0,
        "n_outlier": 0,
        "r_missing": 0.0,
        "r_dup": 0.0,
        "r_outlier": 0.0,
    }
