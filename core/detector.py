"""
IsolationForest 기반 이상치 탐지 모듈.

[자동 인코딩 파이프라인]
Object/Category 컬럼을 LabelEncoder로 수치화한 임시 데이터프레임으로 IsolationForest를
학습·탐지한다. 원본 df는 절대 변경하지 않으므로 UI에서는 원본 문자열이 그대로 표시된다.

[설계 원칙]
contamination 고정값 대신 decision_function 절대 임계값을 사용한다.
데이터가 깨끗해지면 탐지 건수가 자연스럽게 0으로 수렴한다.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

_ANOMALY_SCORE_THRESHOLD = -0.05


@st.cache_data(show_spinner=False)
def detect_anomalies(df: pd.DataFrame) -> pd.Series:
    """
    수치형 + 문자열 컬럼 모두를 활용해 이상치를 탐지한다.

    - 수치형: 그대로 사용
    - Object/Category: LabelEncoder로 정수 인코딩 (원본 df 불변)
    - 결측치: 탐지 전용 임시 중앙값 대체 (원본 df 불변)
    - 임계값 기반 분류: contamination 고정 비율 폐기
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    object_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not numeric_cols and not object_cols:
        return pd.Series(False, index=df.index)

    if len(df) < 10:
        return pd.Series(False, index=df.index)

    # 탐지 전용 임시 데이터프레임 생성 (원본 df 불변)
    X = df[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=df.index)

    # 문자열/카테고리 컬럼 인코딩
    for col in object_cols:
        try:
            le = LabelEncoder()
            # NaN을 "__missing__" 문자열로 대체해 unseen 에러 방지
            col_data = df[col].astype(str).fillna("__missing__")
            X[col] = le.fit_transform(col_data)
        except Exception:
            # 인코딩 불가 컬럼은 조용히 건너뜀
            continue

    if X.empty or X.shape[1] == 0:
        return pd.Series(False, index=df.index)

    # 결측치 임시 대체 (탐지 목적에만 사용)
    for col in X.columns:
        median_val = X[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        X[col] = X[col].fillna(median_val)

    clf = IsolationForest(contamination="auto", random_state=42, n_jobs=-1)
    clf.fit(X)

    scores = clf.decision_function(X)
    return pd.Series(scores < _ANOMALY_SCORE_THRESHOLD, index=df.index)


def build_hitl_groups(df: pd.DataFrame, anomaly_mask: pd.Series) -> list[dict]:
    """
    이상치를 '패턴 그룹' 단위로 묶어 HITL UI에 넘길 구조를 반환한다.

    컬럼이 많은 데이터에서 그룹이 폭발하는 것을 막기 위해:
    - 결측치 그룹: 건수 상위 MAX_COL_GROUPS개만 표시, 나머지는 묶음 그룹으로
    - 이상치 그룹: 건수 상위 MAX_COL_GROUPS개만 표시, 나머지는 묶음 그룹으로
    """
    MAX_COL_GROUPS = 5

    groups: list[dict] = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # ── Group A: 전체 수치 컬럼이 NaN인 Garbage rows ─────────────────
    garbage_set: set[int] = set()
    if numeric_cols:
        garbage_mask = df[numeric_cols].isnull().all(axis=1)
        garbage_idx = df.index[garbage_mask].tolist()
        if garbage_idx:
            garbage_set = set(garbage_idx)
            groups.append(
                {
                    "key": "garbage_rows",
                    "label": f"쓰레기 행 (전체 컬럼 NaN) — {len(garbage_idx)}건",
                    "indices": garbage_idx,
                    "description": "모든 수치 컬럼이 비어있는 완전 무효 행입니다.",
                    "col": None,
                    "action": "delete",
                }
            )

    # ── Group B: 컬럼별 결측치 그룹 ──────────────────────────────────
    missing_candidates: list[dict] = []
    for col in numeric_cols:
        miss_idx = [i for i in df.index[df[col].isnull()].tolist() if i not in garbage_set]
        if miss_idx:
            missing_candidates.append(
                {
                    "key": f"missing_{col}",
                    "label": f"결측치 — {col} ({len(miss_idx)}건)",
                    "indices": miss_idx,
                    "description": f"'{col}' 컬럼이 비어있는 행입니다.",
                    "col": col,
                    "action": "replace_mean",
                }
            )

    missing_candidates.sort(key=lambda g: len(g["indices"]), reverse=True)
    groups.extend(missing_candidates[:MAX_COL_GROUPS])

    if len(missing_candidates) > MAX_COL_GROUPS:
        rest = missing_candidates[MAX_COL_GROUPS:]
        rest_cols = [g["col"] for g in rest]
        rest_idx = sorted({i for g in rest for i in g["indices"]})
        col_preview = ", ".join(rest_cols[:6]) + ("..." if len(rest_cols) > 6 else "")
        groups.append(
            {
                "key": "missing_others",
                "label": f"결측치 — 기타 {len(rest_cols)}개 컬럼 ({len(rest_idx)}건)",
                "indices": rest_idx,
                "description": (
                    f"상위 {MAX_COL_GROUPS}개 외 나머지 컬럼의 결측치입니다.  \n"
                    f"대상 컬럼: {col_preview}  \n"
                    "평균 대체 시 해당 행의 모든 결측 컬럼을 각 컬럼 평균으로 채웁니다."
                ),
                "col": None,
                "action": "replace_mean",
            }
        )

    # ── Group C: IsolationForest 탐지 이상치 ─────────────────────────
    if anomaly_mask is not None:
        anomaly_idx = df.index[anomaly_mask].tolist()
        if anomaly_idx and numeric_cols:
            anomaly_df = df.loc[anomaly_idx, numeric_cols]
            col_means = df[numeric_cols].mean()
            col_stds = df[numeric_cols].std().replace(0, 1)

            z_scores = (anomaly_df - col_means).abs() / col_stds
            primary_col_per_row = z_scores.idxmax(axis=1)

            outlier_candidates: list[dict] = []
            for col in numeric_cols:
                col_anomaly_idx = [
                    idx for idx in anomaly_idx if primary_col_per_row.get(idx) == col
                ]
                if col_anomaly_idx:
                    col_vals = df.loc[col_anomaly_idx, col].dropna()
                    val_range = (
                        f"{col_vals.min():.3g} ~ {col_vals.max():.3g}"
                        if len(col_vals) > 0 else "N/A"
                    )
                    outlier_candidates.append(
                        {
                            "key": f"outlier_{col}",
                            "label": f"이상치 — {col} ({len(col_anomaly_idx)}건)",
                            "indices": col_anomaly_idx,
                            "description": (
                                f"IsolationForest가 '{col}'의 극단값을 탐지했습니다.  \n"
                                f"값 범위: {val_range}"
                            ),
                            "col": col,
                            "action": "replace_mean",
                        }
                    )

            outlier_candidates.sort(key=lambda g: len(g["indices"]), reverse=True)
            groups.extend(outlier_candidates[:MAX_COL_GROUPS])

            if len(outlier_candidates) > MAX_COL_GROUPS:
                rest = outlier_candidates[MAX_COL_GROUPS:]
                rest_cols = [g["col"] for g in rest]
                rest_idx = sorted({i for g in rest for i in g["indices"]})
                col_preview = ", ".join(rest_cols[:6]) + ("..." if len(rest_cols) > 6 else "")
                groups.append(
                    {
                        "key": "outlier_others",
                        "label": f"이상치 — 기타 {len(rest_cols)}개 컬럼 ({len(rest_idx)}건)",
                        "indices": rest_idx,
                        "description": (
                            f"상위 {MAX_COL_GROUPS}개 외 나머지 컬럼의 이상치입니다.  \n"
                            f"대상 컬럼: {col_preview}"
                        ),
                        "col": None,
                        "action": "replace_mean",
                    }
                )

    return groups
