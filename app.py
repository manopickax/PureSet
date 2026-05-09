"""
PureSet — B2B 데이터 신뢰도 진단 & 자동 교정 SaaS MVP
실행: streamlit run app.py
"""

import pandas as pd
import streamlit as st

from core.detector import build_hitl_groups, detect_anomalies
from core.scorer import compute_metrics
from core.state import init_state
from ui.actions import render_actions
from ui.chart import render_chart
from ui.header import render_header
from ui.sidebar import render_sidebar


def _apply_global_css() -> None:
    st.markdown(
        """
        <style>
            /* 배경/사이드바: 하드코딩 제거 → Streamlit 다크/라이트 모드 자동 적용 */

            /* 메트릭 카드: 배경 제거, 테두리만 유지 (모드별 자동 대응) */
            [data-testid="stMetric"] {
                border: 1px solid rgba(128, 128, 128, 0.25);
                border-radius: 10px;
                padding: 0.8rem 1rem;
                overflow: hidden;
            }

            /* 탭 스타일 */
            .stTabs [data-baseweb="tab-list"] {
                gap: 6px;
                border-bottom: 2px solid rgba(128, 128, 128, 0.25);
            }
            .stTabs [data-baseweb="tab"] {
                border-radius: 8px 8px 0 0;
                font-weight: 600;
            }

            /* expander: 흰 배경 하드코딩 제거 */
            div[data-testid="stExpander"] {
                border: 1px solid rgba(128, 128, 128, 0.25);
                border-radius: 10px;
                margin-bottom: 0.5rem;
            }

            /* primary 버튼 */
            .stButton > button[kind="primary"] {
                background: linear-gradient(135deg, #1E40AF, #3B82F6);
                border: none;
                font-weight: 700;
                letter-spacing: 0.02em;
            }

            /* 구분선 */
            hr { border-color: rgba(128, 128, 128, 0.25); }

            /* 메트릭 라벨 줄바꿈 허용 — 좁은 컬럼에서 잘림 방지 */
            [data-testid="stMetricLabel"] > div {
                white-space: normal !important;
                overflow: visible !important;
                line-height: 1.35 !important;
                font-size: 0.8rem !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    # ── 페이지 설정 ────────────────────────────────────────────
    st.set_page_config(
        page_title="PureSet — 데이터 신뢰도 진단",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _apply_global_css()

    # ── 세션 상태 초기화 ───────────────────────────────────────
    init_state()

    df = st.session_state["df"]

    # ── 이상치 탐지 ────────────────────────────────────────────
    # mask가 None일 때만 탐지 (최초 로딩 또는 사용자가 '재스캔' 버튼 클릭 시)
    # 수정/삭제 액션 후에는 자동 재탐지하지 않는다 → 무한 루프 방지
    mask = st.session_state.get("anomaly_mask")
    if mask is None:
        if df is not None and len(df) >= 10:
            with st.spinner("🔍 IsolationForest 이상치 탐지 중..."):
                st.session_state["anomaly_mask"] = detect_anomalies(df)
        else:
            st.session_state["anomaly_mask"] = pd.Series(dtype=bool)

    # sparse 인덱스 안전 처리: df.index에 없는 mask 항목 제거
    anomaly_mask = st.session_state["anomaly_mask"]
    if anomaly_mask is not None and df is not None:
        anomaly_mask = anomaly_mask.reindex(df.index, fill_value=False)

    # ── 파생 데이터 계산 ───────────────────────────────────────
    hitl_groups = build_hitl_groups(df, anomaly_mask)

    # ── 무시된 그룹의 인덱스 집합 계산 ───────────────────────────
    ignored: set[str] = st.session_state.get("ignored_groups", set())
    ignored_indices: set[int] = {
        idx
        for g in hitl_groups if g["key"] in ignored
        for idx in g["indices"]
    }

    metrics = compute_metrics(df, anomaly_mask, ignored_indices)

    # ── UI 렌더링 ──────────────────────────────────────────────
    render_sidebar(df, metrics)
    render_header(metrics)
    render_chart(df, anomaly_mask, ignored_indices)
    render_actions(df, hitl_groups)


if __name__ == "__main__":
    main()
