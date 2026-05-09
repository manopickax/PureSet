"""상단 게이지 차트 + 요약 메트릭 렌더링 모듈."""

import streamlit as st
import plotly.graph_objects as go


def _score_color(score: float) -> str:
    if score >= 80:
        return "#10B981"   # green
    elif score >= 60:
        return "#F59E0B"   # amber
    return "#EF4444"       # red


def _score_label(score: float) -> str:
    if score >= 80:
        return "우수"
    elif score >= 60:
        return "보통"
    return "위험"


def render_header(metrics: dict) -> None:
    st.markdown("## 📊 데이터 신뢰도 현황")

    col_gauge, col_metrics = st.columns([4, 6])

    # ── 게이지 차트 ────────────────────────────────────────────
    with col_gauge:
        score = metrics["score"]
        color = _score_color(score)
        label = _score_label(score)

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score,
                number={
                    "suffix": "점",
                    "font": {"size": 48, "color": color, "family": "Arial Black"},
                },
                title={
                    "text": f"신뢰도 스코어 — <b>{label}</b>",
                    "font": {"size": 15},  # color 제거 → 모드 자동 대응
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickcolor": "rgba(128,128,128,0.6)",
                        "tickvals": [0, 20, 40, 60, 80, 100],
                    },
                    "bar": {"color": color, "thickness": 0.28},
                    "bgcolor": "rgba(0,0,0,0)",  # 투명 → 다크모드 자동 대응
                    "borderwidth": 0,
                    "steps": [
                        # 다크모드에서도 식별 가능한 채도 높은 반투명 색상
                        {"range": [0, 60], "color": "rgba(239,68,68,0.15)"},
                        {"range": [60, 80], "color": "rgba(245,158,11,0.15)"},
                        {"range": [80, 100], "color": "rgba(16,185,129,0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": color, "width": 5},
                        "thickness": 0.8,
                        "value": score,
                    },
                },
            )
        )
        fig.update_layout(
            height=240,
            margin=dict(l=30, r=30, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            transition={"duration": 0},   # sidebar 토글 시 위치 밀림 방지
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})

    # ── 요약 메트릭 카드 ───────────────────────────────────────
    with col_metrics:
        st.markdown("### 품질 요약 메트릭")

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric(
                label="전체 행",
                value=f"{metrics['n_total']:,}",
                help="현재 데이터셋의 총 행 수",
            )

        with m2:
            pct = metrics["r_missing"] * 100
            st.metric(
                label="🟡 결측치",
                value=f"{metrics['n_missing']:,}건",
                delta=f"{pct:.1f}%" if pct > 0 else "없음",
                delta_color="inverse",
                help="하나 이상의 수치 컬럼이 비어있는 행",
            )

        with m3:
            pct = metrics["r_dup"] * 100
            st.metric(
                label="🔵 중복 행",
                value=f"{metrics['n_duplicated']:,}건",
                delta=f"{pct:.1f}%" if pct > 0 else "없음",
                delta_color="inverse",
                help="모든 컬럼 값이 완전히 동일한 행",
            )

        with m4:
            n_pending = metrics.get("n_outlier_pending", metrics["n_outlier"])
            n_ignored = metrics.get("n_outlier_ignored", 0)
            pct = metrics["r_outlier"] * 100
            st.metric(
                label="🔴 이상치\n(미처리)",
                value=f"{n_pending:,}건",
                delta=(
                    f"✓ {n_ignored:,}건 수용"
                    if n_ignored > 0
                    else (f"{pct:.1f}%" if pct > 0 else "없음")
                ),
                delta_color="off" if n_ignored > 0 else "inverse",
                help=(
                    f"IsolationForest 탐지 총 {metrics['n_outlier']:,}건 중 "
                    f"{n_ignored:,}건 수용 · {n_pending:,}건 미처리"
                    if n_ignored > 0
                    else "IsolationForest가 탐지한 패턴 이탈 데이터"
                ),
            )

        # ── 스코어 세부 분해 바 ─────────────────────────────────
        st.markdown("**점수 감점 분해**")
        deduction_missing = min(metrics["r_missing"] * 40, 40)
        deduction_dup = min(metrics["r_dup"] * 30, 30)
        deduction_outlier = min(metrics["r_outlier"] * 30, 30)

        cats = ["결측치", "중복", "이상치"]
        actuals = [deduction_missing, deduction_dup, deduction_outlier]
        maxes = [40, 30, 30]
        fg_colors = ["#FCD34D", "#93C5FD", "#FCA5A5"]
        bg_colors = [
            "rgba(252,211,77,0.15)",
            "rgba(147,197,253,0.15)",
            "rgba(252,165,165,0.15)",
        ]

        bar_fig = go.Figure()
        for cat, actual, max_val, fg, bg in zip(cats, actuals, maxes, fg_colors, bg_colors):
            # 배경: 최대 감점 가능 범위
            bar_fig.add_trace(go.Bar(
                x=[max_val], y=[cat], orientation="h",
                marker_color=bg, showlegend=False, hoverinfo="skip",
            ))
            # 실제 감점
            outside = actual < max_val * 0.25
            bar_fig.add_trace(go.Bar(
                x=[actual], y=[cat], orientation="h",
                marker_color=fg, showlegend=False,
                text=f"{actual:.1f}p / {max_val}p",
                textposition="outside" if outside else "inside",
                textfont=dict(size=11),
                hovertemplate=f"{cat}: {actual:.1f}p 감점 (최대 {max_val}p)<extra></extra>",
            ))

        bar_fig.update_layout(
            barmode="overlay",
            height=140,
            margin=dict(l=0, r=60, t=5, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(range=[0, 44], showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=True, tickfont=dict(size=11)),
            showlegend=False,
        )
        st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})
