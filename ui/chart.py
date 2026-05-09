"""데이터 분포 Scatter Plot 렌더링 모듈. 컬럼을 동적으로 감지한다."""

from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.graph_objects as go


def render_chart(
    df: pd.DataFrame | None,
    anomaly_mask: pd.Series | None,
    ignored_indices: set[int] | None = None,
) -> None:
    st.markdown("---")
    st.markdown("## 🗺️ 데이터 분포 시각화")

    if df is None or len(df) == 0:
        st.info("데이터가 없습니다. 사이드바에서 데이터를 불러오세요.")
        return

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) == 0:
        st.warning("수치형 컬럼이 없어 산점도를 그릴 수 없습니다.")
        return

    if len(numeric_cols) == 1:
        st.info(f"수치형 컬럼이 1개(`{numeric_cols[0]}`)뿐입니다. 산점도는 2개 이상 필요합니다.")
        return

    # ── 축 선택 UI ──────────────────────────────────────────────
    st.caption(
        "정상(🔵) · 검토 수용(⬜ ◇) · 미처리 이상치(🔴 ×)를 실시간으로 확인하세요. "
        "축을 바꿔가며 다양한 관점에서 분포를 살펴볼 수 있습니다."
    )

    sel_col1, sel_col2, sel_col3 = st.columns([3, 3, 4])
    with sel_col1:
        x_col = st.selectbox("X축", numeric_cols, index=0, key="chart_x_col")
    with sel_col2:
        default_y = next((c for c in numeric_cols if c != x_col), numeric_cols[0])
        y_col = st.selectbox(
            "Y축", numeric_cols, index=numeric_cols.index(default_y), key="chart_y_col"
        )
    with sel_col3:
        size_options = ["없음"] + numeric_cols
        size_col = st.selectbox("마커 크기 (선택)", size_options, index=0, key="chart_size_col")

    # ── 3-way 마스크 분리 ───────────────────────────────────────
    if anomaly_mask is not None and len(anomaly_mask) == len(df):
        is_anomaly = anomaly_mask.reindex(df.index, fill_value=False)
    else:
        is_anomaly = pd.Series(False, index=df.index)

    _ignored = ignored_indices or set()
    is_ignored = pd.Series(
        [i in _ignored for i in df.index], index=df.index, dtype=bool
    )

    # 카테고리 ①: 정상 (이상치도 무시도 아닌)
    normal_df = df[~is_anomaly & ~is_ignored].dropna(subset=[x_col, y_col])
    # 카테고리 ②: 검토 후 수용 (anomaly_mask=True이고 무시됨)
    accepted_df = df[is_anomaly & is_ignored].dropna(subset=[x_col, y_col])
    # 카테고리 ③: 미처리 이상치 (anomaly_mask=True이고 무시 안 됨)
    anomaly_df = df[is_anomaly & ~is_ignored].dropna(subset=[x_col, y_col])

    # 무시됐지만 anomaly_mask가 False인 행(결측/쓰레기 그룹 무시)은
    # is_ignored=True & is_anomaly=False → 정상 카테고리에서도 빠짐.
    # 이런 행은 별도로 정상 그룹에 합산한다.
    neutral_ignored_df = df[~is_anomaly & is_ignored].dropna(subset=[x_col, y_col])

    fig = go.Figure()

    # ── ① 정상 데이터 ────────────────────────────────────────────
    all_normal = pd.concat([normal_df, neutral_ignored_df])
    if len(all_normal) > 0:
        fig.add_trace(
            go.Scatter(
                x=all_normal[x_col],
                y=all_normal[y_col],
                mode="markers",
                name=f"정상 ({len(all_normal):,}건)",
                marker=dict(color="#3B82F6", size=5, opacity=0.55, line=dict(width=0)),
                hovertemplate=(
                    f"<b>정상</b><br>{x_col}: %{{x:.3g}}<br>{y_col}: %{{y:.3g}}<extra></extra>"
                ),
            )
        )

    # ── ② 검토 수용 (회색 ◇) ────────────────────────────────────
    if len(accepted_df) > 0:
        fig.add_trace(
            go.Scatter(
                x=accepted_df[x_col],
                y=accepted_df[y_col],
                mode="markers",
                name=f"검토 수용 ({len(accepted_df):,}건)",
                marker=dict(
                    color="rgba(156,163,175,0.7)",  # gray-400
                    size=9,
                    opacity=0.8,
                    symbol="diamond-open",
                    line=dict(width=1.5, color="#9CA3AF"),
                ),
                hovertemplate=(
                    f"<b>✅ 검토 수용</b><br>"
                    f"{x_col}: %{{x:.3g}}<br>{y_col}: %{{y:.3g}}<extra></extra>"
                ),
            )
        )

    # ── ③ 미처리 이상치 (빨간 ×) ────────────────────────────────
    if len(anomaly_df) > 0:
        if size_col != "없음" and size_col in anomaly_df.columns:
            vib = anomaly_df[size_col].fillna(anomaly_df[size_col].median())
            vmin, vmax = vib.min(), vib.max()
            sizes = 8 + (vib - vmin) / (vmax - vmin + 1e-9) * 12
        else:
            sizes = 12

        fig.add_trace(
            go.Scatter(
                x=anomaly_df[x_col],
                y=anomaly_df[y_col],
                mode="markers",
                name=f"이상치 ({len(anomaly_df):,}건)",
                marker=dict(
                    color="#EF4444",
                    size=sizes,
                    opacity=0.85,
                    symbol="x",
                    line=dict(width=2, color="#B91C1C"),
                ),
                hovertemplate=(
                    f"<b>⚠️ 이상치</b><br>"
                    f"{x_col}: %{{x:.3g}}<br>{y_col}: %{{y:.3g}}<extra></extra>"
                ),
            )
        )

    # ── 레이아웃 ────────────────────────────────────────────────
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            font=dict(size=13),
        ),
        height=420,
        margin=dict(l=50, r=20, t=30, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(128,128,128,0.25)", zeroline=False),
        yaxis=dict(gridcolor="rgba(128,128,128,0.25)", zeroline=False),
        hovermode="closest",
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── 분포 요약 ────────────────────────────────────────────────
    total_plot = len(all_normal) + len(accepted_df) + len(anomaly_df)
    if total_plot > 0:
        parts = [f"정상: {len(all_normal):,}건"]
        if len(accepted_df) > 0:
            parts.append(f"검토 수용: {len(accepted_df):,}건")
        if len(anomaly_df) > 0:
            pct = len(anomaly_df) / total_plot * 100
            parts.append(f"이상치: {len(anomaly_df):,}건 ({pct:.1f}%)")
        st.caption("  |  ".join(parts) + f"  |  X: `{x_col}` · Y: `{y_col}`")
