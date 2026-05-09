"""Auto-Fix 탭 + HITL expander + 데이터 미리보기 렌더링 모듈."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.callbacks import (
    cb_auto_fix_all,
    cb_delete_group,
    cb_ignore_group,
    cb_remove_duplicates,
    cb_remove_garbage,
    cb_replace_mean,
)


def render_actions(df: pd.DataFrame | None, hitl_groups: list[dict]) -> None:
    st.markdown("---")

    tab_auto, tab_hitl, tab_preview = st.tabs(
        ["🔧  Auto-Fix (자동 수정)", "🧠  HITL 패턴 검토", "📋  데이터 미리보기"]
    )

    # ── Tab 1: Auto-Fix ─────────────────────────────────────────────
    with tab_auto:
        _render_autofix(df)

    # ── Tab 2: HITL ─────────────────────────────────────────────────
    with tab_hitl:
        _render_hitl(df, hitl_groups)

    # ── Tab 3: Preview ──────────────────────────────────────────────
    with tab_preview:
        _render_preview(df)


# ── 내부 렌더링 함수 ────────────────────────────────────────────────────────


def _render_autofix(df: pd.DataFrame | None) -> None:
    st.markdown("### 완전 자동 정제 (1-Click)")
    st.info(
        "**100% 완전 중복 행**과 **모든 컬럼이 비어있는 쓰레기 행**을 즉시 제거합니다. "
        "실행 전 상태는 자동으로 저장되므로 언제든 롤백할 수 있습니다.",
        icon="ℹ️",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        n_dup = int(df.duplicated().sum()) if df is not None else 0
        st.button(
            f"⚡  전체 Auto-Fix 실행",
            on_click=cb_auto_fix_all,
            use_container_width=True,
            type="primary",
            help="중복 행 제거 + 쓰레기 행 제거를 한 번에 실행합니다.",
        )

    with col2:
        st.button(
            f"🔁  중복 행 제거",
            on_click=cb_remove_duplicates,
            use_container_width=True,
            help="완전히 동일한 행(모든 컬럼 값 일치)을 제거합니다.",
        )

    with col3:
        st.button(
            f"🗑️  쓰레기 행 제거",
            on_click=cb_remove_garbage,
            use_container_width=True,
            help="모든 수치 컬럼이 NaN인 완전 무효 행을 제거합니다.",
        )

    # 현재 상태 요약
    if df is not None and len(df) > 0:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        n_dup = int(df.duplicated().sum())
        n_garbage = int(df[numeric_cols].isnull().all(axis=1).sum()) if numeric_cols else 0

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            status = "✅ 없음" if n_dup == 0 else f"⚠️ {n_dup:,}건 존재"
            st.markdown(f"**중복 행 현황:** {status}")
        with c2:
            status = "✅ 없음" if n_garbage == 0 else f"⚠️ {n_garbage:,}건 존재"
            st.markdown(f"**쓰레기 행 현황:** {status}")


def _render_hitl(df: pd.DataFrame | None, hitl_groups: list[dict]) -> None:
    st.markdown("### 패턴별 이상 데이터 검토 (Human-in-the-Loop)")
    st.caption(
        "IsolationForest가 탐지한 이상 패턴을 그룹 단위로 보여줍니다. "
        "각 그룹을 펼쳐 데이터를 확인하고 처리 방식을 선택하세요."
    )
    st.info(
        "**정제 후 새로운 이상치가 탐지될 수 있습니다.**  \n"
        "IsolationForest는 절대적 기준이 아닌 **상대 밀도 비교**로 작동합니다. "
        "극단적 이상치를 제거하면 이전에 정상이었던 경계값이 새 최하위권이 될 수 있습니다.  \n"
        "→ **물리적으로 불가능한 값**(온도 130°C 등)은 수정/삭제하고, "
        "**정제 후 새로 탐지된 borderline 값**은 '무시'를 선택하는 것이 올바른 판단입니다.",
        icon="💡",
    )

    if df is None or len(df) == 0:
        st.info("데이터가 없습니다.")
        return

    ignored: set[str] = st.session_state.get("ignored_groups", set())

    # 무시되지 않은 그룹 + 인덱스가 현재 df에 실제로 존재하는 그룹만 표시
    visible: list[dict] = []
    for group in hitl_groups:
        if group["key"] in ignored:
            continue
        valid_idx = [i for i in group["indices"] if i in df.index]
        if valid_idx:
            visible.append({**group, "valid_idx": valid_idx})

    if not visible:
        st.success(
            "✅ 검토할 이상 데이터 그룹이 없습니다. 데이터가 깨끗한 상태입니다!",
            icon="🎉",
        )
        if ignored:
            st.caption(f"무시 처리된 그룹 {len(ignored)}개 포함")
        return

    st.markdown(f"**{len(visible)}개 그룹**이 검토를 기다리고 있습니다.")

    for group in visible:
        valid_idx: list[int] = group["valid_idx"]
        key: str = group["key"]
        col: str | None = group["col"]

        with st.expander(f"⚠️  {group['label']}", expanded=False):
            st.caption(group["description"])

            # 미리보기 (최대 5행)
            preview_df = df.loc[valid_idx[:5]]
            st.dataframe(preview_df, use_container_width=True, hide_index=False)
            if len(valid_idx) > 5:
                st.caption(f"... 외 {len(valid_idx) - 5:,}건 더 있음")

            st.markdown("")

            # 액션 버튼 3종
            btn1, btn2, btn3 = st.columns(3)

            with btn1:
                st.button(
                    "✅  평균값으로 대체",
                    key=f"replace_{key}",
                    on_click=cb_replace_mean,
                    args=(valid_idx, col),
                    use_container_width=True,
                    type="primary",
                    help=(
                        f"'{col}' 컬럼의 값을 현재 컬럼 평균으로 교체합니다."
                        if col
                        else "모든 수치 컬럼을 각 컬럼 평균으로 채웁니다."
                    ),
                )

            with btn2:
                st.button(
                    "🗑️  해당 행 전체 삭제",
                    key=f"delete_{key}",
                    on_click=cb_delete_group,
                    args=(valid_idx,),
                    use_container_width=True,
                    help="이 그룹에 해당하는 행을 데이터셋에서 영구 제거합니다.",
                )

            with btn3:
                st.button(
                    "👁️  무시 (이 그룹 건너뜀)",
                    key=f"ignore_{key}",
                    on_click=cb_ignore_group,
                    args=(key,),
                    use_container_width=True,
                    help="데이터는 변경하지 않고 이 검토 그룹을 숨깁니다.",
                )


def _render_preview(df: pd.DataFrame | None) -> None:
    st.markdown("### 현재 데이터 미리보기 (상위 10행)")

    if df is None or len(df) == 0:
        st.info("데이터가 없습니다. 사이드바에서 초기화하세요.")
        return

    # 수치 컬럼에 색상 그라디언트 적용
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    styled = (
        df.head(10)
        .style.background_gradient(subset=numeric_cols, cmap="Blues", axis=0)
        .highlight_null(color="#FEE2E2")
        .format({col: "{:.3f}" for col in numeric_cols}, na_rep="NaN")
    )

    st.dataframe(styled, use_container_width=True, hide_index=False)
    st.caption(f"전체 {len(df):,}행 중 상위 10행 표시 | 빨간 셀 = 결측치(NaN)")
