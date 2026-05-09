"""
모든 on_click 콜백 함수.
원칙: st.* 호출 금지, session_state 직접 조작만 수행.
모든 df 변경 전에 반드시 history에 스냅샷을 저장한다.

[마스크 관리 원칙]
- 값 교체(Winsorize/fillna) → _remove_from_mask(): 해당 인덱스만 정상으로 표시
- 행 삭제                   → _sync_mask_to_df():  삭제된 인덱스를 마스크에서 제거
- 자동 재탐지(re-detect)는 절대 하지 않는다. 재탐지는 사용자가 '재스캔' 버튼으로만 요청.

[감사 이력 원칙]
- 데이터를 변경하는 모든 액션은 _audit()으로 타임스탬프 + 내용을 audit_log에 기록한다.
- cb_ignore_group은 데이터 변경이 없으므로 기록하지 않는다.
- cb_rollback은 '롤백 실행됨'을 기록하여 이력이 단절 없이 이어진다.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st


# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────

def _snapshot() -> None:
    """현재 df를 history 스택에 push한다."""
    if "df" in st.session_state and st.session_state["df"] is not None:
        st.session_state["history"].append(st.session_state["df"].copy())


def _audit(msg: str) -> None:
    """타임스탬프와 함께 정제 작업 내역을 audit_log에 기록한다."""
    if "audit_log" not in st.session_state:
        st.session_state["audit_log"] = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["audit_log"].append(f"{ts} — {msg}")


def _remove_from_mask(indices: list[int]) -> None:
    """
    값 교체 후 해당 인덱스를 anomaly_mask에서 False(정상)로 표시한다.
    IsolationForest를 재실행하지 않으므로 새 이상치가 생성되지 않는다.
    """
    mask: pd.Series | None = st.session_state.get("anomaly_mask")
    if mask is None or not indices:
        return
    valid = [i for i in indices if i in mask.index]
    if valid:
        mask = mask.copy()
        mask.loc[valid] = False
        st.session_state["anomaly_mask"] = mask


def _sync_mask_to_df() -> None:
    """
    행 삭제 후 현재 df.index에 존재하지 않는 마스크 항목을 제거한다.
    reset_index를 사용하지 않으므로 인덱스가 sparse해져도 안전하게 동기화된다.
    """
    mask: pd.Series | None = st.session_state.get("anomaly_mask")
    df: pd.DataFrame | None = st.session_state.get("df")
    if mask is None or df is None:
        return
    st.session_state["anomaly_mask"] = mask.reindex(df.index, fill_value=False)


# ── Track 1: Auto-Fix ────────────────────────────────────────────────────────

def cb_remove_duplicates() -> None:
    """완전 중복 행을 제거한다."""
    _snapshot()
    df: pd.DataFrame = st.session_state["df"]
    n_before = len(df)
    st.session_state["df"] = df.drop_duplicates()
    n_removed = n_before - len(st.session_state["df"])
    _sync_mask_to_df()
    _audit(f"중복 행 제거 — {n_removed:,}건 삭제")


def cb_remove_garbage() -> None:
    """모든 수치 컬럼이 NaN인 garbage row를 제거한다."""
    _snapshot()
    df: pd.DataFrame = st.session_state["df"]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    n_removed = 0
    if numeric_cols:
        garbage_mask = df[numeric_cols].isnull().all(axis=1)
        n_removed = int(garbage_mask.sum())
        st.session_state["df"] = df[~garbage_mask]
    _sync_mask_to_df()
    _audit(f"쓰레기 행 제거 (전체 수치 컬럼 NaN) — {n_removed:,}건 삭제")


def cb_auto_fix_all() -> None:
    """중복 제거 + Garbage 행 제거를 한 번에 실행한다."""
    _snapshot()
    df: pd.DataFrame = st.session_state["df"]
    n_before = len(df)
    df = df.drop_duplicates()
    n_dup_removed = n_before - len(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    n_garbage = 0
    if numeric_cols:
        garbage_mask = df[numeric_cols].isnull().all(axis=1)
        n_garbage = int(garbage_mask.sum())
        df = df[~garbage_mask]
    st.session_state["df"] = df
    _sync_mask_to_df()
    _audit(f"Auto-Fix 실행 — 중복 {n_dup_removed:,}건 + 쓰레기 행 {n_garbage:,}건 제거")


# ── Track 2: HITL ────────────────────────────────────────────────────────────

def cb_replace_mean(group_indices: list[int], col: str | None) -> None:
    """
    결측치는 컬럼 평균으로 채우고, 이상치는 ±2σ 경계로 Winsorize(클리핑)한다.
    처리 완료된 인덱스는 mask에서 즉시 정상(False)으로 표시하여 재탐지를 차단한다.
    """
    _snapshot()
    df: pd.DataFrame = st.session_state["df"].copy()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    targets = [col] if (col and col in numeric_cols) else numeric_cols

    valid_idx = [i for i in group_indices if i in df.index]
    for c in targets:
        col_mean = df[c].mean()
        col_std = df[c].std()
        if pd.isna(col_mean):
            col_mean = 0.0
        if pd.isna(col_std) or col_std == 0:
            col_std = 1.0

        df.loc[valid_idx, c] = df.loc[valid_idx, c].fillna(col_mean)
        lower = col_mean - 2 * col_std
        upper = col_mean + 2 * col_std
        df.loc[valid_idx, c] = df.loc[valid_idx, c].clip(lower=lower, upper=upper)

    st.session_state["df"] = df
    _remove_from_mask(valid_idx)
    col_label = f"'{col}' 컬럼" if col else "전체 수치 컬럼"
    _audit(f"{col_label} 결측/이상치 {len(valid_idx):,}건 — 평균값·Winsorize(±2σ) 처리 완료")


def cb_delete_group(group_indices: list[int]) -> None:
    """지정 그룹의 행을 df에서 삭제한다."""
    _snapshot()
    df: pd.DataFrame = st.session_state["df"]
    valid_idx = [i for i in group_indices if i in df.index]
    st.session_state["df"] = df.drop(index=valid_idx)
    _sync_mask_to_df()
    _audit(f"행 삭제 승인 — {len(valid_idx):,}건 제거")


def cb_ignore_group(group_key: str) -> None:
    """
    데이터는 건드리지 않고 해당 그룹 키를 ignored_groups에 추가한다.
    UI에서 해당 expander가 사라지며, 스코어 계산에는 여전히 반영된다.
    데이터 변경 없으므로 audit 기록 제외.
    """
    if "ignored_groups" not in st.session_state:
        st.session_state["ignored_groups"] = set()
    st.session_state["ignored_groups"].add(group_key)


# ── Rollback ─────────────────────────────────────────────────────────────────

def cb_rollback() -> None:
    """history 스택에서 마지막 스냅샷을 꺼내 df와 mask를 함께 복원한다."""
    if st.session_state.get("history"):
        st.session_state["df"] = st.session_state["history"].pop()
        st.session_state["anomaly_mask"] = None
        st.session_state["ignored_groups"] = set()
        _audit("롤백 실행 — 직전 상태로 복원됨 (이후 이상치 재스캔 권장)")


# ── 재스캔 (사용자 명시 요청 시에만) ────────────────────────────────────────

def cb_rescan() -> None:
    """anomaly_mask를 None으로 초기화해 다음 렌더링에서 IsolationForest를 재실행한다."""
    st.session_state["anomaly_mask"] = None
    st.session_state["ignored_groups"] = set()


# ── 파일 업로드 ──────────────────────────────────────────────────────────────

def cb_load_file() -> None:
    """
    st.file_uploader의 on_change 콜백.
    CSV / Excel 파일을 읽어 session_state["df"]로 교체한다.
    - CSV: utf-8 → cp949(EUC-KR) 순서로 인코딩 자동 시도
    - 수치형 컬럼이 없어도 OK: LabelEncoder 파이프라인이 문자열 컬럼을 자동 처리
    """
    uploaded = st.session_state.get("_uploaded_file")
    st.session_state["_upload_error"] = None

    if uploaded is None:
        return

    try:
        name = uploaded.name.lower()
        if name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded)
        else:
            raw = uploaded.read()
            for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
                try:
                    import io
                    df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                    break
                except (UnicodeDecodeError, Exception):
                    continue
            else:
                st.session_state["_upload_error"] = (
                    "인코딩을 인식할 수 없습니다. UTF-8 또는 EUC-KR 파일을 사용하세요."
                )
                return

        if df.empty:
            st.session_state["_upload_error"] = "파일이 비어있습니다."
            return

        if df.shape[1] == 0:
            st.session_state["_upload_error"] = "파일에 컬럼이 없습니다."
            return

        st.session_state["df"] = df.reset_index(drop=True)
        st.session_state["history"] = []
        st.session_state["anomaly_mask"] = None
        st.session_state["ignored_groups"] = set()
        st.session_state["audit_log"] = []  # 새 파일 로드 시 이력 초기화

    except Exception as e:
        st.session_state["_upload_error"] = f"파일 로드 실패: {e}"


# ── 데이터 초기화 ────────────────────────────────────────────────────────────

def cb_reset_data() -> None:
    """더미 데이터를 새로 생성하고 모든 상태를 초기화한다."""
    from core.data_generator import generate_dataset

    st.session_state["df"] = generate_dataset()
    st.session_state["history"] = []
    st.session_state["anomaly_mask"] = None
    st.session_state["ignored_groups"] = set()
    st.session_state["audit_log"] = []
    st.session_state["_upload_error"] = None
