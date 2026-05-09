"""
session_state 초기화 모듈.
앱 최초 실행 시 한 번만 호출된다.
"""

import streamlit as st
from core.data_generator import generate_dataset


def init_state() -> None:
    """session_state 키가 없을 때만 초기값을 세팅한다."""
    if "initialized" not in st.session_state:
        st.session_state["df"] = generate_dataset()
        st.session_state["history"] = []           # List[pd.DataFrame]
        st.session_state["anomaly_mask"] = None    # pd.Series | None
        st.session_state["ignored_groups"] = set() # Set[str]
        st.session_state["audit_log"] = []         # List[str] — 정제 이력
        st.session_state["initialized"] = True
