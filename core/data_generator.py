"""
더미 데이터 생성 모듈.
사인파 기반 시계열 정상 패턴을 만들고, 결측치/중복/이상치를 고의로 주입한다.
"""

import numpy as np
import pandas as pd


def generate_dataset(n_base: int = 1_000, seed: int = 42) -> pd.DataFrame:
    """
    n_base 행의 기본 시계열 데이터를 생성한 뒤
    이상 3종(결측 10%, 중복 5%, 극단 이상치 5%)을 주입하여 반환한다.
    """
    rng = np.random.default_rng(seed)

    # ── 정상 패턴 생성 ──────────────────────────────────────────────
    t = np.linspace(0, 4 * np.pi, n_base)  # 시간 축 (0 ~ 4π)

    # temperature: 60~80°C 사인 주기 변동 + 작은 가우시안 노이즈
    temperature = 70 + 10 * np.sin(t) + rng.normal(0, 0.8, n_base)

    # pressure: temperature에 물리적으로 비례 (0.3 계수) + 독립 노이즈
    pressure = 1.5 * temperature - 85 + rng.normal(0, 1.2, n_base)

    # vibration: 정상 가동 시 낮은 진동 + 소음
    vibration = 0.5 + 0.1 * np.abs(np.sin(t * 2)) + rng.normal(0, 0.05, n_base)

    timestamps = pd.date_range("2024-01-01", periods=n_base, freq="10min")

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature": temperature,
            "pressure": pressure,
            "vibration": vibration,
        }
    )

    # ── 이상 주입 ───────────────────────────────────────────────────
    df = _inject_outliers(df, rng, ratio=0.05)
    df = _inject_duplicates(df, rng, ratio=0.05)
    df = _inject_missing(df, rng, ratio=0.10)

    df = df.reset_index(drop=True)
    return df


# ── 주입 헬퍼 함수들 ────────────────────────────────────────────────────────


def _inject_outliers(df: pd.DataFrame, rng: np.random.Generator, ratio: float) -> pd.DataFrame:
    """패턴을 완전히 벗어난 극단값을 주입한다."""
    n = len(df)
    n_outlier = int(n * ratio)
    numeric_cols = ["temperature", "pressure", "vibration"]

    # 어떤 행에 이상치를 넣을지 선택 (timestamp는 건드리지 않음)
    outlier_idx = rng.choice(n, size=n_outlier, replace=False)

    for idx in outlier_idx:
        col = rng.choice(numeric_cols)
        if col == "temperature":
            # 정상 범위(60-80)를 한참 벗어난 값: 110~130 또는 -10~5
            df.loc[idx, col] = rng.choice(
                [rng.uniform(110, 130), rng.uniform(-10, 5)]
            )
        elif col == "pressure":
            # 압력 정상 범위(~20) 대비 극단값
            df.loc[idx, col] = rng.choice(
                [rng.uniform(60, 80), rng.uniform(-20, -5)]
            )
        else:  # vibration
            # 정상(~0.5) 대비 10배+
            df.loc[idx, col] = rng.uniform(5, 10)

    return df


def _inject_duplicates(df: pd.DataFrame, rng: np.random.Generator, ratio: float) -> pd.DataFrame:
    """기존 행을 그대로 복사해 삽입한다 (100% 완전 중복)."""
    n = len(df)
    n_dup = int(n * ratio)
    src_idx = rng.choice(n, size=n_dup, replace=False)
    dup_rows = df.iloc[src_idx].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    # 삽입 후 순서를 섞어 중복이 연속으로 몰리지 않게 함
    df = df.sample(frac=1, random_state=int(rng.integers(0, 9999))).reset_index(drop=True)
    return df


def _inject_missing(df: pd.DataFrame, rng: np.random.Generator, ratio: float) -> pd.DataFrame:
    """
    전체 셀의 ratio 비율을 NaN으로 마킹.
    그 중 일부는 timestamp를 제외한 모든 수치 컬럼이 NaN인 garbage row로 만든다.
    """
    n = len(df)
    numeric_cols = ["temperature", "pressure", "vibration"]
    total_cells = n * len(numeric_cols)
    n_missing = int(total_cells * ratio)

    # 일반 결측: 랜덤 (row, col) 쌍
    rows = rng.integers(0, n, size=n_missing)
    cols = rng.choice(numeric_cols, size=n_missing)
    for r, c in zip(rows, cols):
        df.loc[r, c] = np.nan

    # Garbage rows: 수치 컬럼 전체가 NaN인 행 (~1% 추가)
    n_garbage = max(1, int(n * 0.01))
    garbage_idx = rng.choice(n, size=n_garbage, replace=False)
    for idx in garbage_idx:
        for col in numeric_cols:
            df.loc[idx, col] = np.nan

    return df
