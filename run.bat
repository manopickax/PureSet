@echo off
chcp 65001 > nul
title PureSet - 데이터 신뢰도 진단
cd /d "%~dp0"

echo.
echo  ====================================
echo   PureSet 시작 중...
echo   브라우저에서 열립니다: http://localhost:8501
echo   종료하려면 이 창을 닫으세요.
echo  ====================================
echo.

streamlit run app.py --server.headless false

pause
