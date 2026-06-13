@echo off
cd /d "C:\Users\sownd\OneDrive\Documents\Desktop\trading"

:: Start Streamlit in the background
start /B streamlit run ai_prediction_app.py --server.port 8501 --server.enableCORS=false --server.enableXsrfProtection=false


:: Wait 10 seconds for Streamlit to start
timeout /t 10 /nobreak

:: Keep localtunnel running permanently (auto-restarts if it crashes)
:loop
npx -y localtunnel --port 8501 --subdomain ai-trading-pro-sowndhar
timeout /t 10 /nobreak
goto loop
