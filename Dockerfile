FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -e ".[dev]"

EXPOSE 7860

# 启动后端 API + Streamlit 前端
CMD sh -c "uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 600 & \
           streamlit run src/frontend/app.py --server.port 7860 --server.address 0.0.0.0"
