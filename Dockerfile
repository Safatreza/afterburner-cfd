FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY main.py ./main.py
COPY ns_solver_config.py ./ns_solver_config.py
ENTRYPOINT ["python", "main.py"]
CMD ["--config", "/app/config.yaml"] 