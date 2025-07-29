FROM python:3.11

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install flask numpy pandas yfinance scikit-learn tensorflow streamlit

EXPOSE 5000
EXPOSE 8501

CMD ["python", "app.py"]
# For Streamlit apps, comment above and uncomment below:
# CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
