FROM python:3.8-slim-buster
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY app/main.py main.py
COPY app/remove_stopwords.py remove_stopwords.py
COPY app/text_preprocess.py text_preprocess.py
COPY app/stopwords.txt stopwords.txt
COPY models/naive_bayes.pkl naive_bayes.pkl

CMD ["uvicorn","main:app","--reload","--host","0.0.0.0","--port","8000"]