FROM continuumio/miniconda3:4.12.0

WORKDIR /app

COPY local_requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5000

ENTRYPOINT [ "python" ]
CMD ["-m", "gunicorn", "-w", "1", "--bind", "0.0.0.0:5000", "--timeout", "0", "--preload", "--log-level", "debug", "--access-logfile", "/app/access_logfile.txt", "--error-logfile", "/app/error_logfile.txt", "app:create_app()"]