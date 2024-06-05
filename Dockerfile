# comment next line for testing
FROM python:3.10.13

WORKDIR /workspace

COPY app.py image_builder.py requirements.txt ./

RUN  pip install -r requirements.txt

# by default out build function will add or replace entrypoint
# even if you set an entrypoint
