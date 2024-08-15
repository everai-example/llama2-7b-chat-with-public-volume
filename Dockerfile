# the following basic images including：
#     transformers~=4.39.3
#     torch~=2.2.2
#     accelerate~=0.30.1
#     protobuf~=5.26.1
#     sentencepiece~=0.2.0
#     Flask~=3.0.3
#     Werkzeug~=3.0.3
######################################
#FROM everai2024/transformers-pytorch-gpu:v0.0.1
FROM quay.io/everai2024/transformers-pytorch-gpu:v0.0.1

WORKDIR /workspace

COPY app.py image_builder.py requirements.txt ./

RUN  pip install -r requirements.txt

# by default out build function will add or replace entrypoint
# even if you set an entrypoint
