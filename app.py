import flask
import typing
from threading import Thread

from everai.app import App, context, VolumeRequest
from everai_autoscaler.builtin import FreeWorkerAutoScaler
from everai.image import Image, BasicAuth
from everai.resource_requests import ResourceRequests
from everai.placeholder import Placeholder
from image_builder import IMAGE

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, PreTrainedTokenizerBase, TextIteratorStreamer

APP_NAME = 'llama2-7b-chat'
VOLUME_NAME = 'expvent/models--meta-llama--llama-2-7b-chat-hf'
MODEL_NAME = 'meta-llama/Llama-2-7b-chat-hf'
HUGGINGFACE_SECRET_NAME = 'your-huggingface-secret-name'
QUAY_IO_SECRET_NAME = 'your-quay-io-secret-name'
CONFIGMAP_NAME = 'llama2-configmap'

tokenizer: typing.Optional[PreTrainedTokenizerBase] = None
model = None

image = Image.from_registry(IMAGE, auth=BasicAuth(
        username=Placeholder(QUAY_IO_SECRET_NAME, 'username', kind='Secret'),
        password=Placeholder(QUAY_IO_SECRET_NAME, 'password', kind='Secret'),
    ))

app = App(
    APP_NAME,
    image=image,
    volume_requests=[
        VolumeRequest(name=VOLUME_NAME),
    ],
    secret_requests=[
        HUGGINGFACE_SECRET_NAME,
        QUAY_IO_SECRET_NAME
    ],
    configmap_requests=[CONFIGMAP_NAME],
    autoscaler=FreeWorkerAutoScaler(
        # keep running workers even no any requests, that make reaction immediately for new request
        min_workers=Placeholder(kind='ConfigMap', name=CONFIGMAP_NAME, key='min_workers'),
        # the maximum works setting, protect your application avoid to pay a lot of money
        # when an attack or sudden traffic
        max_workers=Placeholder(kind='ConfigMap', name=CONFIGMAP_NAME, key='max_workers'),
        # this factor controls autoscaler how to scale up your app
        min_free_workers=Placeholder(kind='ConfigMap', name=CONFIGMAP_NAME, key='min_free_workers'),
        # this factor controls autoscaler how to scale down your app
        max_idle_time=Placeholder(kind='ConfigMap', name=CONFIGMAP_NAME, key='max_idle_time'),
        # this factor controls autoscaler how many steps to scale up your app from queue 
        scale_up_step=Placeholder(kind='ConfigMap', name=CONFIGMAP_NAME, key='scale_up_step'),
    ),
    resource_requests=ResourceRequests(
        cpu_num=2,
        memory_mb=20480,
        gpu_num=1,
        gpu_constraints=[
            "A100 40G"
        ],
    ),
)

@app.prepare()
def prepare_model():
    volume = context.get_volume(VOLUME_NAME)
    assert volume is not None and volume.ready

    secret = context.get_secret(HUGGINGFACE_SECRET_NAME)
    assert secret is not None
    huggingface_token = secret.get('token-key-as-your-wish')

    model_dir = volume.path

    global model
    global tokenizer
    
    #model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, local_files_only=True)
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME,
                                             token=huggingface_token,
                                             cache_dir=model_dir,
                                             torch_dtype=torch.float16,
                                             local_files_only=True)
    
    #tokenizer = LlamaTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME,
                                               token=huggingface_token,
                                               cache_dir=model_dir,
                                               local_files_only=True)
    
    if torch.cuda.is_available():
        model.cuda(0)

    


# service entrypoint
# api service url looks https://everai.expvent.com/api/routes/v1/default/llama2-7b-chat/chat
# for test local url is http://127.0.0.1/chat
@app.service.route('/chat', methods=['GET','POST'])
def chat():
    if flask.request.method == 'POST':
        data = flask.request.json
        prompt = data['prompt']
    else:
        prompt = flask.request.args["prompt"]

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to('cuda:0')
    output = model.generate(input_ids, max_length=256, num_beams=4, no_repeat_ngram_size=2)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    text = f'{response}'
    
    # return text with some information
    resp = flask.Response(text, mimetype='text/plain', headers={'x-prompt-hash': 'xxxx'})
    return resp

@app.service.route('/sse', methods=['GET','POST'])
def sse():
    if flask.request.method == 'POST':
        data = flask.request.json
        prompt = data['prompt']
    else:
        prompt = flask.request.args["prompt"]

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to('cuda:0')

    streamer = TextIteratorStreamer(
        tokenizer, timeout=600.0, skip_prompt=True, skip_special_tokens=True
    )

    generate_args = [input_ids]

    generate_kwargs = dict(
        streamer=streamer,
        max_new_tokens=250,
        do_sample=True,
        top_p=0.95,
        temperature=float(0.8),
        top_k=1,
    )

    t = Thread(target=model.generate, args=generate_args, kwargs=generate_kwargs)
    def generator():
        for text in streamer:
            yield text
    t.start()

    # return active messages from the server to the client
    resp = flask.Response(generator(), mimetype='text/event-stream', headers={})

    return resp


@app.clear()
def clear():
    print('clear called')
