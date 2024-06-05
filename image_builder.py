from everai.image import Builder

IMAGE = 'quay.io/mc_jones/llama2-7b-chat:v0.0.10'

image_builder = Builder.from_dockerfile(
    'Dockerfile',
    labels={
        "any-your-key": "value",
    },
    repository=IMAGE,
    platform=['linux/arm64', 'linux/x86_64'],
)
