#FROM registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-py37-torch1.11.0-tf1.15.5-1.1.2
FROM modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-py310-torch2.3.0-1.19.2

# httpx pytest pytest-asyncio for debug
RUN pip install fastapi==0.88.0 uvicorn==0.20.0 \
    loguru python-multipart \
    httpx pytest pytest-asyncio;
