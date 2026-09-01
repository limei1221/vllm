#!/bin/bash
# Serve a DeepSeek model with tensor parallelism and send one streaming request.
vllm serve deepseek-ai/DeepSeek-V2-Lite-Chat --tensor-parallel-size 2 &

# Wait for the server to be ready
sleep 30

curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true,
    "max_tokens": 64
  }'
