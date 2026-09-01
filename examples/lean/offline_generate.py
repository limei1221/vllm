from vllm import LLM, SamplingParams

llm = LLM(model="deepseek-ai/DeepSeek-V2-Lite-Chat", tensor_parallel_size=2)
outputs = llm.generate(
    ["Explain continuous batching in one paragraph."],
    SamplingParams(temperature=0, max_tokens=128),
)
print(outputs[0].outputs[0].text)
