"""
Created by Analitika at 12/02/2025
contact@analitika.fr
"""

# External imports
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
import torch

# Check if GPU is available
is_gpu = torch.cuda.is_available()
device = "cuda" if is_gpu else "cpu"

model_id = "facebook/opt-350m"  # Change to your preferred model

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16
    if is_gpu
    else torch.float32,  # Use FP16 on GPU, FP32 on CPU
    device_map="auto" if is_gpu else None,  # Assign to GPU if available
)

# Create a text generation pipeline
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=2048,  # ⬆ Increase from 512 to 2048
    temperature=0.7,
    do_sample=True,
    truncation=True,  # Keep truncation to avoid exceeding model limits
)

# Wrap the pipeline in LangChain's LLM class
local_llm = HuggingFacePipeline(pipeline=llm_pipeline)

if __name__ == "__main__":
    # Test the LLM
    response = local_llm.invoke("How do I get to heaven?")
    print(response)
