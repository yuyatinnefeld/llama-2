import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from dataclasses import dataclass


@dataclass
class PromptObject:
    model_name: str
    
    def create_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

    def run_input_prompt(self, prompt):
        inputs = self.tokenizer.encode(
            prompt,
            return_tensors="pt"
        )
        print("🦙💬 Llama-2 Prompt Setup ✅")
        return inputs

    def generate_prompt_results(self, inputs, input_token_length):
        print("🦙💬 Llama-2 Prompt Result getting...")
        timeStart = time.time()
        outputs = self.model.generate(
            inputs,
            max_new_tokens=int(input_token_length),
        )

        output_str = self.tokenizer.decode(outputs[0])

        print(output_str)
        print("🦙💬 Llama-2 Prompt Output ✅")
        print("Time taken: ", -timeStart + time.time())