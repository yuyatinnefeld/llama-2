from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteriaList,
    MaxLengthCriteria,
)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

# set pad_token_id to eos_token_id because OPT does not have a PAD token
model.config.pad_token_id = model.config.eos_token_id
input_prompt = "What is DevOps?"

input_ids = tokenizer(input_prompt, return_tensors="pt")
stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=64)])
outputs = model.contrastive_search(
    **input_ids, penalty_alpha=0.6, top_k=4, stopping_criteria=stopping_criteria
)
result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(result)