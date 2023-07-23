import utils


print("# Create the PromptObject")
MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
promptObj = utils.PromptObject(MODEL_NAME)

print("# Download Llama-2 Model")
promptObj.create_model()

print("# Define Prompt")
prompt = input('Enter: ')
input_token_length = input('Enter length: ')
inputs = promptObj.run_input_prompt(prompt)

print("# Generate Outputs")
promptObj.generate_prompt_results(inputs, input_token_length)
