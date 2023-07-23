# LLAMA-2 API + Python / JupyterLab

## Source
- https://huggingface.co/docs/transformers/main_classes/text_generation

## Setup Workspace Environments

1. register by meta to download the model

- https://ai.meta.com/resources/models-and-libraries/llama-downloads/

2. Create a Hugging Face Account
- site: https://huggingface.co/

3. Create a HF Token
- HuggieFace -> Settings -> Access Token > Create
![Screenshot](/img/hf-token.png)

5. install python packages
    ```bash
    pip install -r requirements.txt
    python -m pip install --upgrade transformers
    ```

6. Test the Token
    ```bash
    huggingface-cli login
    ```

7. Request Llama-2 model access
- site: https://huggingface.co
- model: meta-llama/Llama-2-7b-chat-hf
- action: Request Submit
- It may take a day to get access

8. Receive the permission notification by Email

## Run Llama 2 in JupyterLab
1. install packages
    ```bash
    pip install -r requirements.txt
    ```

2. start jupyter lab
    ```bash
    jupyter lab 
    ```

3. Open Notebook 
    ```bash
    cd notebook/step-1.ipynb
    ```

4. Login to HF to get access to the model

![Screenshot](/img/login-hf.png)

5. Execute scripts in Notebook

- step-1.ipynb
- step-2.ipynb


## Run Llama 2 in your IDE

1. Login HuggingFace
    ```bash
    huggingface-cli login
    ```

2. install packages
    ```bash
    pip install -r requirements.txt
    ```
3. download model
    ```bash
    python src/main.py
    ```