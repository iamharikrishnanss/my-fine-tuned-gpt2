import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.title("AI Text Generator")
st.write("Interact with your fine-tuned language model hosted on Hugging Face Hub.")

# Specify your model repository from Hugging Face
model_name = "iamharikrisnan/fine-tuned-gpt-medium"  

# Option 1: If you are logged in via huggingface-cli, you can omit the token
# Option 2: If the repository is private or you haven't logged in, pass the token
token = "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"  # Replace with your Hugging Face access token if needed

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

user_input = st.text_area("Enter a prompt:", "Once upon a time in a land far away,")

if st.button("Generate Text"):
    if user_input.strip():
        with st.spinner("Generating..."):
            inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model.generate(**inputs, max_length=50, num_return_sequences=1)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.subheader("Generated Text:")
        st.write(generated_text)
    else:
        st.warning("Please enter a valid prompt!")
