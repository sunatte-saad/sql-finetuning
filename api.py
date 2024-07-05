from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 't5-small'
tokenizer = AutoTokenizer.from_pretrained(model_name)
finetuned_model = AutoModelForSeq2SeqLM.from_pretrained("finetuned_model_2_epoch")
finetuned_model = finetuned_model.to(device)

def generate_sql_from_text(question, context, tokenizer, finetuned_model, device):
    """
    Generate an SQL query from natural language text using a fine-tuned model.
    
    Parameters:
        question (str): The question in natural language.
        context (str): The context or tables information in natural language.
        tokenizer (object): The tokenizer to process the input.
        finetuned_model (object): The fine-tuned model to generate the SQL query.
        device (str): The device to use for computation ('cpu' or 'cuda').
        
    Returns:
        dict: A dictionary containing the input prompt and the model-generated SQL query.
    """
    prompt = f"""Tables:
    {context}

    Question:
    {question}

    SQL Query:
    """

    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = inputs.to(device)

    output = tokenizer.decode(
        finetuned_model.generate(
            inputs["input_ids"],
            max_new_tokens=200,
        )[0],
        skip_special_tokens=True
    )

    result = {
        "input_prompt": prompt,
        "generated_sql_query": output
    }

    return result

app = FastAPI()

class Query(BaseModel):
    question: str
    context: str

@app.post("/generate_sql")
def generate_sql(query: Query):
    result = generate_sql_from_text(query.question, query.context, tokenizer, finetuned_model, device)
    return result

def gradio_generate_sql(question, context):
    result = generate_sql_from_text(question, context, tokenizer, finetuned_model, device)
    return result['generated_sql_query']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the application with the specified method and port")
    parser.add_argument('--port', type=int, default=7860, help='Port number to run the server on')
    parser.add_argument('--method', type=str, choices=['api', 'gradio'], default='api', help='Method to run the application: api or gradio')
    args = parser.parse_args()

    if args.method == 'api':
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    elif args.method == 'gradio':
        iface = gr.Interface(fn=gradio_generate_sql, inputs=["text", "text"], outputs="text")
        iface.launch(server_name="0.0.0.0", server_port=args.port)
