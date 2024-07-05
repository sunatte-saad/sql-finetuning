
# SQL-Finetuning

Code for fine-tuning a language model to output SQL queries from natural language inputs.

## Instructions

### Running on Google Colab

1. Open `sql-finetuning.ipynb`.
2. Change your runtime to GPU.
3. Run the notebook.

### Running Locally

If you prefer to run the notebook locally, follow these steps to set up a virtual environment:

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

2. Activate the virtual environment:
   ```bash
   # On macOS and Linux
   source venv/bin/activate

   # On Windows
   .\venv\Scripts\activate
   ```

**Note**: You do not need a virtual environment on Google Colab.

### Hardware Requirements

Ensure you have access to a GPU, as training will take a long time without one.

## Pre-trained Weights

If you don't want to train the model yourself, you can download the pre-trained weights for inference:

[Download Pre-trained Weights](https://drive.google.com/file/d/1KhS2AxLuTfrcb0CzHEzDpAVZWXAto6OM/view?usp=sharing)
