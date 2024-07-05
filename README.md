# sql-finetuning
Code for finetuning a language model to output SQL querries with inputs in the natural language

Open the sql-finetuning.ipynb and run it on google colab and change your run time to GPU. 
If you are running the ipynb locally, then you need to create a virtual env using CLI
"""python3 -m venv venv"""
"""source venv/bin/activate"""
you would not need the env on google colab.
make sure you have a GPU if you are runnning locally, because it will take a long time on GPU
Download the trained weights for inference if you dont want to train it
