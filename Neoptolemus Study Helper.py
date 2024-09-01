#Importing neccasary functions
import pandas as pd
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from tkinter import Tk, filedialog
import warnings
import random
import csv

#Setting up variables
counter = 0
Questions = ["Why are ","Why is","What is","What are"]

# Hides a warning about the tokenization
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*", category=FutureWarning)

# Function to let the user upload a file using a file dialog
def upload_csv():
    root = Tk()
    root.withdraw()  
    file_path = filedialog.askopenfilename(
        title="Select CSV File",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
    )
    root.destroy()
    return file_path

# Asks the user to upload and loads the csv file full of notes
if __name__ == "__main__":
    csv_file = upload_csv()
    if csv_file:
        df = pd.read_csv(csv_file)
        print("File selected\n")
    else:
        print("No file selected.\n")

#Inputs to make the AI more specific
subject = input("What subject are you studying? \n").lower
age = input("What grade are you in? \n")

# Initializes the GPT-Neo model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"  # You can choose other models like 'gpt-neo-1.3B'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

#Progress point
print("Model and Tokenizer initialized\n")

# Set pad_token_id to eos_token_id if pad_token_id is not set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

#Progress point
print ("Token pad ID Set\n ")

# Loads the CSV
with open(csv_file, mode='r') as file:
    reader = csv.reader(file)
    rows = list(reader)

#Progress point
print ("CSV loaded\n")


# Initialize row counter
counter = 1

#Progress point
print("AI booting up\n")

#Loop
while counter <= 5:
    
    #Picks a random term/definition and question starter to make the prompt
    Question = random.choice(Questions)
    data_string = random.choice(rows)  
    prompt = random.choice(data_string)

    # Inputs into the AI
    inputs = tokenizer(
        f"{age} {subject} Questions: 1. {Question} {prompt}",
        return_tensors="pt",
        padding=True,  # Ensure padding is added
        truncation=True,  # Truncate if needed
        max_length=1024   # Adjust to fit within limits
    )

    # The output from GPT-Neo
    outputs = model.generate(
        inputs['input_ids'], 
        attention_mask=inputs['attention_mask'],  # Pass the attention mask
        max_new_tokens=100,  # Generate up to 100 new tokens
        do_sample=True, 
        pad_token_id=tokenizer.pad_token_id  # Ensure pad token ID is set
    )

    #Decodes and print the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"", generated_text)

    #Progress Point
    print(f"Cycle {counter} completed")
    
    # Increases the count
    counter += 1