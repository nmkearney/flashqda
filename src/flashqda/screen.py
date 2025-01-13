from .analyze import send_to_openai
import csv, os
import pandas as pd
from pandas.errors import EmptyDataError
from tqdm import tqdm
import json
from collections import Counter
from datetime import datetime
import importlib.resources as res

def initialize_files(file_name):

    # Read the abstracts from the data file and create a list
    data_folder = os.path.join('Data', file_name)
    df = pd.read_csv(data_folder, index_col = 0)
    abstracts = df.index.tolist()
    
    # Create a results folder if does not already exist
    results_folder = os.path.join('Results', file_name)
    results_file = os.path.join(results_folder, f"{file_name}_screening.csv")

    temp_folder = os.path.join(results_folder, 'temp')
    log_folder = os.path.join(results_folder, 'log')
 
    if not os.path.exists(log_folder):
        os.makedirs(results_folder, exist_ok = True)
        os.makedirs(log_folder, exist_ok = True)
        os.makedirs(temp_folder, exist_ok = True)

    if not os.path.exists(results_file):
        
        # Create a results file from the list of abstracts
        abstracts_df = pd.DataFrame(index = abstracts)
        abstracts_df.to_csv(results_file)
    
    # Ensure log file exists or create with default values
    log_file = os.path.join(log_folder, f"{file_name}_screening_log.csv")
    if not os.path.exists(log_file):
        with open(log_file, 'w') as log:
            writer = csv.writer(log)
            writer.writerow(['start_time', 'end_time', 'last_processed_row', 'num_rows'])
            writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 0, 0, len(abstracts)])

    # Initialize temp file with empty DataFrame if it doesn't exist
    temp_file = os.path.join(temp_folder, f"{file_name}_screening_temp.csv")
    if not os.path.exists(temp_file):
        abstracts_df.to_csv(temp_file)
    else:
        abstracts_df = pd.read_csv(temp_file, index_col = 0)
        abstracts = abstracts_df.index.tolist()

    return results_file, temp_file, log_file, abstracts_df, abstracts

def get_start_ids(log_file):
    try:
        with open(log_file, 'r') as log:
            reader = csv.reader(log)
            next(reader)  # Skip header
            row = next(reader)
            start_time = row[0]
            end_time = row[1]
            start_abstract_id, num_abstracts = map(int, row[2:])
    except EmptyDataError:
        start_time, end_time, start_abstract_id, num_abstracts = 0, 0, 0, 0  # Default values

    return start_time, end_time, start_abstract_id, num_abstracts

def update_log(log_file, start_time, end_time, abstract_id, num_abstracts):
    with open(log_file, 'w') as log:
        writer = csv.writer(log)
        writer.writerow(['start_time', 'end_time', 'last_processed_row', 'num_rows'])
        writer.writerow([start_time, end_time, abstract_id, num_abstracts])

def append_to_context(context_window, abstract, context_length):
    if context_length > 0:
        if len(context_window) == context_length:
            context_window.pop(0)
        context_window.append(abstract)
    return context_window

def handle_abstract(abstract, context_window, criteria, query_count):

    decisions = []
    
    for _ in range(query_count):

        resource_package = 'flashqda.prompts'
        resource_path = 'label_text.txt'
        system_prompt = "You are a helpful assistant that labels text. Follow the user's instructions carefully. Respond using JSON."
        
        # Read the prompt file content
        with res.open_text(resource_package, resource_path) as file:
            prompt = file.read()
        prompt = prompt.format(text = abstract, context_window = context_window, criteria = criteria)
        
        # Send to OpenAI
        decision = send_to_openai(system_prompt, prompt)

        if decision:

            data = json.loads(decision)
            for key, value in data["criteria"].items():
                decisions.append(value)
    
    decisions_count = Counter(decisions)

    return decisions_count

def screen_abstracts(file_name, criteria, context_length=0, query_count=3):

    # Retrieve abstracts
    results_file, temp_file, log_file, abstracts_df, abstracts = initialize_files(file_name)
    start_time, end_time, start_abstract_id, num_abstracts = get_start_ids(log_file)
    context_window = []
    print(f"Starting labelling at row {start_abstract_id+1}.")

    # Screen abstracts
    for i in tqdm(range(num_abstracts)):

        if i >= start_abstract_id:

            decisions_count = handle_abstract(abstracts[i], context_window, criteria, query_count)
            context_window = append_to_context(context_window, abstracts[i], context_length)
            for key, value in decisions_count.items():
                if key not in abstracts_df.columns:
                    abstracts_df[key] = 0
                abstracts_df.loc[abstracts_df.index[i], key] = value / query_count * 100
            end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            try:
                pd.DataFrame(abstracts_df).to_csv(temp_file)
                update_log(log_file, start_time, end_time, i+1, num_abstracts)
            except KeyboardInterrupt:
                print(f"Saving interrupted at abstract {i}. Continuing until saving completed.")
                pd.DataFrame(abstracts_df).to_csv(temp_file)
                update_log(log_file, start_time, end_time, i+1, num_abstracts)
                raise

    start_abstract_id += 1

    pd.DataFrame(abstracts_df).to_csv(results_file)
    print(f"Finished labelling at row {i+1}. Results saved at {results_file}.")