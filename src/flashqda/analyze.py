from .general import save_to_csv, read_csv_file
import re, csv, os, threading, string, requests, random, time
from openai import OpenAI
from tqdm import tqdm
from pandas.errors import EmptyDataError
import pandas as pd
import json
import numpy as np
from datetime import datetime
import importlib.resources as res

classification_analyses = ["tenses", "relationships_classify"]
extraction_analyses = ["relationships_extract"]
all_analyses = classification_analyses + extraction_analyses #+ keyword_analyses

def api_call_wrapper(system_prompt, prompt, result_container, event):
    
    """Try prompting OpenAI."""

    client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format = {"type": "json_object"},
            temperature = 0.5,
            timeout=10
        )
        result_container[0] = response
        event.set()
    except Exception as e:
        result_container[1] = e
        event.set()

def send_to_openai(system_prompt, prompt, max_retries=3, manual_timeout=15):
    
    """Handle time-outs by OpenAI."""
    
    base_delay = 1  # Base delay in seconds
    max_delay = 480  # Max delay in seconds
    retries = 0
    
    while retries < max_retries:
        #print("Making API call...")
        
        result_container = [None, None]
        done_event = threading.Event()
        
        api_thread = threading.Thread(target=api_call_wrapper, args=(system_prompt, prompt, result_container, done_event))
        api_thread.start()
        api_thread.join(timeout=manual_timeout)
        
        if done_event.is_set():
            response, error = result_container
            if response:
                clean_response = response.choices[0].message.content.strip().lower()
                return clean_response
            elif isinstance(error, requests.exceptions.Timeout):
                retries += 1
                delay = base_delay * (2 ** retries)  # Exponential backoff
                delay = min(max_delay, delay) + random.uniform(0, 0.1*base_delay)  # Add some jitter
                time.sleep(delay)
                print("Timeout exception")
            else:
                raise error  # Some other exception occurred
        else:
            # The manual timeout was triggered
            retries += 1
            print(f"Manual timeout triggered after {manual_timeout} seconds")
            
    return None  # Default return value if all retries fail

def contains_whole_words(sentence, terms_to_check):
    
    """Check whether the sentence contains any words in a specified list and report the count."""

    sentence_text = sentence.lower()
    count = 0
    
    for word in terms_to_check:
        matches = re.findall(rf'\b{re.escape(word)}\b', sentence_text)
        count += len(matches)
    
    return count

def get_decision_for_sentence(sentence, context_window, analysis_type, terms_to_check):
    
    """Get a decision for the current sentence, based on the current type of analysis."""
    
    if analysis_type in classification_analyses:
        resource_package = 'flashqda.prompts'
        resource_path = f'{analysis_type}.txt'
        system_prompt = "You are a helpful assistant that classifies sentences. Follow the user's instructions carefully. Respond using JSON."
        
        # Read the prompt file content
        with res.open_text(resource_package, resource_path) as file:
            prompt = file.read()
        prompt = prompt.format(sentence=sentence, context_window=context_window)

        # Sent to OpenAI
        decision = send_to_openai(system_prompt, prompt)

    elif analysis_type in extraction_analyses:
        resource_package = 'flashqda.prompts'
        resource_path = f'{analysis_type}.txt'
        system_prompt = "You are a helpful assistant that extracts and lists causal relationships from sentences. Follow the user's instructions carefully. Respond using JSON."
        
        # Read the prompt file content
        with res.open_text(resource_package, resource_path) as file:
            prompt = file.read()
        prompt = prompt.format(sentence=sentence, context_window=context_window)

        # Sent to OpenAI
        decision = send_to_openai(system_prompt, prompt)

    elif analysis_type == 'list_of_terms':
        decision = contains_whole_words(sentence, terms_to_check)
    return decision

def count_decisions_for_sentence(analysis_type, decisions):
    
    """For the current sentence and the current type of analysis, count the number of decisions that match the specified criteria."""
    
    if analysis_type == "tenses":
        decisions_count = {
            'simple past': decisions.count('simple past'),
            'past perfect': decisions.count('past perfect'),
            'past continuous': decisions.count('past continuous'),
            'past perfect continuous': decisions.count('past perfect continuous'),
            'simple present': decisions.count('simple present') + decisions.count('present simple'),
            'present continuous': decisions.count('present continuous'),
            'present perfect': decisions.count('present perfect'),
            'present perfect continuous': decisions.count('present perfect continuous'),
            'simple future': decisions.count('simple future'),
            'future perfect': decisions.count('future perfect'),
            'future continuous': decisions.count('future continuous'),
            'future perfect continuous': decisions.count('future perfect continuous')
        }
    
    elif analysis_type == "relationships_classify":
        decisions_count = {
            'causal': decisions.count('causal'),
            'correlational': decisions.count('correlational'),
            'none': decisions.count('none')
        }
    
    elif analysis_type == "relationships_extract":
        decisions_count = decisions

    elif analysis_type == "list_of_terms":
        decisions_count = decisions[0]
    
    return decisions_count

def calculate_subscore_for_sentence(analysis_type, decisions, subscore_basis):

    """For the current sentence and the current type of analysis, calculate a subscore."""

    if subscore_basis is None:
        subscore_basis = []    

    if len(decisions) > 0:
        if analysis_type == "tenses":
            subscore = sum(decisions.count(key) for key in subscore_basis) / len(decisions)
        
        elif analysis_type == "relationships_classify":
            subscore = sum(decisions.count(key) for key in subscore_basis) / len(decisions)
        
        elif analysis_type == "relationships_extract":
            subscore = None

        elif analysis_type == "list_of_terms":
            subscore = decisions[0] if decisions[0] <= 1 else 1
    else:
        subscore = 0
    
    return subscore

def analyze_the_current_sentence(sentence, context_window, analysis_type, subscore_basis, terms_to_check, query_count):
    
    """Analyze the current sentence based on the specified analysis type."""
    
    decisions = []
    
    # Classify the current sentence by tense or relationship type (causal, correlational, none)
    if analysis_type in classification_analyses:    
        type = "tenses" if analysis_type == "tenses" else "relationships"
        for _ in range(query_count):
            decision = get_decision_for_sentence(sentence["sentence"], context_window, analysis_type, terms_to_check)
            data = json.loads(decision)
            for key, value in data[type].items():
                decisions.append(value)
    
    # Extract causal relationships from the current sentence
    elif analysis_type in extraction_analyses:
        decision = get_decision_for_sentence(sentence["sentence"], context_window, analysis_type, terms_to_check)
        decisions = decision        
    
    # Detect keywords in the current sentence
    elif analysis_type == 'list_of_terms':
        decision = get_decision_for_sentence(sentence["sentence"], context_window, analysis_type, terms_to_check)
        decisions.append(decision)
    
    decisions_count = count_decisions_for_sentence(analysis_type, decisions)
    subscore = calculate_subscore_for_sentence(analysis_type, decisions, subscore_basis)
    
    return decisions_count, subscore

def initialize_files_analysis(directory, save_name, analysis_type, terms_to_check):
    results_folder = os.path.join(directory, 'Results', save_name)
    results_file = os.path.join(results_folder, f'{save_name}_sentences.csv')
    
    temp_folder = os.path.join(results_folder, 'temp')
    log_folder = os.path.join(results_folder, 'log')
    
    if analysis_type == "list_of_terms":
        temp_file = os.path.join(temp_folder, f'{save_name}_{analysis_type}_{terms_to_check[0]}_temp.csv')
        log_file = os.path.join(log_folder, f'{save_name}_{analysis_type}_{terms_to_check[0]}_log.csv')
    else:
        temp_file = os.path.join(temp_folder, f'{save_name}_{analysis_type}_temp.csv')
        log_file = os.path.join(log_folder, f'{save_name}_{analysis_type}_log.csv')

    if not os.path.exists(log_folder):
        os.makedirs(results_folder, exist_ok = True)
        os.makedirs(log_folder, exist_ok = True)
        os.makedirs(temp_folder, exist_ok = True)
    
    # Ensure results file exists or create an empty one
    if not os.path.exists(results_file):
        pd.DataFrame().to_csv(results_file, index=False)
    
    # Ensure log file exists or create with default values
    if not os.path.exists(log_file):
        with open(log_file, 'w') as log:
            writer = csv.writer(log)
            writer.writerow(['start_time', 'end_time', 'last_processed_document', 'last_processed_sentence'])
            writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 0, 1, -1])

    # Initialize temp file with empty DataFrame if it doesn't exist
    if not os.path.exists(temp_file):
        pd.DataFrame(columns=["document_id", "filename", "sentence_id", "sentence"]).to_csv(temp_file, index=False)

    # Read existing results from temp file if it exists, otherwise initialize as empty list
    try:
        existing_results = pd.read_csv(temp_file).to_dict('records')
    except EmptyDataError:
        existing_results = []

    return temp_file, log_file, existing_results

def get_start_ids(log_file):
    try:
        with open(log_file, 'r') as log:
            reader = csv.reader(log)
            next(reader)  # Skip header
            row = next(reader)
            start_time = row[0]
            end_time = row[1]
            start_document_id, start_sentence_id = map(int, row[2:])
    except EmptyDataError:
        start_time, end_time, start_document_id, start_sentence_id = 0, 0, 1, -1  # Default values

    return start_time, end_time, start_document_id, start_sentence_id

def update_log(log_file, start_time, end_time, document_id, sentence_id):
    with open(log_file, 'w') as log:
        writer = csv.writer(log)
        writer.writerow(['start_time', 'end_time', 'last_processed_document', 'last_processed_sentence'])
        writer.writerow([start_time, end_time, document_id, sentence_id])

def append_to_context(context_window, sentence, context_length):
    if context_length > 0:
        if len(context_window) == context_length:
            context_window.pop(0)
        context_window.append(sentence)
    return context_window

def check_filter_for_analysis(sentence, context_window, analysis_type, filter, subscore_basis, filter_cutoff, terms_to_check, query_count):
    decisions_count, subscore = None, None

    if filter:
        if filter == 'score':
            success = sentence.get(filter, None)
        else:
            success = sentence.get(f'subscore_{filter}', None)
        if success not in ['', 'nan', None]:
            success = float(success)
            if success >= filter_cutoff:
                decisions_count, subscore = analyze_the_current_sentence(sentence, context_window, analysis_type, subscore_basis, terms_to_check, query_count)
    else:
        decisions_count, subscore = analyze_the_current_sentence(sentence, context_window, analysis_type, subscore_basis, terms_to_check, query_count)

    return decisions_count, subscore

def handle_classified_sentence(sentence, analysis_type, decisions_count, subscore, existing_results, terms_to_check):
    
    classified_sentence = {}

    for key, value in sentence.items():
        if key not in classified_sentence:
            classified_sentence[key] = value

    if analysis_type in classification_analyses:
        classified_sentence.update({
            f"decisions_{analysis_type}": decisions_count,
            f"subscore_{analysis_type}": subscore
        })
    elif analysis_type == "list_of_terms":
        classified_sentence.update({
            f"decisions_{analysis_type}_{terms_to_check[0]}": decisions_count,
            f"subscore_{analysis_type}_{terms_to_check[0]}": subscore
        })
    else:
        if decisions_count:
            i = 1
            data = json.loads(decisions_count)
            for relationship in data["relationships"]:
                classified_sentence.update({
                    "cause": relationship["cause"],
                    "effect": relationship["effect"],
                    "relationship_id": i
                })
                existing_results.append(classified_sentence.copy())
                i += 1
            return
        else:
            classified_sentence.update({"cause": None, "effect": None})

    existing_results.append(classified_sentence)

def document_analysis(log_file, filter, subscore_basis, filter_cutoff, terms_to_check):
    with open(log_file, 'a', newline='') as log:
        writer = csv.writer(log)
        writer.writerow(['filter', 'subscore_basis', 'filter_cutoff', 'terms_to_check'])
        writer.writerow([filter, subscore_basis, filter_cutoff, terms_to_check])

def analyze_sentences(sentences, save_name, analysis_type, context_length=0, terms_to_check=[], subscore_basis=None, filter=None, filter_cutoff=0, query_count=3, directory = '.'):
    temp_file, log_file, existing_results = initialize_files_analysis(directory, save_name, analysis_type, terms_to_check)
    start_time, end_time, start_document_id, start_sentence_id = get_start_ids(log_file)

    print(f"Starting '{analysis_type}' analysis with {subscore_basis} subscore basis, '{filter}' filter (filter cutoff: {filter_cutoff}) at document {start_document_id}, sentence {start_sentence_id}")

    context_window = []
    for sentence in tqdm(sentences):
        document_id = int(sentence["document_id"])
        sentence_id = int(sentence["sentence_id"])

        # Skip the document if all sentences have been analyzed (if not, skip all sentences that have been analyzed)
        if document_id < start_document_id or (document_id == start_document_id and sentence_id <= start_sentence_id):
            continue

        # Reset context window at the start of each document
        if sentence_id == 1:
            context_window = []
        # Todo: Recreate context window if analysis interrupted and restarted midway

        decisions_count, subscore = check_filter_for_analysis(sentence, context_window, analysis_type, filter, subscore_basis, filter_cutoff, terms_to_check, query_count)
        context_window = append_to_context(context_window, sentence["sentence"], context_length)
        handle_classified_sentence(sentence, analysis_type, decisions_count, subscore, existing_results, terms_to_check)
        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            pd.DataFrame(existing_results).to_csv(temp_file, index=False)
            update_log(log_file, start_time, end_time, document_id, sentence_id)
        except KeyboardInterrupt:
            print(f"Saving interrupted at document {document_id}, sentence {sentence_id}. Continuing until saving completed.")
            pd.DataFrame(existing_results).to_csv(temp_file, index=False)
            update_log(log_file, start_time, end_time, document_id, sentence_id)
            raise

    results_folder = os.path.join('Results', save_name)
    pd.DataFrame(existing_results).to_csv(os.path.join(results_folder, f'{save_name}_sentences.csv'), index=False)
    document_analysis(log_file, filter, subscore_basis, filter_cutoff, terms_to_check)

    return existing_results
    
def score_sentences(sentences, subscores, weights, save_name):
    """Calculate score for the list of sentences."""
    TEMP_FILE = os.path.join('Results' + save_name + '_scores_temp.csv')

    scored_sentences = []

    for sentence in sentences:
        score = 0
        for i in range(len(subscores)):
            for key in sentence.keys():
                if key == 'subscore_' + subscores[i]:
                    if sentence[key] not in ['', 'nan']:
                        score += float(sentence[key]) * weights[i]
                        break

        scored_sentence = {
            "score": round(score * 100, 2)
        }

        # Dynamically add keys from the CSV file to classified_sentence
        for key, value in sentence.items():
            if key not in scored_sentence:
                scored_sentence[key] = value

        scored_sentences.append(scored_sentence)  # Append modified dictionary to new list

    # Save the list of sentences as a csv file
    with open(TEMP_FILE, 'w') as f:
        pd.DataFrame(scored_sentences).to_csv(f, index=False)
    results_folder = 'Results' + save_name + '/'
    save_to_csv(scored_sentences, results_folder, save_name + '_sentences.csv')
    os.remove(TEMP_FILE)

def retrieve_concepts(save_name):
    df = pd.read_csv(os.path.join("Results", save_name))

    causes = df["cause"].tolist()
    effects = df["effect"].tolist()

    all_concepts = causes + effects
    valid_concepts = [concept for concept in all_concepts if pd.notna(concept)]

    concepts = list(set(valid_concepts))
    
    return concepts

def create_matrix(concepts):
    size = len(concepts)

    matrix = np.zeros((size, size), dtype = int)
    matrix = pd.DataFrame(matrix, index = concepts, columns = concepts)

    return matrix

def initialize_files_comparison(directory, save_name):

    results_folder = os.path.join(directory, 'Results', save_name)
    results_file = os.path.join(results_folder, f'{save_name}_comparison.csv')
    
    temp_folder = os.path.join(results_folder, 'temp')
    temp_file = os.path.join(temp_folder, f'{save_name}_comparison_temp.csv')    

    log_folder = os.path.join(results_folder, 'log')
    log_file = os.path.join(log_folder, f'{save_name}_comparison_log.csv')

    if not os.path.exists(log_folder):
        os.makedirs(results_folder, exist_ok = True)
        os.makedirs(log_folder, exist_ok = True)
        os.makedirs(temp_folder, exist_ok = True)
    
    # Ensure results file exists or create an empty one
    if not os.path.exists(results_file):
        
        # Create a list of concepts from causes and effects
        concepts = retrieve_concepts(save_name)

        # Create a symmetrical matrix from the list of concepts
        matrix_df = create_matrix(concepts)
        matrix_df.to_csv(results_file)
    
    # Ensure log file exists or create with default values
    if not os.path.exists(log_file):
        with open(log_file, 'w') as log:
            writer = csv.writer(log)
            writer.writerow(['start_time', 'end_time', 'last_processed_concept_1', 'last_processed_concept_2', 'concepts'])
            writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 0, 0, 0, len(concepts)])

    # Initialize temp file with empty DataFrame if it doesn't exist
    if not os.path.exists(temp_file):
        matrix_df.to_csv(temp_file)
    else:
        matrix_df = pd.read_csv(temp_file, index_col = 0)
        concepts = matrix_df.index.tolist()

    return results_file, temp_file, log_file, matrix_df, concepts

def get_start_ids_comparison(log_file):
    try:
        with open(log_file, 'r') as log:
            reader = csv.reader(log)
            next(reader)  # Skip header
            row = next(reader)
            start_time = row[0]
            end_time = row[1]
            start_concept_1_id, start_concept_2_id, num_concepts = map(int, row[2:])
    except EmptyDataError:
        start_time, end_time, start_concept_1_id, start_concept_2_id, num_concepts = 0, 0, 0, 0, 0  # Default values

    return start_time, end_time, start_concept_1_id, start_concept_2_id, num_concepts

def update_log_comparison(log_file, start_time, end_time, concept_1_id, concept_2_id, num_concepts):
    with open(log_file, 'w') as log:
        writer = csv.writer(log)
        writer.writerow(['start_time', 'end_time', 'last_processed_concept_1', 'last_processed_concept_2', 'num_concepts'])
        writer.writerow([start_time, end_time, concept_1_id, concept_2_id, num_concepts])

def handle_concept_pair(concept_1, concept_2, query_count):

    decisions = []
    
    # Compare the pair of concepts
    for _ in range(query_count):

        analysis_type = "compare_concepts"
        resource_package = 'flashqda.prompts'
        resource_path = f'{analysis_type}.txt'
        system_prompt = "You are a helpful assistant that compares concepts. Follow the user's instructions carefully. Respond using JSON."
        
        # Read the prompt file
        with res.open_text(resource_package, resource_path) as file:
            prompt = file.read()
        prompt = prompt.format(concept_1 = concept_1, concept_2 = concept_2)
        
        # Send to OpenAI
        decision = send_to_openai(system_prompt, prompt)

        data = json.loads(decision)
        for key, value in data.items():
            decisions.append(value)

    similarity = (decisions.count("equivalent") * 1 + 
                    decisions.count("very similar") * 0.75 + 
                    decisions.count("somewhat similar") * 0.50 +
                    decisions.count("not very similar") * 0.25 +
                    decisions.count("unrelated") * 0) / len(decisions)
    
    return similarity

def compare_concepts(project_name, file_name, query_count=3):

    # Compare each pair of concepts
    results_file, temp_file, log_file, matrix_df, concepts = initialize_files_comparison(directory, save_name)
    start_time, end_time, start_concept_1_id, start_concept_2_id, num_concepts = get_start_ids_comparison(log_file)
    print(f"Starting concept comparison at concept {start_concept_1_id}, concept {start_concept_2_id}")

    for i in tqdm(range(num_concepts)):

        if i >= start_concept_1_id:

            for j in tqdm(range(num_concepts)):
                
                if j >= start_concept_2_id:

                    if i == j:
                        continue
                    if i > j:
                        matrix_df.iloc[i, j] = matrix_df.iloc[j, i]
                    else:
                        similarity = handle_concept_pair(concepts[i], concepts[j], query_count)
                        matrix_df.iloc[i, j] = similarity

                end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                try:
                    pd.DataFrame(matrix_df).to_csv(temp_file)
                    update_log_comparison(log_file, start_time, end_time, i, j, num_concepts)
                except KeyboardInterrupt:
                    print(f"Saving interrupted at concept {i}, concept {j}. Continuing until saving completed.")
                    pd.DataFrame(matrix_df).to_csv(temp_file)
                    update_log_comparison(log_file, start_time, end_time, i, j, num_concepts)
                    raise

                start_concept_2_id += 1

            start_concept_2_id = 0
            start_concept_1_id += 1

    pd.DataFrame(matrix_df).to_csv(results_file)
    print(f"Finished comparing concepts at concept {i+1}, concept {j+1}. Results saved at {results_file}.")