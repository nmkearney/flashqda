from .general import save_to_csv, read_csv_file
from tqdm import tqdm
import re, os, nltk
nltk.download('punkt_tab')

def segment_document(document_text, custom_items):
    """
    Segment .txt documents into sentences.
    """

    # Replace non-breaking spaces with regular spaces
    document_text = document_text.replace("\u00A0", " ")

    # Replace 'e. ' and 'i. ' with 'e.' and 'i.' respectively
    document_text = re.sub(r'\b(e\.)\s+(?=\S)', r'\1', document_text)
    document_text = re.sub(r'\b(i\.)\s+(?=\S)', r'\1', document_text)

    # Search for instances of items and check for comma after instance; if no comma, add one
    items_to_check = ['E.g.', 'e.g.', 'i.e.', 'et al.', 'Fig.', 'fig.', 'Figs.', 'ca.', 'c.', 'Eq.', 'eq.', 'approx.', 'Mr.', 'Ms.', 'Mrs.', 'Dr.'] + custom_items
    for item in items_to_check:
        document_text = re.sub(r'({0})(?![,])'.format(re.escape(item)), r'\1,', document_text)

    # Use regex to extract sentences that start with a capital letter and end with a valid sentence ending
    sentence_pattern = r'([A-Z].*?[.!?]["\'\)\]]?(?:\s|$))'
    # [A-Z] Match an uppercase letter (start of a sentence)
    # .*? Match any characters (non-greedy)
    # [``.!?] Match a valid sentence-ending punctuation (period, exclamation point, or question mark)
    # ["\'\)\]]? Match optional quotation mark or bracket
    # (?:\s|$) Match a whitespace character or end of line
    extracted_sentences = re.findall(sentence_pattern, document_text)

    # Exclude sentences that contain carriage returns
    extracted_sentences = [sentence for sentence in extracted_sentences if '\r' not in sentence]
    
    # Join the extracted sentences to form the preprocessed article
    segmented_document = ''.join(extracted_sentences)
    
    return segmented_document

def get_documents():
    """
    Collect .txt documents from the data folder.

    """

    documents = [] # Clears the document list
    
    # Get a list of all .txt files in the folder
    txt_files = [f for f in os.listdir('Data') if f.endswith(".txt")]

    for txt_file in txt_files:
        filename = txt_file
        txt_path = os.path.join('Data', txt_file)

        # Read text from the txt file
        with open(txt_path, 'r', encoding='utf-8') as txt_file:
            text = txt_file.read()

        document = {
            "filename": filename,
            "text": text
        }
        documents.append(document)

    return documents

def preprocess_documents(save_name, custom_items=[]):
    """
    Segment .txt documents into sentences and save the sentences in a csv file with the filename,
    a filename ID, the sentence, and a sentence ID.

    param save_name: Name for csv file that stores the segmented sentences
    """
    #Creates results folder if does not already exist
    results_folder = os.path.join('Results', save_name)
    log_folder = os.path.join(results_folder, 'log')
    temp_folder = os.path.join(results_folder, 'temp')
    if not os.path.exists(log_folder):
        os.makedirs(results_folder)
        os.makedirs(log_folder)
        os.makedirs(temp_folder)

    all_sentences = [] # Clears the all_sentences list
    document_id_counter = 1  # Initialize a document ID counter outside the loop
    
    documents = get_documents()

    # Loop through the documents
    for document in tqdm(documents):
        sentence_id_counter = 0  # Initialize a sentence ID counter
        filename = document["filename"]
        text = document["text"]
        segmented_document = segment_document(text, custom_items)
        sentences = nltk.sent_tokenize(segmented_document) # Extract the sentences from the segmented document
        
        # Extend the all_sentences list with document ID, sentence ID, filename, and sentence
        all_sentences.extend([{
            "document_id": document_id_counter,
            "filename": filename,
            "sentence_id": sentence_id_counter + idx,
            "sentence": sentence
        } for idx, sentence in enumerate(sentences, start=1)])
        
        # Increment the sentence ID counter for the next document
        sentence_id_counter += len(sentences)

        # Increment the document ID counter for the next document
        document_id_counter += 1

    # Save the list of sentences as a csv file
    save_to_csv(all_sentences, results_folder, save_name + '_sentences.csv')