## preprocess.py

import pandas as pd
from flashqda.text_utils import segment_sentences, segment_paragraphs
from flashqda.document_io import get_documents
from tqdm import tqdm
import re, os

def preprocess_documents(
        project, 
        granularity=None,
        custom_items=[],
        save_name=None
        ):
    """
    Segment .txt documents into paragraphs or sentences and save them in a csv file.
     
    The output file will include a document_id for each document, the filename of each document, an item_id for each item, and the items.

    Args:
        project (flashqda.ProjectContext): A Project object representing document(s) to process.
        granularity (str, optional): Segmentation level. Options are "sentence" and "paragraph". 
            Defaults to "sentence".
        custom_items (str or list of str, optional): Custom strings containing punctuation that should not be treated as sentence boundaries (e.g., abbreviations). 
            For multiple items, use a list of quoted strings, e.g., ["i.tem", "it.em"].
        save_name (str, optional): Name for csv file that stores the segmented items.
            Defaults to "sentences.csv" or "paragraphs.csv".
    
    Returns:
        None: The function writes a CSV file to disk and returns nothing.
    """

    granularity = granularity if granularity in ("sentence", "paragraph") else "sentence"
    save_name = save_name if save_name else f"{granularity}.csv"

    all_items = [] # Clears the all_items list
    document_id_counter = 1  # Initialize a document ID counter outside the loop
    
    documents = get_documents(project)

    updated_count = 0

    # Loop through the documents
    for document in tqdm(documents):
        item_id_counter = 0  # Initialize an item ID counter
        filename = document["filename"]
        text = document["text"]
        
        if granularity == "sentence":
            segmented_document = segment_sentences(text, custom_items)
            items = segmented_document # Extract the sentences from the segmented document
        else:
            segmented_document = segment_paragraphs(text)
            items = segmented_document
        
        # Extend the all_sentences list with document ID, sentence ID, filename, and sentence
        all_items.extend([{
            "document_id": document_id_counter,
            "filename": filename,
            f"{granularity}_id": item_id_counter + idx,
            f"{granularity}": item
        } for idx, item in enumerate(items, start=1)])
        
        # Increment the sentence ID counter for the next document
        item_id_counter += len(items)
        updated_count += item_id_counter

        # Increment the document ID counter for the next document
        document_id_counter += 1

    # Save the list of sentences as a csv file
    project.save_data(pd.DataFrame(all_items), save_name)

    num_docs = document_id_counter - 1
    print(f"Segmented {updated_count} {granularity}s in {num_docs} documents.")
