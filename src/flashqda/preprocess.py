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
    Segment documents into paragraphs or sentences and save as CSV.

    Supported file formats: .txt, .pdf (requires pdfplumber), .docx (requires python-docx).
    All files in project.data/ with a supported extension are processed in alphabetical order.
    Files with unrecognised extensions are skipped with a printed warning.

    Args:
        project (flashqda.ProjectContext): Project context providing the data directory.
        granularity (str, optional): "sentence" (default) or "paragraph".
        custom_items: Deprecated. Retained for backward compatibility; has no effect.
        save_name (str, optional): Output CSV filename. Defaults to "{granularity}.csv".

    Returns:
        None: Writes a CSV to project.data with columns:
            document_id, filename, {granularity}_id, {granularity}.
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
