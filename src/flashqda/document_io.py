## document_io.py
import os

def get_documents(project):
    """
    Collect .txt documents from the data folder.

    """

    documents = [] # Clears the document list
    
    # Get a list of all .txt files in the folder
    txt_files = [f for f in os.listdir(project.data) if f.endswith(".txt")]

    for txt_file in txt_files:
        filename = txt_file
        txt_path = project.data / txt_file

        # Read text from the txt file
        with open(txt_path, mode = 'r', encoding='utf-8') as txt_file:
            text = txt_file.read()

        document = {
            "filename": filename,
            "text": text
        }
        documents.append(document)

    return documents