##text_utils.py
import re

def segment_sentences(document_text, custom_items):
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
    
    return extracted_sentences

def segment_paragraphs(text):
    # Split by double newlines or other paragraph markers
    return [p.strip() for p in text.split("\n") if p.strip()]
