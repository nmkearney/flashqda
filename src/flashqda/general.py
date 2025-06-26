# general.py — utility functions for shared logic

import re

def append_to_context(context_window, sentence, context_length):
    if len(context_window) >= context_length:
        context_window.pop(0)
    context_window.append(sentence)
    return context_window

def contains_whole_words(sentence, terms_to_check):
    return any(re.search(rf"\b{re.escape(term)}\b", sentence, re.IGNORECASE) for term in terms_to_check)

def parse_boolean_response(text):
    lowered = text.lower().strip()
    if any(yes in lowered for yes in ["yes", "relevant", "true"]):
        return True
    if any(no in lowered for no in ["no", "irrelevant", "false"]):
        return False
    return None  # Or raise an exception if ambiguity is unacceptable
