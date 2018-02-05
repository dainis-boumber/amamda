import spacy
from textacy import preprocess as pp

def preprocess(text, fix_unicode=True, normalize_white_space = False, lowercase=False, transliterate=False,
                    no_urls=False, no_emails=False, no_phone_numbers=False,
                    no_numbers=False, no_currency_symbols=False, no_punct=False,
                    no_contractions=False, no_accents=False):
    if normalize_white_space:
        text = pp.normalize_whitespace(text)
    text = pp.preprocess_text(text, fix_unicode, lowercase, transliterate,
                    no_urls, no_emails, no_phone_numbers,
                    no_numbers, no_currency_symbols, no_punct,
                    no_contractions, no_accents)
    return text