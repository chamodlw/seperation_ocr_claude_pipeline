"""
Sinhala Text Cleaning Utilities
================================
Used by inference and post-processing steps.
"""

import re
import unicodedata


def clean_sinhala_text(text: str) -> str:
    """
    Clean OCR-extracted Sinhala text:
    - Normalize Unicode (NFC)
    - Keep Sinhala characters, digits, and common punctuation
    - Collapse whitespace
    """
    if not text:
        return ''

    # Unicode normalization
    text = unicodedata.normalize('NFC', text)

    # Keep Sinhala Unicode block (U+0D80–U+0DFF), digits, spaces, punctuation
    text = re.sub(r'[^\u0D80-\u0DFF\u0020-\u0040\u005B-\u0060\u007B-\u007E0-9\s]', '', text)

    # Collapse multiple spaces / newlines
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def clean_article(article: dict) -> dict:
    """Clean news text of an extracted article."""
    if 'news' in article:
        return {
            'news': clean_sinhala_text(article.get('news', '')),
        }
    # Backwards compatibility for headline/body format
    return {
        'headline': clean_sinhala_text(article.get('headline', '')),
        'body':     clean_sinhala_text(article.get('body', '')),
    }


def clean_all_results(results: dict) -> dict:
    """Clean all articles in a results dictionary."""
    return {
        page: [clean_article(a) for a in articles]
        for page, articles in results.items()
    }
