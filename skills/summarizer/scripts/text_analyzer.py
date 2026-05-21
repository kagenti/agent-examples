"""
Text Analysis Module for Summarization

Provides functions to analyze text and identify important sentences,
key terms, and structural elements for effective summarization.
"""

import re
from collections import Counter
from typing import Dict, List, Tuple


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into words, removing punctuation and converting to lowercase.

    Args:
        text: Input text to tokenize

    Returns:
        List of lowercase word tokens
    """
    # Remove punctuation and convert to lowercase
    text = re.sub(r"[^\w\s]", " ", text.lower())
    # Split into words and filter empty strings
    words = [word for word in text.split() if word]
    return words


def get_stopwords() -> set:
    """
    Return a set of common English stopwords.

    Returns:
        Set of stopword strings
    """
    return {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "will",
        "with",
        "this",
        "but",
        "they",
        "have",
        "had",
        "what",
        "when",
        "where",
        "who",
        "which",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "can",
        "just",
        "should",
        "now",
        "also",
        "been",
        "were",
        "their",
        "there",
        "if",
    }


def extract_key_terms(text: str, top_n: int = 10) -> List[Tuple[str, int]]:
    """
    Extract the most frequent meaningful terms from text.

    Args:
        text: Input text to analyze
        top_n: Number of top terms to return

    Returns:
        List of (term, frequency) tuples, sorted by frequency
    """
    words = tokenize(text)
    stopwords = get_stopwords()

    # Filter out stopwords and short words
    meaningful_words = [w for w in words if w not in stopwords and len(w) > 3]

    # Count frequencies
    word_freq = Counter(meaningful_words)

    return word_freq.most_common(top_n)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using basic punctuation rules.

    Args:
        text: Input text to split

    Returns:
        List of sentence strings
    """
    # Split on sentence-ending punctuation
    sentences = re.split(r"[.!?]+", text)

    # Clean up and filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def score_sentence(sentence: str, key_terms: List[str]) -> float:
    """
    Score a sentence based on presence of key terms.

    Args:
        sentence: Sentence to score
        key_terms: List of important terms

    Returns:
        Score (higher is more important)
    """
    sentence_lower = sentence.lower()
    words = tokenize(sentence)

    # Base score: sentence length (normalized)
    score = len(words) / 100.0

    # Add points for each key term present
    for term in key_terms:
        if term in sentence_lower:
            score += 1.0

    # Bonus for sentences with numbers (often contain facts)
    if re.search(r"\d+", sentence):
        score += 0.5

    # Bonus for sentences with quotes (often important)
    if '"' in sentence or "'" in sentence:
        score += 0.3

    # Penalty for very short sentences
    if len(words) < 5:
        score *= 0.5

    # Penalty for very long sentences (harder to extract)
    if len(words) > 40:
        score *= 0.8

    return score


def analyze_text(text: str) -> Dict:
    """
    Perform comprehensive text analysis for summarization.

    Args:
        text: Input text to analyze

    Returns:
        Dictionary containing:
        - key_terms: List of (term, frequency) tuples
        - sentences: List of sentences
        - sentence_scores: List of (sentence, score) tuples
        - word_count: Total word count
        - sentence_count: Total sentence count
    """
    # Extract key terms
    key_terms = extract_key_terms(text, top_n=15)
    key_term_words = [term for term, _ in key_terms]

    # Split into sentences
    sentences = split_into_sentences(text)

    # Score each sentence
    sentence_scores = [(sent, score_sentence(sent, key_term_words)) for sent in sentences]

    # Sort by score (descending)
    sentence_scores.sort(key=lambda x: x[1], reverse=True)

    # Count words
    words = tokenize(text)

    return {
        "key_terms": key_terms,
        "sentences": sentences,
        "sentence_scores": sentence_scores,
        "word_count": len(words),
        "sentence_count": len(sentences),
    }


def extract_key_sentences(text: str, num_sentences: int = 5) -> List[str]:
    """
    Extract the most important sentences from text.

    Args:
        text: Input text to analyze
        num_sentences: Number of sentences to extract

    Returns:
        List of key sentences in order of importance
    """
    analysis = analyze_text(text)

    # Get top N sentences by score
    top_sentences = [sent for sent, score in analysis["sentence_scores"][:num_sentences]]

    return top_sentences


def identify_structure(text: str) -> Dict:
    """
    Identify structural elements in the text.

    Args:
        text: Input text to analyze

    Returns:
        Dictionary with structural information:
        - has_headers: Whether text contains markdown-style headers
        - has_lists: Whether text contains bullet points or numbered lists
        - has_code: Whether text contains code blocks
        - paragraphs: Number of paragraphs
    """
    # Check for headers (lines starting with #)
    has_headers = bool(re.search(r"^#+\s", text, re.MULTILINE))

    # Check for lists (lines starting with -, *, or numbers)
    has_lists = bool(re.search(r"^[\s]*[-*•]\s", text, re.MULTILINE)) or bool(
        re.search(r"^[\s]*\d+\.\s", text, re.MULTILINE)
    )

    # Check for code blocks
    has_code = "```" in text or "    " in text

    # Count paragraphs (separated by blank lines)
    paragraphs = len(re.split(r"\n\s*\n", text.strip()))

    return {"has_headers": has_headers, "has_lists": has_lists, "has_code": has_code, "paragraphs": paragraphs}


if __name__ == "__main__":
    # Example usage
    sample_text = """
    Artificial intelligence is transforming the technology industry. Machine learning
    algorithms can now process vast amounts of data in seconds. Companies are investing
    billions in AI research and development. The impact on jobs and society is significant.
    Experts predict AI will create 97 million new jobs by 2025. However, it may also
    displace 85 million jobs in the same period. Education and retraining programs are
    essential for workers to adapt to this change.
    """

    print("Text Analysis Example")
    print("=" * 50)

    analysis = analyze_text(sample_text)

    print(f"\nWord Count: {analysis['word_count']}")
    print(f"Sentence Count: {analysis['sentence_count']}")

    print("\nTop Key Terms:")
    for term, freq in analysis["key_terms"][:5]:
        print(f"  - {term}: {freq}")

    print("\nTop 3 Sentences:")
    for i, (sent, score) in enumerate(analysis["sentence_scores"][:3], 1):
        print(f"  {i}. [{score:.2f}] {sent}")

    print("\nKey Sentences:")
    for sent in extract_key_sentences(sample_text, num_sentences=3):
        print(f"  • {sent}")

# Made with Bob
