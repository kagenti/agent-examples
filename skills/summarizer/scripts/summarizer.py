"""
Summarization Module

Provides various summarization functions including bullet-point summaries,
executive summaries, and key point extraction.
"""

from typing import Dict, List

from .text_analyzer import analyze_text, extract_key_sentences, split_into_sentences, tokenize


def create_bullet_summary(text: str, max_bullets: int = 7) -> List[str]:
    """
    Create a bullet-point summary of the text.

    Args:
        text: Input text to summarize
        max_bullets: Maximum number of bullet points

    Returns:
        List of bullet point strings
    """
    # Extract key sentences
    key_sentences = extract_key_sentences(text, num_sentences=max_bullets)

    # Convert to bullet points (shorten if needed)
    bullets = []
    for sentence in key_sentences:
        # Trim very long sentences
        if len(sentence) > 150:
            words = sentence.split()
            sentence = " ".join(words[:20]) + "..."
        bullets.append(sentence)

    return bullets


def create_executive_summary(text: str, max_length: int = 200) -> str:
    """
    Create a concise executive summary.

    Args:
        text: Input text to summarize
        max_length: Maximum length in words

    Returns:
        Executive summary string
    """
    analysis = analyze_text(text)

    # Get top sentences
    num_sentences = min(3, len(analysis["sentences"]))
    top_sentences = [sent for sent, score in analysis["sentence_scores"][:num_sentences]]

    # Combine sentences
    summary = " ".join(top_sentences)

    # Trim to max length if needed
    words = summary.split()
    if len(words) > max_length:
        summary = " ".join(words[:max_length]) + "..."

    return summary


def extract_key_points(text: str, num_points: int = 5) -> List[str]:
    """
    Extract key points from text as concise statements.

    Args:
        text: Input text to analyze
        num_points: Number of key points to extract

    Returns:
        List of key point strings
    """
    analysis = analyze_text(text)

    # Get top sentences
    top_sentences = [sent for sent, score in analysis["sentence_scores"][:num_points]]

    # Convert to key points (remove unnecessary words)
    key_points = []
    for sentence in top_sentences:
        # Remove common sentence starters
        point = sentence
        for starter in ["However,", "Therefore,", "Moreover,", "Furthermore,", "Additionally,"]:
            if point.startswith(starter):
                point = point[len(starter) :].strip()

        # Capitalize first letter
        if point:
            point = point[0].upper() + point[1:]

        key_points.append(point)

    return key_points


def create_structured_summary(text: str) -> Dict[str, any]:
    """
    Create a structured summary with multiple components.

    Args:
        text: Input text to summarize

    Returns:
        Dictionary with:
        - main_point: Single sentence main point
        - key_points: List of 3-5 key points
        - details: List of supporting details
        - conclusion: Concluding statement
    """
    analysis = analyze_text(text)
    sentences = analysis["sentences"]
    scored_sentences = analysis["sentence_scores"]

    # Main point: highest scoring sentence
    main_point = scored_sentences[0][0] if scored_sentences else ""

    # Key points: next 3-5 highest scoring
    key_points = [sent for sent, score in scored_sentences[1:6]]

    # Details: medium-scoring sentences
    details = [sent for sent, score in scored_sentences[6:10]]

    # Conclusion: last sentence or second-highest scoring
    conclusion = sentences[-1] if sentences else ""
    if len(scored_sentences) > 1:
        conclusion = scored_sentences[1][0]

    return {"main_point": main_point, "key_points": key_points, "details": details, "conclusion": conclusion}


def create_progressive_summary(text: str) -> Dict[str, str]:
    """
    Create a progressive summary with increasing detail levels.

    Args:
        text: Input text to summarize

    Returns:
        Dictionary with:
        - one_sentence: Single sentence summary
        - one_paragraph: One paragraph summary
        - detailed: Detailed multi-paragraph summary
    """
    analysis = analyze_text(text)
    scored_sentences = analysis["sentence_scores"]

    # One sentence: highest scoring
    one_sentence = scored_sentences[0][0] if scored_sentences else ""

    # One paragraph: top 3-4 sentences
    para_sentences = [sent for sent, score in scored_sentences[:4]]
    one_paragraph = " ".join(para_sentences)

    # Detailed: top 8-10 sentences
    detail_sentences = [sent for sent, score in scored_sentences[:10]]
    detailed = " ".join(detail_sentences)

    return {"one_sentence": one_sentence, "one_paragraph": one_paragraph, "detailed": detailed}


def summarize_by_ratio(text: str, ratio: float = 0.3) -> str:
    """
    Summarize text to a specific ratio of original length.

    Args:
        text: Input text to summarize
        ratio: Target ratio (0.0 to 1.0) of original length

    Returns:
        Summary string
    """
    analysis = analyze_text(text)
    total_sentences = len(analysis["sentences"])

    # Calculate number of sentences to include
    target_sentences = max(1, int(total_sentences * ratio))

    # Get top sentences
    top_sentences = [sent for sent, score in analysis["sentence_scores"][:target_sentences]]

    # Combine in original order
    original_sentences = analysis["sentences"]
    summary_sentences = [s for s in original_sentences if s in top_sentences]

    return " ".join(summary_sentences)


def extract_action_items(text: str) -> List[str]:
    """
    Extract action items from text (sentences with action verbs).

    Args:
        text: Input text to analyze

    Returns:
        List of action item strings
    """
    sentences = split_into_sentences(text)

    # Action verbs to look for
    action_verbs = [
        "complete",
        "finish",
        "create",
        "develop",
        "implement",
        "review",
        "update",
        "send",
        "schedule",
        "prepare",
        "analyze",
        "research",
        "contact",
        "follow up",
        "submit",
        "approve",
        "test",
        "deploy",
        "assign",
        "delegate",
        "coordinate",
        "organize",
        "plan",
    ]

    action_items = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        # Check if sentence contains action verbs
        if any(verb in sentence_lower for verb in action_verbs):
            action_items.append(sentence)

    return action_items


def extract_key_data(text: str) -> List[str]:
    """
    Extract sentences containing numbers, dates, or statistics.

    Args:
        text: Input text to analyze

    Returns:
        List of data-rich sentences
    """
    import re

    sentences = split_into_sentences(text)

    data_sentences = []
    for sentence in sentences:
        # Check for numbers, percentages, dates, or currency
        if re.search(r"\d+", sentence):
            data_sentences.append(sentence)

    return data_sentences


def create_meeting_summary(text: str) -> Dict[str, List[str]]:
    """
    Create a structured meeting summary.

    Args:
        text: Meeting notes text

    Returns:
        Dictionary with:
        - decisions: List of decisions made
        - action_items: List of action items
        - key_points: List of key discussion points
    """
    sentences = split_into_sentences(text)

    # Keywords for categorization
    decision_keywords = ["decided", "agreed", "approved", "resolved", "concluded"]
    action_keywords = ["will", "should", "must", "need to", "assigned", "responsible"]

    decisions = []
    action_items = []
    key_points = []

    for sentence in sentences:
        sentence_lower = sentence.lower()

        # Categorize sentence
        if any(kw in sentence_lower for kw in decision_keywords):
            decisions.append(sentence)
        elif any(kw in sentence_lower for kw in action_keywords):
            action_items.append(sentence)
        else:
            # Check if it's a key point (has important terms)
            words = tokenize(sentence)
            if len(words) > 5:  # Substantial sentence
                key_points.append(sentence)

    # Limit to most important
    return {"decisions": decisions[:5], "action_items": action_items[:7], "key_points": key_points[:5]}


if __name__ == "__main__":
    # Example usage
    sample_text = """
    The quarterly review meeting was held on January 15, 2026. The team discussed
    the product launch timeline and budget allocation. It was decided to move the
    launch date to March 1, 2026, to allow for additional testing. The marketing
    budget was approved at $250,000. Sarah will finalize the marketing plan by
    January 22. Mike needs to complete beta testing by February 1. The team agreed
    to review the pricing strategy in the next meeting. Revenue projections show
    a 23% increase over last quarter. Customer satisfaction scores improved to 4.5
    out of 5. The development team identified three critical bugs that must be
    fixed before launch. John will coordinate with the QA team to prioritize these
    issues. The meeting concluded with a commitment to weekly status updates.
    """

    print("Summarization Examples")
    print("=" * 60)

    print("\n1. Bullet Summary:")
    bullets = create_bullet_summary(sample_text, max_bullets=5)
    for bullet in bullets:
        print(f"   • {bullet}")

    print("\n2. Executive Summary:")
    exec_summary = create_executive_summary(sample_text, max_length=50)
    print(f"   {exec_summary}")

    print("\n3. Key Points:")
    points = extract_key_points(sample_text, num_points=4)
    for i, point in enumerate(points, 1):
        print(f"   {i}. {point}")

    print("\n4. Meeting Summary:")
    meeting = create_meeting_summary(sample_text)
    print("   Decisions:")
    for decision in meeting["decisions"]:
        print(f"     - {decision}")
    print("   Action Items:")
    for action in meeting["action_items"]:
        print(f"     - {action}")

    print("\n5. Progressive Summary:")
    progressive = create_progressive_summary(sample_text)
    print(f"   One Sentence: {progressive['one_sentence']}")
    print(f"\n   One Paragraph: {progressive['one_paragraph']}")

# Made with Bob
