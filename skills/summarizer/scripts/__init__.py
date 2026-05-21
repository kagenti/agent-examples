"""
Summarizer Scripts Package

Helper modules for text summarization.
"""

from .formatter import (
    format_as_bullets,
    format_as_markdown,
    format_as_numbered_list,
    format_executive_summary,
    format_meeting_summary,
    format_progressive_summary,
    format_technical_summary,
)
from .summarizer import (
    create_bullet_summary,
    create_executive_summary,
    create_meeting_summary,
    create_progressive_summary,
    create_structured_summary,
    extract_action_items,
    extract_key_data,
    extract_key_points,
)
from .text_analyzer import analyze_text, extract_key_sentences, extract_key_terms, identify_structure

__all__ = [
    # Text Analyzer
    "analyze_text",
    "extract_key_sentences",
    "extract_key_terms",
    "identify_structure",
    # Summarizer
    "create_bullet_summary",
    "create_executive_summary",
    "extract_key_points",
    "create_structured_summary",
    "create_progressive_summary",
    "extract_action_items",
    "extract_key_data",
    "create_meeting_summary",
    # Formatter
    "format_as_markdown",
    "format_as_bullets",
    "format_as_numbered_list",
    "format_progressive_summary",
    "format_executive_summary",
    "format_meeting_summary",
    "format_technical_summary",
]

# Made with Bob
