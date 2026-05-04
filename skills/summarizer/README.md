# Summarizer Skill

A comprehensive skill for creating clear, concise summaries of any type of information.

## Overview

The Summarizer skill helps you create crisp, clear, and effective summaries of documents, articles, meeting notes, technical documentation, and more. It provides both guidance on summarization techniques and Python helper scripts for automated text analysis and summarization.

## Features

- **Multiple Summarization Techniques**: Extractive, abstractive, and hybrid approaches
- **Various Summary Formats**: Executive summaries, bullet points, structured summaries, progressive summaries
- **Domain-Specific Guidelines**: Tailored approaches for technical docs, meetings, research papers, news articles
- **Python Helper Scripts**: Automated text analysis and summarization tools
- **Best Practices**: Comprehensive guidelines for creating effective summaries

## Quick Start

### Using the Skill

When you need to summarize information, this skill provides:

1. **Techniques** for different types of content
2. **Formats** for different use cases
3. **Guidelines** for quality summaries
4. **Scripts** for automated processing

### Using the Python Scripts

```python
from scripts import (
    analyze_text,
    create_bullet_summary,
    create_executive_summary,
    format_as_markdown
)

# Analyze text
analysis = analyze_text(long_document)
print(f"Key terms: {analysis['key_terms']}")

# Create bullet summary
bullets = create_bullet_summary(long_document, max_bullets=5)
for bullet in bullets:
    print(f"• {bullet}")

# Create executive summary
exec_summary = create_executive_summary(long_document, max_length=200)
print(exec_summary)
```

## File Structure

```
summarizer/
├── SKILL.md                    # Main skill documentation
├── README.md                   # This file
└── scripts/                    # Python helper scripts
    ├── __init__.py            # Package initialization
    ├── text_analyzer.py       # Text analysis functions
    ├── summarizer.py          # Summarization functions
    └── formatter.py           # Output formatting functions
```

## Python Scripts

### text_analyzer.py

Analyzes text to identify important content:

- `analyze_text(text)` - Comprehensive text analysis
- `extract_key_sentences(text, num_sentences)` - Extract most important sentences
- `extract_key_terms(text, top_n)` - Find most frequent meaningful terms
- `identify_structure(text)` - Identify structural elements

### summarizer.py

Creates various types of summaries:

- `create_bullet_summary(text, max_bullets)` - Bullet-point summary
- `create_executive_summary(text, max_length)` - Executive summary
- `extract_key_points(text, num_points)` - Key points extraction
- `create_structured_summary(text)` - Multi-component structured summary
- `create_progressive_summary(text)` - Progressive detail levels
- `extract_action_items(text)` - Extract action items from meetings
- `extract_key_data(text)` - Extract data-rich sentences
- `create_meeting_summary(text)` - Structured meeting summary

### formatter.py

Formats summaries in different styles:

- `format_as_markdown(summary_dict)` - Markdown with sections
- `format_as_bullets(items)` - Bullet list
- `format_as_numbered_list(items)` - Numbered list
- `format_progressive_summary(summary_dict)` - Progressive format
- `format_executive_summary(...)` - Executive summary format
- `format_meeting_summary(...)` - Meeting summary format
- `format_technical_summary(...)` - Technical documentation format

## Usage Examples

### Example 1: Summarize a Long Article

```python
from scripts import create_bullet_summary, format_as_bullets

article = """
[Long article text here...]
"""

# Create bullet summary
bullets = create_bullet_summary(article, max_bullets=5)

# Format and display
print("Article Summary:")
print(format_as_bullets(bullets))
```

### Example 2: Create Executive Summary

```python
from scripts import create_executive_summary, format_executive_summary

report = """
[Long report text here...]
"""

# Create summary
summary = create_executive_summary(report, max_length=150)

# Format as executive summary
formatted = format_executive_summary(
    title="Q4 Performance Report",
    main_point=summary,
    key_findings=[
        "Revenue up 23%",
        "Customer satisfaction: 4.5/5",
        "Market share increased 5%"
    ],
    recommendation="Continue current strategy with increased marketing spend."
)

print(formatted)
```

### Example 3: Summarize Meeting Notes

```python
from scripts import create_meeting_summary, format_meeting_summary

meeting_notes = """
[Meeting notes text here...]
"""

# Extract meeting components
summary = create_meeting_summary(meeting_notes)

# Format
formatted = format_meeting_summary(
    title="Sprint Planning Meeting",
    date="January 15, 2026",
    decisions=summary['decisions'],
    action_items=summary['action_items']
)

print(formatted)
```

## When to Use This Skill

Use the Summarizer skill when:

- User asks to "summarize", "condense", or "give key points"
- Dealing with long documents or articles
- Creating executive summaries or briefs
- Extracting action items from meetings
- Simplifying complex technical content
- Creating quick reference guides
- Preparing content for different audiences

## Integration with Generic Agent

To use this skill with the generic agent:

```bash
export SKILL_FOLDERS="/path/to/skills/summarizer"
```

The agent will automatically load the skill and make it available for summarization tasks.

## Best Practices

1. **Understand the Purpose**: Know why you're summarizing (quick reference, decision-making, etc.)
2. **Know Your Audience**: Tailor the summary to who will read it
3. **Preserve Key Information**: Don't lose critical context or data
4. **Be Concise**: Aim for 10-30% of original length
5. **Maintain Accuracy**: Verify facts and figures
6. **Use Clear Structure**: Organize logically with headers
7. **Test Comprehension**: Ensure the summary is understandable standalone

## Requirements

The Python scripts use only standard library modules:
- `re` - Regular expressions
- `collections` - Counter for frequency analysis
- `typing` - Type hints

No external dependencies required!

## License

Part of the Kagenti agent-examples project.
