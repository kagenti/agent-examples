---
name: summarizer
description: Use this skill when you need to create clear, concise summaries of information. This includes summarizing long documents, articles, meeting notes, technical documentation, research papers, or any text that needs to be condensed while preserving key information. The skill provides techniques for extractive and abstractive summarization, bullet-point formatting, and executive summaries.
tags: [summarization, text-processing, information-extraction, content-condensing]
---

# Information Summarization Skill

## Overview

This skill helps you create crisp, clear, and effective summaries of any type of information. Whether you're dealing with lengthy documents, technical content, meeting notes, or research papers, this skill provides structured approaches and tools to extract and present the most important information concisely.

## Core Principles

### 1. Clarity First
- Use simple, direct language
- Avoid jargon unless necessary
- Define technical terms when used
- Write for your target audience

### 2. Conciseness
- Remove redundant information
- Focus on key points only
- Use active voice
- Eliminate filler words

### 3. Structure
- Start with the most important information
- Use hierarchical organization
- Group related concepts
- Maintain logical flow

### 4. Completeness
- Capture all essential points
- Preserve critical context
- Include relevant data/numbers
- Maintain accuracy

## Summarization Techniques

### Extractive Summarization
Select and extract the most important sentences or phrases directly from the source:

**When to use:**
- Technical documentation
- Legal documents
- Scientific papers
- When exact wording matters

**Approach:**
1. Identify topic sentences
2. Extract key facts and figures
3. Select representative quotes
4. Maintain original phrasing

### Abstractive Summarization
Rewrite and rephrase content in your own words:

**When to use:**
- General articles
- Meeting notes
- Narrative content
- When clarity is more important than exact wording

**Approach:**
1. Understand the core message
2. Identify main themes
3. Rephrase in simpler terms
4. Create new sentences that capture essence

### Hybrid Approach
Combine both techniques for optimal results:
1. Extract key facts and data (extractive)
2. Rephrase explanations (abstractive)
3. Organize logically
4. Add context where needed

## Summary Formats

### Executive Summary
**Length:** 10% of original or 1-2 paragraphs
**Structure:**
- Opening statement (main point)
- Key findings (2-4 bullets)
- Conclusion/recommendation

**Example:**
```
This report analyzes Q4 sales performance. Key findings:
• Revenue increased 23% YoY to $4.2M
• Customer acquisition cost decreased 15%
• Churn rate remained stable at 3.2%
Recommendation: Increase marketing budget by 20% to capitalize on momentum.
```

### Bullet-Point Summary
**Best for:** Quick reference, action items, key takeaways

**Structure:**
- Use parallel structure
- Start with action verbs
- Keep bullets concise (1-2 lines)
- Limit to 5-7 main points

**Example:**
```
Meeting Summary:
• Approved Q1 budget of $500K
• Assigned Sarah to lead product launch
• Scheduled follow-up for March 15
• Identified 3 critical risks requiring mitigation
```

### Structured Summary
**Best for:** Complex documents, research papers, technical reports

**Structure:**
```
**Purpose:** [Why this document exists]
**Key Points:**
1. [First main point]
2. [Second main point]
3. [Third main point]

**Details:**
- [Supporting detail 1]
- [Supporting detail 2]

**Conclusion:** [Final takeaway]
```

### Progressive Summary
**Best for:** Long documents, books, multi-part content

**Structure:**
- One-sentence summary
- One-paragraph summary
- Detailed summary (multiple paragraphs)

## Best Practices

### DO:
✓ Lead with the most important information
✓ Use specific numbers and data
✓ Maintain objectivity
✓ Preserve the author's intent
✓ Use clear section headers
✓ Include source attribution
✓ Verify accuracy of extracted information

### DON'T:
✗ Add your own opinions (unless requested)
✗ Include minor details
✗ Use ambiguous language
✗ Lose critical context
✗ Misrepresent the source
✗ Include redundant information

## Domain-Specific Guidelines

### Technical Documentation
- Preserve technical terms
- Include version numbers
- List prerequisites
- Highlight breaking changes
- Summarize API changes

### Meeting Notes
- List attendees
- Capture decisions made
- Extract action items with owners
- Note deadlines
- Record next steps

### Research Papers
- State the research question
- Summarize methodology
- Present key findings
- Include statistical significance
- Note limitations

### News Articles
- Answer: Who, What, When, Where, Why
- Lead with the main event
- Include key quotes
- Provide context
- Note sources

## Quality Checklist

Before finalizing a summary, verify:

- [ ] Main point is clear in first sentence/paragraph
- [ ] All critical information is included
- [ ] No important context is lost
- [ ] Length is appropriate (typically 10-30% of original)
- [ ] Language is clear and concise
- [ ] Structure aids comprehension
- [ ] Accuracy is maintained
- [ ] Target audience will understand it

## Using the Helper Scripts

This skill includes Python scripts in the `scripts/` folder:

### `text_analyzer.py`
Analyzes text to identify key sentences and important terms:
```python
from scripts.text_analyzer import analyze_text, extract_key_sentences

# Analyze text importance
analysis = analyze_text(long_text)
print(f"Key terms: {analysis['key_terms']}")
print(f"Sentence scores: {analysis['sentence_scores']}")

# Extract top sentences
key_sentences = extract_key_sentences(long_text, num_sentences=5)
```

### `summarizer.py`
Provides various summarization functions:
```python
from scripts.summarizer import (
    create_bullet_summary,
    create_executive_summary,
    extract_key_points
)

# Create bullet-point summary
bullets = create_bullet_summary(text, max_bullets=7)

# Generate executive summary
exec_summary = create_executive_summary(text, max_length=200)

# Extract key points
points = extract_key_points(text, num_points=5)
```

### `formatter.py`
Formats summaries in different styles:
```python
from scripts.formatter import (
    format_as_markdown,
    format_as_bullets,
    format_progressive_summary
)

# Format as markdown with sections
markdown = format_as_markdown(summary_dict)

# Create progressive summary (1 sentence, 1 para, full)
progressive = format_progressive_summary(text)
```

## Examples

### Example 1: Long Article Summary

**Original (500 words):** [Long article about climate change impacts...]

**Summary (50 words):**
```
Climate change is accelerating faster than predicted, with global temperatures
rising 1.2°C since pre-industrial times. Key impacts include: increased extreme
weather events (40% rise in hurricanes), sea level rise (8 inches since 1880),
and ecosystem disruption. Immediate action required to limit warming to 1.5°C.
```

### Example 2: Meeting Notes

**Original:** [30 minutes of discussion notes...]

**Summary:**
```
Product Launch Meeting - Jan 15, 2026

Decisions:
• Launch date set for March 1, 2026
• Budget approved: $250K
• Target: 10K users in first month

Action Items:
• Sarah: Finalize marketing plan (Due: Jan 22)
• Mike: Complete beta testing (Due: Feb 1)
• Team: Review pricing strategy (Due: Jan 29)

Risks:
• Competitor launching similar product in February
• Development 2 weeks behind schedule
```

### Example 3: Technical Documentation

**Original:** [10-page API documentation...]

**Summary:**
```
API v2.0 Release Summary

Breaking Changes:
• Authentication now requires OAuth 2.0 (API keys deprecated)
• Rate limit reduced from 1000 to 500 requests/hour
• Response format changed from XML to JSON only

New Features:
• Webhook support for real-time updates
• Batch operations endpoint (/api/v2/batch)
• GraphQL query support (beta)

Migration: Update by March 31, 2026. See migration guide at docs.example.com/v2-migration
```

## Tips for Different Content Types

### For Long Documents (>10 pages):
1. Read the abstract/introduction first
2. Scan section headers
3. Read conclusion
4. Extract key points from each section
5. Synthesize into coherent summary

### For Technical Content:
1. Preserve technical accuracy
2. Include code examples if critical
3. List prerequisites
4. Summarize step-by-step processes
5. Note version/compatibility info

### For Narrative Content:
1. Identify the main storyline
2. Extract key events
3. Note important characters/entities
4. Preserve cause-and-effect relationships
5. Capture the conclusion

## Common Pitfalls to Avoid

1. **Too Much Detail** - Remember: summary ≠ shortened version
2. **Missing Context** - Include enough background for understanding
3. **Bias Introduction** - Stay objective, don't add interpretation
4. **Poor Organization** - Structure matters as much as content
5. **Inconsistent Style** - Maintain consistent tone and format
6. **Accuracy Issues** - Always verify facts and figures

## When to Use This Skill

Use the summarizer skill when:
- User asks to "summarize", "condense", or "give me the key points"
- Dealing with long documents or articles
- Creating executive summaries or briefs
- Extracting action items from meetings
- Simplifying complex technical content
- Creating quick reference guides
- Preparing content for different audiences
- Time-constrained information needs

## Integration with Other Skills

This skill works well with:
- **PDF Skill**: Summarize PDF documents
- **Research Skill**: Condense research findings
- **Writing Skill**: Create concise content
- **Analysis Skill**: Present analytical findings clearly
