"""
Formatting Module

Provides functions to format summaries in different styles and structures.
"""

from typing import Any, Dict, List


def format_as_markdown(summary_dict: Dict[str, Any]) -> str:
    """
    Format a summary dictionary as markdown with sections.

    Args:
        summary_dict: Dictionary with summary components

    Returns:
        Formatted markdown string
    """
    markdown = []

    # Add title if present
    if "title" in summary_dict:
        markdown.append(f"# {summary_dict['title']}\n")

    # Add main point
    if "main_point" in summary_dict:
        markdown.append(f"## Summary\n\n{summary_dict['main_point']}\n")

    # Add key points
    if "key_points" in summary_dict and summary_dict["key_points"]:
        markdown.append("## Key Points\n")
        for point in summary_dict["key_points"]:
            markdown.append(f"- {point}")
        markdown.append("")

    # Add details
    if "details" in summary_dict and summary_dict["details"]:
        markdown.append("## Details\n")
        for detail in summary_dict["details"]:
            markdown.append(f"- {detail}")
        markdown.append("")

    # Add conclusion
    if "conclusion" in summary_dict:
        markdown.append(f"## Conclusion\n\n{summary_dict['conclusion']}\n")

    return "\n".join(markdown)


def format_as_bullets(items: List[str], prefix: str = "•") -> str:
    """
    Format a list of items as bullet points.

    Args:
        items: List of items to format
        prefix: Bullet character (default: •)

    Returns:
        Formatted bullet list string
    """
    bullets = [f"{prefix} {item}" for item in items]
    return "\n".join(bullets)


def format_as_numbered_list(items: List[str]) -> str:
    """
    Format a list of items as a numbered list.

    Args:
        items: List of items to format

    Returns:
        Formatted numbered list string
    """
    numbered = [f"{i}. {item}" for i, item in enumerate(items, 1)]
    return "\n".join(numbered)


def format_progressive_summary(summary_dict: Dict[str, str]) -> str:
    """
    Format a progressive summary with increasing detail levels.

    Args:
        summary_dict: Dictionary with 'one_sentence', 'one_paragraph', 'detailed'

    Returns:
        Formatted progressive summary
    """
    output = []

    if "one_sentence" in summary_dict:
        output.append("**TL;DR:**")
        output.append(summary_dict["one_sentence"])
        output.append("")

    if "one_paragraph" in summary_dict:
        output.append("**Quick Summary:**")
        output.append(summary_dict["one_paragraph"])
        output.append("")

    if "detailed" in summary_dict:
        output.append("**Detailed Summary:**")
        output.append(summary_dict["detailed"])

    return "\n".join(output)


def format_executive_summary(title: str, main_point: str, key_findings: List[str], recommendation: str = "") -> str:
    """
    Format an executive summary in standard business format.

    Args:
        title: Summary title
        main_point: Opening statement
        key_findings: List of key findings
        recommendation: Optional recommendation

    Returns:
        Formatted executive summary
    """
    output = [f"# {title}\n"]
    output.append(main_point)
    output.append("\n**Key Findings:**")

    for finding in key_findings:
        output.append(f"• {finding}")

    if recommendation:
        output.append(f"\n**Recommendation:** {recommendation}")

    return "\n".join(output)


def format_meeting_summary(
    title: str, date: str, decisions: List[str], action_items: List[str], next_steps: List[str] = None
) -> str:
    """
    Format a meeting summary in standard format.

    Args:
        title: Meeting title
        date: Meeting date
        decisions: List of decisions made
        action_items: List of action items
        next_steps: Optional list of next steps

    Returns:
        Formatted meeting summary
    """
    output = [f"# {title}", f"**Date:** {date}\n"]

    if decisions:
        output.append("## Decisions")
        for decision in decisions:
            output.append(f"• {decision}")
        output.append("")

    if action_items:
        output.append("## Action Items")
        for item in action_items:
            output.append(f"• {item}")
        output.append("")

    if next_steps:
        output.append("## Next Steps")
        for step in next_steps:
            output.append(f"• {step}")

    return "\n".join(output)


def format_technical_summary(
    title: str, overview: str, key_changes: List[str], breaking_changes: List[str] = None, migration_notes: str = ""
) -> str:
    """
    Format a technical documentation summary.

    Args:
        title: Document title
        overview: Brief overview
        key_changes: List of key changes
        breaking_changes: Optional list of breaking changes
        migration_notes: Optional migration information

    Returns:
        Formatted technical summary
    """
    output = [f"# {title}\n", f"## Overview\n{overview}\n"]

    if breaking_changes:
        output.append("## ⚠️ Breaking Changes")
        for change in breaking_changes:
            output.append(f"• {change}")
        output.append("")

    output.append("## Key Changes")
    for change in key_changes:
        output.append(f"• {change}")

    if migration_notes:
        output.append(f"\n## Migration\n{migration_notes}")

    return "\n".join(output)


def format_with_sections(sections: Dict[str, List[str]]) -> str:
    """
    Format content with custom sections.

    Args:
        sections: Dictionary of section_name -> list of items

    Returns:
        Formatted output with sections
    """
    output = []

    for section_name, items in sections.items():
        output.append(f"## {section_name}")
        for item in items:
            output.append(f"• {item}")
        output.append("")

    return "\n".join(output)


def format_comparison_summary(title: str, similarities: List[str], differences: List[str], conclusion: str = "") -> str:
    """
    Format a comparison summary.

    Args:
        title: Comparison title
        similarities: List of similarities
        differences: List of differences
        conclusion: Optional conclusion

    Returns:
        Formatted comparison summary
    """
    output = [f"# {title}\n"]

    if similarities:
        output.append("## Similarities")
        for item in similarities:
            output.append(f"• {item}")
        output.append("")

    if differences:
        output.append("## Differences")
        for item in differences:
            output.append(f"• {item}")
        output.append("")

    if conclusion:
        output.append(f"## Conclusion\n{conclusion}")

    return "\n".join(output)


def wrap_text(text: str, width: int = 80) -> str:
    """
    Wrap text to specified width.

    Args:
        text: Text to wrap
        width: Maximum line width

    Returns:
        Wrapped text
    """
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1  # +1 for space
        if current_length + word_length > width:
            if current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
        else:
            current_line.append(word)
            current_length += word_length

    if current_line:
        lines.append(" ".join(current_line))

    return "\n".join(lines)


def add_metadata(summary: str, metadata: Dict[str, str]) -> str:
    """
    Add metadata header to summary.

    Args:
        summary: Summary text
        metadata: Dictionary of metadata key-value pairs

    Returns:
        Summary with metadata header
    """
    header = ["---"]
    for key, value in metadata.items():
        header.append(f"{key}: {value}")
    header.append("---\n")

    return "\n".join(header) + "\n" + summary


if __name__ == "__main__":
    # Example usage
    print("Formatting Examples")
    print("=" * 60)

    # Example 1: Markdown formatting
    summary_dict = {
        "title": "Q4 Sales Report",
        "main_point": "Sales increased 23% year-over-year.",
        "key_points": ["Revenue reached $4.2M", "Customer acquisition cost decreased 15%", "Churn rate stable at 3.2%"],
        "conclusion": "Strong performance indicates market momentum.",
    }

    print("\n1. Markdown Format:")
    print(format_as_markdown(summary_dict))

    # Example 2: Executive summary
    print("\n2. Executive Summary:")
    exec_summary = format_executive_summary(
        title="Product Launch Analysis",
        main_point="The product launch exceeded expectations with 15K users in first month.",
        key_findings=["User acquisition 50% above target", "Customer satisfaction: 4.7/5", "Revenue: $180K in month 1"],
        recommendation="Increase marketing budget by 30% to capitalize on momentum.",
    )
    print(exec_summary)

    # Example 3: Meeting summary
    print("\n3. Meeting Summary:")
    meeting = format_meeting_summary(
        title="Sprint Planning Meeting",
        date="January 15, 2026",
        decisions=["Approved feature roadmap for Q1", "Allocated $50K for infrastructure"],
        action_items=["Sarah: Create design mockups (Due: Jan 22)", "Mike: Set up CI/CD pipeline (Due: Jan 20)"],
    )
    print(meeting)

# Made with Bob
