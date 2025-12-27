"""
Slug generation utility for lens profile identifiers.

Provides deterministic, boring slug generation with no clever Unicode handling.
ASCII-only, predictable output.
"""

import re


def slugify(text: str) -> str:
    """
    Convert text to a lowercase slug suitable for identifiers.
    
    Rules:
    - Convert to lowercase
    - Replace spaces with hyphens
    - Keep only alphanumeric and hyphens
    - Strip leading/trailing hyphens
    - Collapse multiple consecutive hyphens
    
    Examples:
        "HERO10 Black" -> "hero10-black"
        "iPhone 15 Pro" -> "iphone-15-pro"
        "A7S III" -> "a7s-iii"
        "GoPro HERO11" -> "gopro-hero11"
    
    Args:
        text: Input string to slugify
        
    Returns:
        Slugified string (lowercase, alphanumeric + hyphens only)
    """
    # Convert to lowercase
    slug = text.lower()
    
    # Replace spaces with hyphens
    slug = slug.replace(" ", "-")
    
    # Keep only alphanumeric and hyphens (ASCII only)
    slug = re.sub(r"[^a-z0-9\-]", "", slug)
    
    # Collapse multiple consecutive hyphens
    slug = re.sub(r"-+", "-", slug)
    
    # Strip leading/trailing hyphens
    slug = slug.strip("-")
    
    return slug
