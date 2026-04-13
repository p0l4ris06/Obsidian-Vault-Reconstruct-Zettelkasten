"""
Expand short notes in the Obsidian vault with relevant content.

This script identifies notes with minimal content (< 100 chars after frontmatter)
and expands them with relevant content based on their title and tags.
"""

import re
import json
from pathlib import Path
from datetime import datetime

VAULT_PATH = Path(r"C:\Users\dcrac\Documents\Obsidian Vault")
ZETTELS_PATH = VAULT_PATH / "02_Zettels"

# Content templates for common veterinary topics
CONTENT_TEMPLATES = {
    # Anatomy templates
    "anatomy": """## Structure

- Location:
- Components:
- Relations:

## Function

- Primary role:
- Clinical relevance:

## Key Points

1.
2.
3.

## Related Notes
""",

    "physiology": """## Overview

This note covers the physiological aspects of {title}.

## Key Concepts

1. **Mechanism**:
2. **Regulation**:
3. **Clinical Significance**:

## Related Processes

-

## Clinical Relevance

-

## Related Notes
""",

    "disease": """## Definition

**{title}** is a condition affecting veterinary patients.

## Aetiology

- **Cause**:
- **Risk Factors**:
- **Predisposing Breeds**:

## Clinical Signs

-

## Diagnosis

- **History**:
- **Physical Examination**:
- **Diagnostic Tests**:

## Treatment

-

## Prognosis

-

## Related Notes
""",

    "bacteria": """## Organism

**{title}**

- **Classification**:
- **Morphology**:
- **Gram Stain**:

## Pathogenicity

- **Virulence Factors**:
- **Diseases Caused**:

## Clinical Significance

- **Host Species**:
- **Transmission**:

## Diagnosis & Treatment

-

## Related Notes
""",

    "virus": """## Organism

**{title}**

- **Classification**:
- **Structure**:
- **Genome**:

## Pathogenicity

- **Replication**:
- **Diseases Caused**:

## Clinical Significance

- **Host Species**:
- **Transmission**:

## Diagnosis & Treatment

-

## Related Notes
""",

    "drug": """## Drug Profile

**{title}**

- **Class**:
- **Mechanism of Action**:

## Indications

-

## Contraindications & Precautions

-

## Dosing

- **Dogs**:
- **Cats**:

## Side Effects

-

## Related Notes
""",

    "procedure": """## Overview

**{title}** is a clinical procedure in veterinary practice.

## Indications

-

## Contraindications

-

## Technique

1.
2.
3.

## Complications

-

## Aftercare

-

## Related Notes
""",

    "default": """## Overview

{title}

## Key Points

1.
2.
3.

## Clinical Relevance

-

## Related Notes

"""
}

# Keywords to match templates
KEYWORD_TEMPLATES = {
    # Anatomy keywords
    "anatomy": ["anatomy", "structure", "muscle", "bone", "nerve", "artery", "vein", "organ", "tissue", "gland"],
    # Physiology keywords
    "physiology": ["physiology", "function", "process", "mechanism", "regulation", "system", "pathway"],
    # Disease keywords
    "disease": ["disease", "disorder", "syndrome", "condition", "infection", "inflammation", "-itis", "-osis", "-pathy"],
    # Bacteria keywords
    "bacteria": ["bacteria", "bacterium", "bacterial", "cocci", "bacilli", "streptococc", "staphylococc", "e. coli", "salmonella", "campylobacter", "clostridium", "pasteurella", "borrelia", "leptospira"],
    # Virus keywords
    "virus": ["virus", "viral", "parvovirus", "coronavirus", "herpesvirus", "calicivirus", "distemper", "rabies", "influenza", "poxvirus", "adenovirus", "papilloma"],
    # Drug keywords
    "drug": ["drug", "medication", "antibiotic", "analgesic", "anti-inflammatory", "anaesthetic", "sedative", "fluid", "therapy", "treatment"],
    # Procedure keywords
    "procedure": ["procedure", "technique", "surgery", "surgical", "examination", "diagnostic", "catheter", "intubation", "injection", "venipuncture"],
}


def get_frontmatter_and_content(file_path: Path):
    """Extract frontmatter and content from a markdown file."""
    content = file_path.read_text(encoding="utf-8")

    # Extract frontmatter
    fm_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)

    if fm_match:
        frontmatter = fm_match.group(1)
        body = content[fm_match.end():]
        return frontmatter, body

    return "", content


def parse_frontmatter(frontmatter: str) -> dict:
    """Parse YAML frontmatter into a dict."""
    data = {}
    for line in frontmatter.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key == 'tags':
                data[key] = []
            elif key in data and isinstance(data[key], list):
                data[key].append(value.strip('- '))
            else:
                data[key] = value
        elif line.strip().startswith('- '):
            # List item
            value = line.strip().lstrip('- ').strip()
            if 'tags' in data:
                data['tags'].append(value)
    return data


def get_template_for_note(title: str, tags: list) -> str:
    """Get the most appropriate template for a note."""
    title_lower = title.lower()
    tags_lower = [t.lower() for t in tags]

    # Check each keyword category
    for template_name, keywords in KEYWORD_TEMPLATES.items():
        for keyword in keywords:
            if keyword in title_lower:
                return CONTENT_TEMPLATES[template_name]
            for tag in tags_lower:
                if keyword in tag:
                    return CONTENT_TEMPLATES[template_name]

    return CONTENT_TEMPLATES["default"]


def expand_short_note(file_path: Path, min_content_length: int = 100) -> bool:
    """Expand a short note with relevant content template."""
    try:
        frontmatter, body = get_frontmatter_and_content(file_path)

        # Skip if already has enough content
        if len(body.strip()) >= min_content_length:
            return False

        # Skip if body is just a placeholder
        stripped_body = body.strip()
        if stripped_body in ["Note body here.", "Note body here", "", "# Note body here"]:
            stripped_body = ""

        # Parse frontmatter
        fm_data = parse_frontmatter(frontmatter)
        title = fm_data.get('title', file_path.stem)
        tags = fm_data.get('tags', [])

        # Get appropriate template
        template = get_template_for_note(title, tags)

        # Fill in template
        new_content = template.format(title=title)

        # Preserve any existing non-placeholder content
        if stripped_body and stripped_body != f"# {title}":
            # Keep existing content but add template
            new_content = f"{stripped_body}\n\n{new_content}"

        # Reconstruct file
        full_content = f"---\n{frontmatter}\n---\n\n{new_content}"

        # Write back
        file_path.write_text(full_content, encoding="utf-8")
        return True

    except Exception as e:
        print(f"  Error processing {file_path.name}: {e}")
        return False


def main():
    print("=" * 60)
    print("Expanding Short Notes")
    print("=" * 60)

    # Find all markdown files in zettels
    md_files = list(ZETTELS_PATH.glob("*.md"))
    print(f"Found {len(md_files)} notes in {ZETTELS_PATH.name}")

    expanded_count = 0
    skipped_count = 0

    for md_file in md_files:
        try:
            frontmatter, body = get_frontmatter_and_content(md_file)

            # Check if short
            if len(body.strip()) < 100:
                title = md_file.stem
                if expand_short_note(md_file):
                    expanded_count += 1
                    print(f"  Expanded: {title[:50]}")
                else:
                    skipped_count += 1

        except Exception as e:
            print(f"  Error: {md_file.name}: {e}")

    print("\n" + "=" * 60)
    print(f"Summary: Expanded {expanded_count} notes, skipped {skipped_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()