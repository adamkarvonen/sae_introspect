"""
Shared color mapping for consistent colors across all final paper plots.
Each label maps to the same color across all plots.
"""

import matplotlib.pyplot as plt

# Define a consistent color palette for all labels
# Colors are chosen to be visually distinct and consistent across plots
LABEL_COLOR_MAP = {
    # Core methods - using tab10 colors for consistency
    "LatentQA": "#1f77b4",  # Blue (tab10[0])
    "Classification": "#ff7f0e",  # Orange (tab10[1])
    "LatentQA + Classification": "#2ca02c",  # Green (tab10[2])
    "LatentQA + Classification (400k samples)": "#2ca02c",  # Green (same color)
    "Classification + LatentQA": "#2ca02c",  # Green (normalized variant - same color)
    # Combined methods - highlight color for "Context Prediction" methods
    "Context Prediction + LatentQA + Classification": "#FDB813",  # Gold/Yellow (highlight color)
    "Context Prediction + Classification + LatentQA": "#FDB813",  # Gold/Yellow (normalized variant)
    "Full Dataset": "#FDB813",  # Gold/Yellow (highlight color)
    "Full Dataset (1M samples)": "#FDB813",  # Gold/Yellow (highlight color)
    "Full Dataset (400k samples)": "#FDB813",  # Gold/Yellow (will be overridden to red in some plots)
    # SAE methods
    "SAE + Classification + LatentQA": "#9467bd",  # Purple (tab10[4])
    "SAE + LatentQA + Classification": "#9467bd",  # Purple (normalized variant)
    # Other methods
    "Classification Single Token Training": "#8c564b",  # Brown (tab10[5])
    "Original Model": "#7f7f7f",  # Gray (tab10[7]) - neutral color for baseline
}

# Get tab20 colors for fallback (for any labels not in the map)
TAB20_COLORS = plt.get_cmap("tab20").colors


def get_color_for_label(label: str) -> tuple:
    """
    Get a consistent color for a given label.

    Args:
        label: The human-readable label (from CUSTOM_LABELS)

    Returns:
        Color tuple (RGBA) for the label
    """
    if label in LABEL_COLOR_MAP:
        # Convert hex to RGB tuple
        hex_color = LABEL_COLOR_MAP[label]
        rgb = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
        return (*rgb, 1.0)  # Add alpha channel

    # Fallback: use hash-based color assignment for unknown labels
    # This ensures the same label always gets the same color
    hash_val = hash(label)
    color_idx = abs(hash_val) % len(TAB20_COLORS)
    return TAB20_COLORS[color_idx]


def get_colors_for_labels(
    labels: list[str], highlight_color: str | None = None, highlight_index: int | None = None
) -> list[tuple]:
    """
    Get colors for a list of labels, ensuring consistency.

    Args:
        labels: List of human-readable labels
        highlight_color: Optional hex color for highlighted bar (e.g., "#FDB813")
        highlight_index: Optional index of bar to highlight (will override label color)

    Returns:
        List of color tuples (RGBA) for each label
    """
    colors = [get_color_for_label(label) for label in labels]

    # Override highlight color if specified
    if highlight_color is not None and highlight_index is not None:
        # Convert hex to RGB tuple
        rgb = tuple(int(highlight_color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
        colors[highlight_index] = (*rgb, 1.0)

    return colors


def get_shared_palette(labels: list[str]) -> dict[str, tuple]:
    """
    Get a palette dictionary mapping labels to colors.
    Useful for functions that expect a palette dict.

    Args:
        labels: List of labels to include in palette

    Returns:
        Dictionary mapping label -> color tuple
    """
    return {label: get_color_for_label(label) for label in labels}
