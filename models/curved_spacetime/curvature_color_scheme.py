"""
Curvature Color Scheme: Gradient fill colors based on market curvature values

This module provides a gradient color scheme that maps curvature values to colors:
- Negative curvature (volatile): Red → Orange → Yellow → Gray
- Zero curvature (neutral): Gray  
- Positive curvature (trending): Gray → Purple → Magenta → Cyan

The gradient creates smooth transitions that intuitively represent market states.
"""

import numpy as np
from typing import Tuple


def get_curvature_gradient_color(curvature: float) -> Tuple[str, str]:
    """
    Create gradient color scheme from volatile (red) through neutral (gray) to trending (cyan).
    
    Args:
        curvature: Curvature value, typically in range [-1, 1]
    
    Returns:
        Tuple of (rgba_color_string, description)
    
    Color Mapping:
        -1.0: Deep Red (Highly Volatile)
        -0.5: Orange (Volatile) 
         0.0: Gray (Neutral/Efficient Market)
        +0.5: Purple (Trending)
        +1.0: Cyan (Highly Trending)
    """
    # Normalize curvature to [0, 1] for color mapping
    # -1.0 → 0.0 (deep red/volatile)
    #  0.0 → 0.5 (gray/neutral) 
    # +1.0 → 1.0 (cyan/extreme trending)
    normalized = (curvature + 1.0) / 2.0
    normalized = np.clip(normalized, 0.0, 1.0)
    
    if normalized < 0.5:
        # Volatile side: Red → Orange → Yellow → Gray
        # 0.0 to 0.5 maps to volatile spectrum
        t = normalized * 2.0  # 0 to 1
        
        if t < 0.33:
            # Deep red to orange
            mix = t / 0.33
            r = 255
            g = int(69 * mix)  # 0 to 69
            b = 0
            description = "HIGHLY VOLATILE"
        elif t < 0.67:
            # Orange to yellow
            mix = (t - 0.33) / 0.34
            r = 255
            g = int(69 + (255-69) * mix)  # 69 to 255
            b = 0
            description = "VOLATILE"
        else:
            # Yellow to gray
            mix = (t - 0.67) / 0.33
            r = int(255 - 127 * mix)  # 255 to 128
            g = int(255 - 127 * mix)  # 255 to 128
            b = int(128 * mix)        # 0 to 128
            description = "NEUTRAL-VOLATILE"
            
    else:
        # Trending side: Gray → Purple → Magenta → Cyan
        # 0.5 to 1.0 maps to trending spectrum
        t = (normalized - 0.5) * 2.0  # 0 to 1
        
        if t < 0.33:
            # Gray to purple
            mix = t / 0.33
            r = int(128 + 127 * mix)  # 128 to 255
            g = int(128 * (1 - mix))   # 128 to 0
            b = int(128 + 127 * mix)  # 128 to 255
            description = "NEUTRAL-TRENDING"
        elif t < 0.67:
            # Purple to magenta
            mix = (t - 0.33) / 0.34
            r = 255
            g = 0
            b = int(255 * (1 - 0.5 * mix))  # 255 to 192
            description = "TRENDING"
        else:
            # Magenta to cyan
            mix = (t - 0.67) / 0.33
            r = int(255 * (1 - mix))    # 255 to 0
            g = int(255 * mix)          # 0 to 255
            b = 255
            description = "HIGHLY TRENDING"
    
    rgba_color = f'rgba({r}, {g}, {b}, 0.8)'
    return rgba_color, description


def create_curvature_colorbar_data(n_points: int = 100) -> Tuple[list, list, list]:
    """
    Create data for displaying a curvature colorbar/legend.
    
    Args:
        n_points: Number of points in the colorbar
    
    Returns:
        Tuple of (curvature_values, colors, descriptions)
    """
    curvature_values = np.linspace(-1.0, 1.0, n_points)
    colors = []
    descriptions = []
    
    for curvature in curvature_values:
        color, description = get_curvature_gradient_color(curvature)
        colors.append(color)
        descriptions.append(description)
    
    return curvature_values.tolist(), colors, descriptions


def get_discrete_curvature_levels() -> dict:
    """
    Get discrete curvature level examples for legend/documentation.
    
    Returns:
        Dictionary mapping curvature values to (color, description) tuples
    """
    levels = {
        -1.0: "Highly Volatile Markets",
        -0.5: "Volatile Markets", 
        -0.25: "Slightly Volatile",
        0.0: "Neutral/Efficient Market",
        0.25: "Slightly Trending",
        0.5: "Trending Markets",
        1.0: "Highly Trending Markets"
    }
    
    result = {}
    for curvature, label in levels.items():
        color, description = get_curvature_gradient_color(curvature)
        result[curvature] = (color, f"{description}: {label}")
    
    return result


if __name__ == "__main__":
    # Test the color scheme
    print("🎨 CURVATURE COLOR SCHEME TEST")
    print("=" * 40)
    
    test_curvatures = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    
    for curvature in test_curvatures:
        color, description = get_curvature_gradient_color(curvature)
        print(f"Curvature {curvature:+.2f}: {color} ({description})")
    
    print("\n📊 Discrete levels:")
    levels = get_discrete_curvature_levels()
    for curvature, (color, description) in levels.items():
        print(f"  {curvature:+.2f}: {description}")