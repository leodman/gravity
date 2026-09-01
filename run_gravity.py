"""Recommended launcher for Gravity.

Runs the current application, including:
- Huge, Box, and 2-D Toroid boundary modes
- fixed-memory permanent raster trails with lightweight browser previews
- live Energy, Energy error, Momentum, and diagnostics plots
"""

from gravity_app import demo


if __name__ == "__main__":
    demo.queue().launch(share=True)
