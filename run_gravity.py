"""Recommended launcher for Gravity.

Runs the current application, including:
- Huge, Box, and 2-D Toroid boundary modes
- fixed-memory permanent raster trails with lightweight browser previews
- live Energy, Energy error, Momentum, and diagnostics plots

After pulling repository updates, stop any old Gradio/Python server before
starting this launcher again. A running Gradio process does not hot-reload the
updated source files.
"""

from gravity_app import demo


if __name__ == "__main__":
    print("\n=== Gravity — current application ===")
    print("Boundary modes available: Huge | Box | Toroid")
    print("Toroid: 2-D periodic XY, minimum-image gravity and collisions")
    print("Launcher: run_gravity.py -> gravity_app.py")
    print("If you just pulled an update, open the NEW Gradio URL printed below.")
    print("Do not reuse an older Gradio tab/share link.\n")
    demo.queue().launch(share=True)
