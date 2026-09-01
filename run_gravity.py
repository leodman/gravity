"""Recommended launcher for Gravity.

Keeps the full permanent raster trail internally, but downsamples only the
browser preview so Gradio/Plotly does not transmit a million-cell heatmap on
every refresh. Physics and diagnostics are unchanged.
"""

import numpy as np
import plotly.graph_objects as go

import planetary_sim as sim


def efficient_image_trace(self, opacity, max_preview=250):
    """Return a decimated Plotly preview while preserving the full raster buffer."""
    if not np.any(self.buffer):
        return None

    # The full-resolution uint16 buffer remains untouched.  Only the browser
    # representation is decimated to avoid starving the other live plots.
    stride = max(1, int(np.ceil(self.resolution / float(max_preview))))
    view = self.buffer[::stride, ::stride]

    z = np.log1p(view.astype(np.float32))
    zmax = float(np.max(z))
    if zmax > 0.0:
        z /= zmax

    return go.Heatmap(
        z=z,
        x=np.linspace(-self.limit, self.limit, view.shape[1]),
        y=np.linspace(-self.limit, self.limit, view.shape[0]),
        zmin=0,
        zmax=1,
        showscale=False,
        hoverinfo="skip",
        opacity=max(0.0, min(1.0, float(opacity))),
        colorscale=[[0.0, "rgba(0,0,0,0)"], [1.0, "rgba(120,120,120,1)"]],
    )


# Patch visualization only. The N-body physics, collisions, energy calculations,
# momentum calculations, history, and UI wiring continue to come from planetary_sim.
sim.RasterTrail.image_trace = efficient_image_trace


if __name__ == "__main__":
    sim.demo.queue().launch(share=True)
