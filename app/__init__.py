"""
AI DJ — an automatic, professional-quality continuous DJ mix generator.

This package contains the full analysis, planning, transition, effects and
rendering pipeline plus a FastAPI backend that drives the liquid-glass UI.

Design goals (see README.md):
  * Preserve audio quality (float32 internal, high-quality resampling / stretch).
  * Musically justified, beat-aligned transitions.
  * Conservative, safety-checked effects and final loudness management.
  * Graceful degradation when optional models/deps are missing.
"""

__version__ = "1.0.0"
