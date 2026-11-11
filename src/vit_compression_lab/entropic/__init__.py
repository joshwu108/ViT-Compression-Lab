"""Entropic Student: learned, entropy-coded compression of Qwen visual features.

Adaptation of Matsubara et al., "Supervised Compression for Resource-Constrained
Edge Computing Systems" (WACV 2022) to the Qwen3.5-family vision tower used by
the Qwen3.8-27B checkpoint. See
encoders/Qwen3.8-vision-compression/docs/entropic-student-plan.md.

Model-agnostic pieces live in this package; Qwen-specific loading, teacher
splitting, data handling and injection helpers live in ``entropic.qwen``.
Nothing here downloads weights or data.
"""
