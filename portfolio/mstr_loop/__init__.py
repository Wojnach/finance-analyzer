"""MSTR Loop — multi-strategy trading system for MicroStrategy (MSTR).

Mirrors the metals_loop architecture: one 60s cycle that fetches the MSTR
signal bundle and hands it to each enabled strategy. Strategies decide
independently whether to trade; execution is gated by the PHASE flag
(shadow / paper / live).
"""
