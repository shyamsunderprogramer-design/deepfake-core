"""Face-swap pipeline stages (plan §4).

Composable stage objects — no module-level mutable state — so the consent
gate wraps the pipeline entry point (core.pipeline.pipeline) and stages stay
testable in isolation with stubbed ONNX sessions.
"""
