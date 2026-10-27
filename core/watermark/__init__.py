"""Disclosure watermarking: invisible DCT watermark, C2PA provenance, visible badge."""
from .watermark import VerifyResult, embed, make_payload, verify
from .badge import add_badge
from .c2pa import sign_output, has_c2pa

__all__ = [
    "VerifyResult",
    "embed",
    "make_payload",
    "verify",
    "add_badge",
    "sign_output",
    "has_c2pa",
]
