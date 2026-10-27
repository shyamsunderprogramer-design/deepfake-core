"""Frequency-domain disclosure watermark (DCT-II on 8x8 blocks).

Payload: 256-bit sha256(generator_id | consent_id | model_version | timestamp)
plus a CRC16, tiled across all 8x8 luma blocks at 3 bits per block via
differential encoding of mid-band coefficient pairs. Verification folds all
blocks by per-bit majority vote and checks the CRC. Pure numpy, so the unit
tests (`tests/test_watermark.py`) run without cv2. Durability checking against
detectors (plan §11.5) uses `verify()` on re-encoded output as a *validation*
step only — never to tune the mark around a detector.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

_BLOCK = 8
# mid-band coefficient pairs (row, col) inside an 8x8 DCT block
_PAIRS = (((4, 3), (3, 4)), ((5, 2), (2, 5)), ((4, 5), (5, 4)))
_BITS_PER_BLOCK = len(_PAIRS)
_STRENGTH = 12.0  # minimum coefficient separation per embedded bit
_PAYLOAD_BYTES = 34  # 32-byte sha256 payload + 2-byte CRC16
_PAYLOAD_BITS = _PAYLOAD_BYTES * 8  # 272


def _dct_matrix(n: int = _BLOCK) -> np.ndarray:
    """Orthonormal DCT-II basis: dct2(b) = C @ b @ C.T."""
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    alpha = np.sqrt(2.0 / n) * np.ones(n)
    alpha[0] = np.sqrt(1.0 / n)
    return alpha[:, None] * np.cos(np.pi * k * (2 * i + 1) / (2 * n))


_C = _dct_matrix()


def _crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def make_payload(
    generator_id: str,
    consent_id: str,
    model_version: str,
    timestamp_iso: str,
    key: Optional[bytes] = None,
) -> bytes:
    """32-byte payload tying the output to generator/consent/version/time.

    Without `key` this is a plain sha256 — adequate for *detection* ("is this
    output marked") but forgeable as *attribution* (anyone can recompute the
    same digest). Pass an installation secret (pipeline wires
    DEEPFAKE_WM_KEY) and the payload becomes an HMAC-SHA256, so third parties
    cannot mint marks attributable to this installation. Attribution
    authenticity ultimately rests on the embedded, signed C2PA path — this
    mark is the durable fallback, and verify() only ever proves presence.
    """
    material = "|".join([generator_id, consent_id, model_version, timestamp_iso])
    raw = material.encode("utf-8")
    if key:
        import hmac

        return hmac.new(key, raw, hashlib.sha256).digest()
    return hashlib.sha256(raw).digest()


def _payload_bits(payload32: bytes) -> np.ndarray:
    if len(payload32) != 32:
        raise ValueError("payload must be 32 bytes (sha256 digest)")
    crc = _crc16_ccitt(payload32).to_bytes(2, "big")
    return np.unpackbits(np.frombuffer(payload32 + crc, dtype=np.uint8))


def _split_luma(image: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """(luma float64, rgb float64 | None). RGB inputs are marked with 8-bit range."""
    img = np.asarray(image)
    if img.ndim == 2:
        return img.astype(np.float64), None
    rgb = img[..., :3].astype(np.float64)
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return y, rgb


def _blocks(luma: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """(blocks [nb,8,8], blocks_h, blocks_w) from the cropped top-left region."""
    bh, bw = luma.shape[0] // _BLOCK, luma.shape[1] // _BLOCK
    if bh == 0 or bw == 0:
        raise ValueError(f"image too small for watermarking: {luma.shape}")
    cropped = luma[: bh * _BLOCK, : bw * _BLOCK]
    blocks = (
        cropped.reshape(bh, _BLOCK, bw, _BLOCK)
        .transpose(0, 2, 1, 3)
        .reshape(bh * bw, _BLOCK, _BLOCK)
    )
    return blocks, bh, bw


def embed(
    image: np.ndarray, payload32: bytes, strength: float = _STRENGTH
) -> np.ndarray:
    """Embed the payload in the luma channel; returns uint8 (RGB stays 3-channel)."""
    bits = _payload_bits(payload32)
    luma, rgb = _split_luma(image)
    blocks, bh, bw = _blocks(luma)
    nb = blocks.shape[0]

    tiled = np.resize(bits, nb * _BITS_PER_BLOCK)
    dc = np.einsum("ij,bjk,lk->bil", _C, blocks, _C)  # C @ b @ C.T
    for j, ((r1, c1), (r2, c2)) in enumerate(_PAIRS):
        target = np.where(tiled[j::_BITS_PER_BLOCK] == 1, strength, -strength)
        diff = dc[:, r1, c1] - dc[:, r2, c2]
        up = np.where(target > 0, np.clip(target - diff, 0.0, None), 0.0)
        dn = np.where(target < 0, np.clip(diff - target, 0.0, None), 0.0)
        dc[:, r1, c1] += (up - dn) / 2.0
        dc[:, r2, c2] -= (up - dn) / 2.0
    rebuilt = np.einsum("ji,bjk,kl->bil", _C, dc, _C)  # C.T @ dc @ C

    out_luma = luma.copy()
    out_luma[: bh * _BLOCK, : bw * _BLOCK] = (
        rebuilt.reshape(bh, bw, _BLOCK, _BLOCK)
        .transpose(0, 2, 1, 3)
        .reshape(bh * _BLOCK, bw * _BLOCK)
    )
    if rgb is None:
        return np.clip(out_luma, 0, 255).astype(np.uint8)
    delta = (out_luma - luma)[..., None]
    return np.clip(rgb + delta, 0, 255).astype(np.uint8)


@dataclass
class VerifyResult:
    payload_hex: str  # decoded payload even when crc_ok is False (for logging)
    crc_ok: bool
    confidence: float  # fraction of collected block-bits agreeing with the vote
    blocks: int


def verify(image: np.ndarray) -> VerifyResult:
    """Decode the watermark by per-bit majority vote across all blocks."""
    luma, _ = _split_luma(image)
    blocks, _, _ = _blocks(luma)
    nb = blocks.shape[0]
    dc = np.einsum("ij,bjk,lk->bil", _C, blocks, _C)
    flat = np.concatenate(
        [dc[:, r1, c1] > dc[:, r2, c2] for (r1, c1), (r2, c2) in _PAIRS],
        axis=0,
    )  # shape (3*nb,) but pairs were gathered per-pair; re-interleave to match embed tiling
    flat = flat.reshape(_BITS_PER_BLOCK, nb).T.reshape(-1)

    copies = flat.size // _PAYLOAD_BITS
    if copies < 1:
        return VerifyResult(payload_hex="", crc_ok=False, confidence=0.0, blocks=nb)
    mat = flat[: copies * _PAYLOAD_BITS].reshape(copies, _PAYLOAD_BITS)
    voted = mat.mean(axis=0) > 0.5
    confidence = float((mat == voted[None, :]).mean())

    packed = np.packbits(voted.astype(np.uint8)).tobytes()
    payload, crc_recv = packed[:32], int.from_bytes(packed[32:34], "big")
    crc_ok = _crc16_ccitt(payload) == crc_recv
    return VerifyResult(
        payload_hex=payload.hex(), crc_ok=crc_ok, confidence=confidence, blocks=nb
    )
