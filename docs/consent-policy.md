# Consent & Responsible-Use Policy

This file defines the consent requirements this toolchain enforces *in code*,
the categories of use it refuses, and the rationale. The authoritative
technical specification is `DEEPFAKE_BUILD_PLAN.md` §2; this is the human
readable policy.

## The rule

This software only produces media of people who have explicitly consented.
That is not a guideline — it is an architectural gate:

- No swap, source-embedding, live session, or benchmark can start without a
  valid consent record for the source identity (`core/consent.py`,
  `SwapPipeline.__init__`). The CLI checks the ledger *before* loading any
  model, and the pipeline re-checks at construction.
- A record (`consent/<identity_id>.json`) must exist, cover the project and
  use requested, be unexpired, and be unrevoked.
- **Revocation is destructive by design**: it marks the record and deletes
  that identity's trained models (`assets/models/trained/<identity_id>/`).
- Every output is triple-marked: visible "SYNTHETIC MEDIA" badge, invisible
  DCT watermark (keyable HMAC payload), and C2PA content credentials with the
  consent id. There is no export path that strips provenance.

## Consent and intent

Both matter. Consent without disclosed intent, or disclosure without consent,
is not sufficient. Legitimate use looks like: your own likeness; a licensed
actor with a signed release; consented creative, educational, accessibility,
or VFX work where the audience is told the media is synthetic.

## What this tool will not do

Hard exclusions (these are code-level constraints, not tone):

- Non-consensual intimate imagery. Criminal in most jurisdictions.
- Impersonation for fraud, authentication bypass, or deception.
- Evasion of deepfake detectors, watermark removers, or provenance stripping.
- Political disinformation or election-period manipulation.
- Any use of a person's likeness without a consent record.

Requests to weaken the consent gate, strip provenance, or evade detection
will be declined. The defensive self-check (`§2.4`) exists to *validate*
watermark durability — outputs must remain detectable.

## Known limits (be honest about them)

- The consent ledger proves a record exists, not that the signer is who they
  say. Stronger identity binding (ed25519-signed records, third-party
  attestation) is roadmap work.
- Watermarks survive re-encode and light blur on photographic content; they
  are a detection aid, not misuse prevention. A determined redistributor can
  degrade them. The consent gate, not the watermark, is the primary control.
- The open-source code can be forked and the gates removed. That is a known
  cost of open sourcing; this project's stance is that the norms, defaults,
  and license note matter.

## Compliance notes

Starting points, not legal advice — get counsel before any commercial
release: UK Online Safety Act / Sexual Offences Act; US CA AB 602, AB 730,
TX SB 751; EU AI Act synthetic-content transparency obligations (in force
2026); India DPDP Act. See `DEEPFAKE_BUILD_PLAN.md` §2.3 for the table.

## Reporting misuse

If you encounter output of this tool used against this policy, the embedded
C2PA manifest and watermark carry the consent id and generator id of the
producing installation — use them, and contact the identity owner's
revocation contact (stored in every consent record).
