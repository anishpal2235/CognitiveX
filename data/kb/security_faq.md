# Security & Vendor Review FAQ (Approved — Governed Source)

Owner: Information Security. Review cycle: semi-annual. Classification: internal.

## Vendor integration review

Before any vendor integration goes live it must pass four steps: a completed
security questionnaire, a data flow diagram identifying every personal data
field exchanged, a signed data processing agreement, and sign-off from the
Information Security review board. No production credentials may be issued
before board sign-off.

## Data classification

We use four classes: public, internal, confidential, and regulated. Regulated
covers financial decisioning data, health data, and government identifiers.
Regulated data may not be sent to a model provider without a signed data
processing agreement and confirmed regional data residency.

## Personal data in prompts

Application teams must not place government identifiers, full payment card
numbers, or health records in model prompts. Where a workflow genuinely requires
an identifier, it must be tokenised before the call and re-hydrated after.

## Incident reporting

Suspected data exposure must be reported to the security on-call channel within
one hour of discovery. The reporting duty applies even when exposure is only
suspected and not confirmed.

## AI-specific controls

All generative AI traffic must pass through the approved gateway so that
routing, screening, and audit logging are applied. Direct calls from application
code to a model provider are a policy violation.
