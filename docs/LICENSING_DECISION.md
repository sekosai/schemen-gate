# Licensing policy

Status: **adopted for the Schemen Gate repository.** This document explains the
project's license files and adoption intent; it is not legal advice and does not
change repository visibility, create a release, or replace the license texts.

## Adoption-first policy

The repository uses a simple path-based policy:

- **Outside `research/cdp/`: Apache License 2.0** under the root `LICENSE`,
  including the Gate library, root examples, build and deployment scripts,
  tests, and root documentation unless a path explicitly states otherwise.
- **Inside `research/cdp/`: apply `research/cdp/LICENSES.md`.** That path map
  assigns Apache-2.0 to executable code and Lean proof source, and CC BY 4.0
  only to the designated authored papers, explanatory research prose, figures,
  and result records.
- Keep third-party material under its existing terms and record every exception
  in the relevant notice and provenance files.

This policy is represented by the root `LICENSE` and `NOTICE`, plus
`research/cdp/LICENSES.md` for the nested research bundle. Those files, rather
than this explanation, are the authoritative license instruments and path map;
the presence of CC BY material under `research/cdp/` does not relicense root
documentation or other root-repository paths.

## Why this fits the goal

Apache-2.0 is permissive: companies may use, modify, and redistribute the code,
including commercially. Unlike a minimal permissive license, it contains an
express contributor patent grant and patent-litigation termination language.
That makes the patent posture legible to enterprise adopters and reduces a
common source of procurement uncertainty.

CC BY 4.0 fits papers and authored research material better than a software
license. It permits sharing and adaptation, including commercial use, provided
attribution is given and changes are identified.

Do not add a noncommercial, geographic, ethical-use, or field-of-use
restriction while calling the result open source. The Open Source Definition
requires free redistribution and forbids discrimination against fields of
endeavor. Those restrictions would directly fight the adoption objective.

## Patent consequence: precise, not euphemistic

Publishing under Apache-2.0 is not the same thing as erasing a patent
application. It does, however, grant recipients a royalty-free patent license
within the scope defined by Section 3 of Apache-2.0 for applicable patent claims
licensable by a contributor that are necessarily infringed by the contribution
alone or in its combination with the work. That can materially reduce or remove
the ability to charge users of the open-source contribution for those covered
uses.

The project's adoption decision is:

> Prefer broad Gate adoption and an explicit patent peace for the shipped Gate
> implementation, while reserving any rights genuinely outside the licensed
> contribution and its defined combinations.

This repository does not characterize unpublished claims or predict which
claims may ultimately issue. Filing deadlines, disclosure consequences, and
rights outside the licensed contribution are maintained separately from the
open-source tree.

### Public patent notice

The public-facing repository documentation and packaged `NOTICE` use this
adoption-first notice:

> A U.S. provisional patent application was filed before release for subject
> matter related to portions of Schemen Gate. This notice adds no separate
> restriction. For Apache-2.0 material, patent rights are governed by Section 3
> of the Apache License 2.0. CC BY 4.0 does not license patent rights.

Do not publish the application number, internal docket, unpublished claims,
filing documents, receipts, or prosecution records. Those records belong in
private rights-holder custody. The notice is informational; it does not add a
license, reservation, field-of-use condition, or other term beyond the
applicable license text.

## Options considered

- **MIT or BSD:** very low friction, but less explicit patent treatment is
  unhelpful when a patent filing already exists.
- **GPL or AGPL:** strong reciprocity, but more integration and procurement
  friction than fits a widely embedded execution boundary.
- **Source-available or noncommercial terms:** may preserve control, but are not
  open source under the Open Source Definition.
- **Dual licensing:** usually derives leverage from copyleft. Apache-2.0 already
  grants broad commercial rights, so revenue should come from operational value
  around the complete open Gate rather than exceptions to a crippled core.

## Decision record

- Decision: **adoption-first open-source distribution**
- Root paths outside `research/cdp/`: **Apache-2.0 unless explicitly stated**
- Paths inside `research/cdp/`: **the Apache-2.0/CC BY 4.0 classifications in
  `research/cdp/LICENSES.md`**
- Patent posture: **Apache-2.0 Section 3 governs the shipped Apache-2.0
  contribution; the public notice adds no additional restriction**
- Release operations: **manual maintainer actions, independently verified
  against the tagged commit and artifact digests**

## Primary references

- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- [Apache guidance for applying the license](https://www.apache.org/legal/apply-license.html)
- [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
- [Open Source Definition](https://opensource.org/osd)
- [USPTO provisional application guidance](https://www.uspto.gov/patents/basics/apply/provisional-application)
- [USPTO guidance on filing abroad](https://www.uspto.gov/patents/basics/international-protection/filing-patents-abroad)
