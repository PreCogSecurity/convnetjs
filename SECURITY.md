# Security Policy

## Reporting a vulnerability

Please **do not open a public issue** for a security problem.

Report it privately through GitHub's private vulnerability reporting on this
repository ("Security" tab → "Report a vulnerability"), or by email to the
maintainers. Please include:

- what the issue is and which file or API is involved,
- how to reproduce it, ideally a minimal snippet,
- the impact you believe it has.

You can expect an acknowledgement within a few business days and an assessment
within two weeks. We are happy to credit you in the fix and in `CHANGELOG.md`
if you would like that; tell us how you would like to be named.

Please give us a reasonable window to ship a fix before disclosing publicly.
We aim to acknowledge the constraint and will keep you updated.

## Scope

`convnetjs` is a pure-JavaScript neural network library with **no runtime
dependencies**. It executes in a browser or in Node, and it is often pointed at
data that came from somewhere else, so these areas are the ones we care about
most:

- **Model deserialization** (`Net.fromJSON`, `Vol.fromJSON`,
  `MagicNet.fromJSON`). These accept attacker-controllable documents and are
  treated as an untrusted-input boundary. Dimensions, weight values, layer
  types, and allocation sizes are validated; loads are atomic. A bypass here is
  in scope.
- **Memory exhaustion.** A crafted volume dimension must not be able to make the
  host allocate unbounded memory or exhaust the stack.
- **Type confusion** in anything that reaches a typed array.
- **The demo pages**, which are served as a static site. We are particularly
  interested in anything that reintroduces code execution from page input: the
  demos previously `eval()`-ed the text typed into their network-definition
  textareas, and that was replaced with a validating JSON parser in
  `demo/js/net_spec.js`. A regression to dynamic evaluation of page input is in
  scope.
- **The release pipeline.** Published artifacts should carry npm provenance and
  be traceable to a reviewed commit.

Out of scope:

- Vulnerabilities in a consumer's own application code that uses the library
  correctly.
- Model files the consumer wrote themselves and chose to trust.
- Denial of service caused by intentionally training an enormous network.
- Missing hardening in a demo page that requires the attacker to already
  control the same origin.

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.3.x   | Yes      |
| < 0.3   | No       |

Security fixes land on the default branch and are released under the version in
the table above. See `CHANGELOG.md` for what has shipped.
