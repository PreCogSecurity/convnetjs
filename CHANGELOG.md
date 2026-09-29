# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security

- **Model deserialization is now treated as an untrusted-input boundary.**
  `Net.fromJSON`, `Vol.fromJSON` and `MagicNet.fromJSON` accept documents that
  in practice come from a user upload or a remote URL. Previously a corrupt or
  hostile payload could load "successfully" and produce silently wrong results,
  or ask the host for an unbounded allocation:
  - `Vol.fromJSON` silently produced an **empty** volume when a dimension was
    missing or `NaN` (because `zeros(NaN)` returns `[]`), so a truncated model
    file loaded without error and then returned garbage. Dimensions are now
    validated and the failure is reported.
  - Non-numeric, `null` or `NaN` weights were copied straight into the
    `Float64Array`; they are now rejected.
  - A volume larger than `MAX_VOL_SIZE` (100,000,000 elements) is rejected before
    allocation, so a crafted `sx * sy * depth` cannot exhaust host memory.
  - An unrecognized `layer_type` produced an opaque
    `TypeError: Cannot read properties of undefined`. It now raises an error
    naming the offending type and index.
  - Layer construction resolves through a prototype-safe lookup, so a crafted
    `layer_type` of `constructor` or `__proto__` cannot resolve to a built-in.
  - **Loading is now atomic.** Layers are built into a temporary list and only
    committed once every layer deserialized cleanly, so a rejected payload no
    longer leaves a half-built network in place for a caller that catches the
    error and keeps using it.
- **`randn` no longer recurses without bound.** The Box-Muller sampler rejected
  invalid sample pairs by recursing. With a degenerate or stubbed `Math.random`
  — a common way to get reproducible training — every pair was rejected and the
  library threw `RangeError: Maximum call stack size exceeded`. Sampling is now
  iterative with a bounded retry count and degrades to a zero deviate rather
  than crashing the caller.

### Fixed

- `MaxoutLayer.fromJSON` allocated its backpropagation switch table with
  `group_size` elements instead of `out_sx * out_sy * out_depth`. A maxout network
  restored from JSON therefore had an undersized table, and `backward()` read
  past its end, writing gradients at `undefined` indices. **Training a restored
  maxout network poisoned its parameter gradients with `NaN` on the first step.**
- `Net.makeLayers` desugaring read `def.group_size !== 'undefined'`, which
  compares the *value* against the string `'undefined'` and was therefore always
  true. The intended default only worked by accident, because `MaxoutLayer`
  re-applied its own. An explicit `null` or `0` slipped through, producing a
  `NaN` or infinite `out_depth`. `group_size` is now resolved with a correct
  `typeof` check and validated as a positive integer, in both the desugaring path
  and the `MaxoutLayer` constructor.
- The trainer convergence specs compared a single first loss sample against a
  single last sample. Because training is single-example (`batch_size: 1`), that
  comparison was decided by ordering noise and the suite was flaky. They now
  compare windowed mean loss.

### Changed

- `randn` is stateless. It previously cached the spare Box-Muller deviate in
  module-level state, so that cached value leaked between unrelated networks: a
  second network perturbed the first one's weights, and seeding `Math.random`
  did not yield reproducible initialization. Networks are now independent and
  seeding works. The distribution is unchanged.
- The demo pages emit structured JSON-line logs via the new `demo/js/logger.js`
  instead of raw `console.log`. Output is level-gated (`demoLog.setLevel`),
  caller metadata cannot spoof the `ts`/`level`/`msg` envelope, and the logger
  cannot throw on circular structures or throwing getters.

### Added

- Test coverage for `convnet_magicnet.js` (`MagicNetSpec.js`), which was the
  largest module in `src/` and had no spec at all: candidate sampling, fold
  construction, the training loop, validation scoring, ensembling, and
  round-trip serialization.
- Specs covering every deserialization fix, the maxout fixes, and the RNG
  robustness and independence guarantees.
- Enforced coverage. `npm run test:coverage` runs the suite under `c8` with
  `check-coverage` thresholds configured in `package.json`, and CI gates on it,
  so a regression in coverage fails the build. The library is at 82.5%
  statements / 89.2% branches / 77.0% functions.
- A demo logger spec (`LoggerSpec.js`).
- `SECURITY.md` with a private disclosure policy and an explicit scope.
- `RELEASE.md` documenting the tag-and-publish flow.
- `.github/dependabot.yml` (weekly, for npm and GitHub Actions).
- `.github/workflows/release.yml`: publishes on a `v*` tag with npm provenance,
  after verifying the tag matches `package.json`.
- `.dockerignore`, so the demo image no longer ships the repository history.

### Infrastructure

- CI now runs the build, lint, test and coverage gate on Node 18, 20 and 22.
- CI declares `permissions: contents: read` at the workflow level instead of
  relying on the default token scope, and uses `concurrency` groups to cancel
  superseded runs.
- CI runs `npm audit --audit-level=high` after `npm ci`.
- CI adds `actionlint`, `hadolint` and `docker compose config` validation.

### Security (container)

- The demo image previously ran `COPY . /usr/share/nginx/html`, which published
  the entire `.git/` directory and `node_modules/` as static HTTP content. The
  image now copies only `build/`, `demo/`, `LICENSE` and `README.md`, and runs
  nginx as a non-root user.
- `docker-compose.yml` now binds to `127.0.0.1:8080` instead of all interfaces,
  and runs with a read-only root filesystem, `no-new-privileges`, tmpfs for the
  nginx writable paths, and a healthcheck.

## [0.3.0]

- Prior release. See the repository history for details.
