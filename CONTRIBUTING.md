# Contributing to ConvNetJS

Thanks for your interest in contributing! This document describes the coding
style, how to run the checks, and how to add a new layer type with a matching
spec file.

## Development setup

```sh
npm install     # installs dev dependencies and builds build/convnet.js
npm test        # runs the Jasmine spec suite headlessly
npm run lint    # runs ESLint over src/ and the test specs
npm run build   # regenerates build/convnet.js from src/
npm run test:coverage   # runs the suite under c8 and enforces the thresholds
```

Before opening a pull request, make sure `npm test`, `npm run lint` and
`npm run test:coverage` all pass. CI runs all three on every push, on Node 18,
20 and 22, plus `npm audit --audit-level=high` and the workflow/Dockerfile
linters.

### Tests and coverage

**Ship a test with every change.** A fix without a spec that pins the new
behavior is not finished, and a feature that arrives untested will be reverted
by the coverage gate before anyone notices it is wrong.

Coverage thresholds live in the `c8` block in `package.json` and are enforced by
`check-coverage`, so `npm run test:coverage` exits non-zero when coverage
regresses. If you legitimately lower coverage, lower the thresholds in the same
commit and say why in the PR description.

Tests must be **deterministic**. Network weight initialization draws from
`Math.random`, so a spec that builds a network has to seed it:

```javascript
var originalRandom;
beforeEach(function() { originalRandom = Math.random; });
afterEach(function() { Math.random = originalRandom; });

function makeSeededRandom(seed) {
  var s = seed >>> 0;
  return function() {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}
```

When asserting that training converges, compare **windowed mean loss** over a
leading and a trailing window rather than a single first and last sample. With
`batch_size: 1` the per-step loss is noisy enough that a two-point comparison
flips on ordering alone. `TrainerSpec.js` shows the pattern.

## Coding style

The codebase is classic ES5 JavaScript (it predates ES6 tooling), and new code
should match that style:

- Use `var`, not `let`/`const`.
- Use function expressions and prototype-based objects, not ES6 classes.
- No semicolon-less style: the existing code uses semicolons.
- Keep each source module focused on one concern (see the `src/` layout in
  [README.md](README.md#architecture)).
- Wrap each module in an IIFE that receives the `convnetjs` namespace:
  `(function(global) { "use strict"; ... })(convnetjs);`
- Use `global.getopt(opt, 'field', default)` for optional constructor
  parameters, and `global.assert(condition, message)` for invariants.
- Do not introduce new dependencies unless a discussion in the PR justifies
  them.

## Handling untrusted input

`fromJSON` is a trust boundary. Model files routinely come from a user upload or
a remote URL, so anything read from a serialized document must be validated
before it is used to size an allocation or index an array, and a rejected load
must leave the object untouched. See the "Loading models safely" section of
[README.md](README.md#loading-models-safely) for the guarantees and the helpers
in `src/convnet_util.js` (`isNonNegInt`, `isPlainish`, `validateVolDims`).

Concretely:

- look up layer constructors with a prototype-safe check, never a bare
  `map[type]`;
- build into a local array and assign only after every element succeeded, so a
  failure is atomic;
- reject `NaN`, `Infinity`, negative and fractional dimensions;
- bound any allocation derived from document contents.

## How to add a new layer type

1. Create a new file `src/convnet_layers_<name>.js` (or add the layer to the
   most fitting existing file) following the module pattern above. A layer is
   an object with:
   - a constructor that reads `opt.in_sx`, `opt.in_sy`, `opt.in_depth` and
     computes `out_sx`, `out_sy`, `out_depth`, and `layer_type`;
   - `forward(V, is_training)` returning the output `Vol`;
   - `backward()` computing gradients into `this.in_act.dw` (and any
     parameter gradients);
   - `getParamsAndGrads()` returning `[{params, grads, l1_decay_mul,
     l2_decay_mul}, ...]` (empty array if the layer has no parameters);
   - `toJSON()` / `fromJSON(json)` for serialization.
2. Register the layer type in `src/convnet_net.js`:
   - in the `switch` inside `makeLayers()` (so `{type:'<name>', ...}` works),
   - in the `fromJSON()` type dispatch **and** in its constructor lookup table.
     The lookup is a prototype-safe table, not a bare property access, so that
     a crafted `layer_type` cannot resolve to a built-in.
3. Add a spec file `test/jasmine/spec/<LayerName>Spec.js` that covers:
   - output shape for a known input size,
   - a numeric gradient check at the data (see the pattern in
     `ConvLayerSpec.js` / `NeuralNetSpec.js`),
   - a `toJSON`/`fromJSON` round trip,
   - rejection of malformed input in `fromJSON` (missing fields, wrong types,
     oversized dimensions, unknown `layer_type`), and that a rejected load
     leaves the layer's previous state intact.
4. Run `npm test`, `npm run lint` and `npm run test:coverage`.

## How to add a trainer method

1. Add the update rule in the `train()` method of `src/convnet_trainers.js`
   (the `if(this.method === '...')` chain).
2. Add a "should decrease loss with <method>" spec to `TrainerSpec.js` using
   the seeded-PRNG pattern already there, so the test is reproducible.

## Commit guidelines

- Keep each feature or fix in its own commit (or small PR) that includes the
  tests pinning the new behavior.
- Avoid bulk commits that mix formatting, refactors, and features.
- Run `npm test` locally before pushing; CI will run it again.

## Reporting issues

Please open a GitHub issue with a minimal reproduction: the layer definitions,
the trainer options, and the expected vs. observed behavior.

## Reporting security issues

Do not open a public issue for a security problem. See
[SECURITY.md](SECURITY.md) for the private reporting path and the scope.