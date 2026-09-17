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
```

Before opening a pull request, make sure `npm test` and `npm run lint` both
pass. CI runs `npm run build`, `npm run lint`, and `npm test` on every push.

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
   - in the `fromJSON()` type dispatch.
3. Add a spec file `test/jasmine/spec/<LayerName>Spec.js` that covers:
   - output shape for a known input size,
   - a numeric gradient check at the data (see the pattern in
     `ConvLayerSpec.js` / `NeuralNetSpec.js`),
   - a `toJSON`/`fromJSON` round trip.
4. Run `npm test` and `npm run lint` and make sure everything passes.

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