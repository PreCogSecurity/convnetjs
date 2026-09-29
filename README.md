
# ConvNetJS

ConvNetJS is a Javascript implementation of Neural networks, together with nice browser-based demos. It currently supports:

- Common **Neural Network modules** (fully connected layers, non-linearities)
- Classification (SVM/Softmax) and Regression (L2) **cost functions**
- Ability to specify and train **Convolutional Networks** that process images
- An experimental **Reinforcement Learning** module, based on Deep Q Learning

For much more information, see the main page at [convnetjs.com](http://convnetjs.com)

## Installation

**From npm (Node.js >= 18):**

```bash
npm install convnetjs
```

**From source (for development):**

```bash
git clone https://github.com/PreCogSecurity/convnetjs.git
cd convnetjs
npm install
```

This builds the concatenated library (`build/convnet.js`) from the individual source files in `src/`.

## Use in Node

```javascript
var convnetjs = require("convnetjs");

var layer_defs = [];
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});
layer_defs.push({type:'fc', num_neurons:20, activation:'relu'});
layer_defs.push({type:'softmax', num_classes:10});

var net = new convnetjs.Net();
net.makeLayers(layer_defs);

var x = new convnetjs.Vol([0.3, -0.5]);
var prob = net.forward(x);
console.log('probabilities:', prob.w);
```

## Running the Demos

After installing, open any demo page in a browser:

```bash
# Using a simple static file server (pick any)
npx http-server .
# then open http://localhost:8080/demo/classify2d.html
```

Or with Docker:

```bash
docker compose up
# then open http://localhost:8080/demo/classify2d.html
```

The Compose service binds to **loopback only** (`127.0.0.1:8080:80`) and runs with a read-only root filesystem, `no-new-privileges`, and nginx as a non-root user. The demo site is unauthenticated, so it should not be published on a routable interface by accident. To reach it from another host, change the port mapping deliberately.

Note that the image contains only `build/`, `demo/`, `LICENSE` and `README.md`. The repository history (`.git/`) and `node_modules/` are excluded by `.dockerignore` and are not part of the image.

### Available Demos

- [Convolutional Neural Network on MNIST digits](demo/mnist.html)
- [Convolutional Neural Network on CIFAR-10](demo/cifar10.html)
- [Toy 2D data](demo/classify2d.html)
- [Toy 1D regression](demo/regression.html)
- [Training an Autoencoder on MNIST digits](demo/autoencoder.html)
- [Deep Q Learning Reinforcement Learning demo](demo/rldemo.html)
- [Image Regression ("Painting")](demo/image_regression.html)
- [Comparison of SGD/Adagrad/Adadelta on MNIST](demo/trainers.html)
- [Automatic prediction with MagicNet](demo/automatic.html)

## Running the Tests

```bash
npm test
```

This rebuilds the library from `src/` and then runs the Jasmine test suite in `test/jasmine/`.

### Coverage

```bash
npm run test:coverage
```

Runs the suite under [`c8`](https://github.com/bcoe/c8) and writes a text summary plus LCOV output to `coverage/`. The thresholds live in the `c8` block in `package.json` and are enforced (`check-coverage`), so the command exits non-zero if coverage drops below them. CI runs this as a gate, which is what stops an untested change from silently landing.

## Linting

```bash
npm run lint
```

## Loading models safely

`Net.fromJSON` and `MagicNet.fromJSON` are the entry points for model files, and model files frequently arrive from a user upload or a remote URL. Treat them as untrusted input, because that is what they are:

- Dimensions and weight values are validated before anything is allocated. A missing or non-numeric dimension now raises a clear error instead of quietly producing an empty `Vol` that loads "successfully" and then returns garbage predictions.
- A volume larger than `convnetjs.MAX_VOL_SIZE` (100,000,000 elements) is rejected, so a hostile `sx * sy * depth` cannot ask the host for gigabytes and take the process down.
- An unrecognized `layer_type` raises an error naming the type, rather than an opaque `TypeError`.
- Layer construction is looked up with a prototype-safe check, so a crafted `layer_type` of `constructor` or `__proto__` cannot resolve to a built-in.
- Loading is **atomic**: layers are built into a temporary list and only committed once every layer deserialized cleanly. A rejected payload leaves your existing network untouched rather than half-replaced.

```javascript
var net = new convnetjs.Net();
try {
  net.fromJSON(JSON.parse(untrustedJson));
} catch (e) {
  // e.message names the offending field or layer index
  console.warn('rejecting model file:', e.message);
}
```

`net.toJSON()` stores weights in a `Float64Array`, and `JSON.stringify` serializes a typed array as an object keyed by stringified index rather than as a list. `fromJSON` accepts both that shape and a plain array, so a round trip through `JSON.stringify` is safe.

## Reproducible runs

`randn` is now stateless: it samples iteratively (Marsaglia polar method) instead of caching the spare Box-Muller deviate in module-level state. Two consequences worth knowing:

- Seeding `Math.random` now actually gives you reproducible weight initialization. Previously the cached spare leaked between unrelated networks, so a second network perturbed the first one's weights.
- A constant or stubbed `Math.random` no longer crashes the library. The old implementation recursed on rejected sample pairs without bound, so a degenerate RNG exhausted the call stack and threw `RangeError: Maximum call stack size exceeded`.

```javascript
Math.random = mySeededRandom;   // now sufficient for reproducibility
var net = new convnetjs.Net();
net.makeLayers(layer_defs);
```

## Example Code

Here's a minimum example of defining a **2-layer neural network** and training
it on a single data point:

```javascript
// species a 2-layer neural network with one hidden layer of 20 neurons
var layer_defs = [];
// input layer declares size of input. here: 2-D data
// ConvNetJS works on 3-Dimensional volumes (sx, sy, depth), but if you're not dealing with images
// then the first two dimensions (sx, sy) will always be kept at size 1
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});
// declare 20 neurons, followed by ReLU (rectified linear unit non-linearity)
layer_defs.push({type:'fc', num_neurons:20, activation:'relu'});
// declare the linear classifier on top of the previous hidden layer
layer_defs.push({type:'softmax', num_classes:10});

var net = new convnetjs.Net();
net.makeLayers(layer_defs);

// forward a random data point through the network
var x = new convnetjs.Vol([0.3, -0.5]);
var prob = net.forward(x);

// prob is a Vol. Vols have a field .w that stores the raw data, and .dw that stores gradients
console.log('probability that x is class 0: ' + prob.w[0]); // prints 0.50101

var trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.01, l2_decay:0.001});
trainer.train(x, 0); // train the network, specifying that x is class zero

var prob2 = net.forward(x);
console.log('probability that x is class 0: ' + prob2.w[0]);
// now prints 0.50374, slightly higher than previous 0.50101: the networks
// weights have been adjusted by the Trainer to give a higher probability to
// the class we trained the network with (zero)
```

and here is a small **Convolutional Neural Network** if you wish to predict on images:

```javascript
var layer_defs = [];
layer_defs.push({type:'input', out_sx:32, out_sy:32, out_depth:3}); // declare size of input
// output Vol is of size 32x32x3 here
layer_defs.push({type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'});
// the layer will perform convolution with 16 kernels, each of size 5x5.
// the input will be padded with 2 pixels on all sides to make the output Vol of the same size
// output Vol will thus be 32x32x16 at this point
layer_defs.push({type:'pool', sx:2, stride:2});
// output Vol is of size 16x16x16 here
layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
// output Vol is of size 16x16x20 here
layer_defs.push({type:'pool', sx:2, stride:2});
// output Vol is of size 8x8x20 here
layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
// output Vol is of size 8x8x20 here
layer_defs.push({type:'pool', sx:2, stride:2});
// output Vol is of size 4x4x20 here
layer_defs.push({type:'softmax', num_classes:10});
// output Vol is of size 1x1x10 here

net = new convnetjs.Net();
net.makeLayers(layer_defs);

// helpful utility for converting images into Vols is included
var x = convnetjs.img_to_vol(document.getElementById('#some_image'))
var output_probabilities_vol = net.forward(x)
```

## Getting Started

A [Getting Started](http://cs.stanford.edu/people/karpathy/convnetjs/started.html) tutorial is available on the main page.

The full [Documentation](http://cs.stanford.edu/people/karpathy/convnetjs/docs.html) can also be found there.

## Architecture

The library source lives in `src/`, split into focused per-concern modules:

| File | Purpose |
|------|---------|
| `convnet_init.js` | Library namespace (`convnetjs` global) |
| `convnet_util.js` | Random number generators, array utilities, `assert()` |
| `convnet_vol.js` | `Vol` -- the fundamental 3-D data structure |
| `convnet_vol_util.js` | Vol helpers (`augment`, `img_to_vol`, `maxmin`) |
| `convnet_layers_input.js` | Input layer |
| `convnet_layers_dotproducts.js` | Fully connected (`fc`) and convolutional (`conv`) layers |
| `convnet_layers_pool.js` | Max-pooling layer |
| `convnet_layers_nonlinearities.js` | ReLU, Sigmoid, Tanh, Maxout |
| `convnet_layers_loss.js` | Softmax, SVM, Regression loss layers |
| `convnet_layers_dropout.js` | Dropout layer |
| `convnet_layers_normalization.js` | Local Response Normalization (LRN) |
| `convnet_net.js` | `Net` -- assembles layers, forward/backward pass, serialization |
| `convnet_trainers.js` | `Trainer` (SGD, Adam, Adagrad, Adadelta, Nesterov) |
| `convnet_magicnet.js` | `MagicNet` -- automatic hyperparameter search |
| `convnet_export.js` | Node.js `module.exports` for `require()` usage |

These are concatenated (in dependency order) by `compile/build.js` into the single distributable file `build/convnet.js`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for coding style guidelines and instructions on adding a new layer type with matching test specs.

## Releasing

See [RELEASE.md](RELEASE.md) for the tag-and-publish flow, including the one-time npm Trusted Publishing setup.

## Security

Please report vulnerabilities privately rather than in a public issue. See [SECURITY.md](SECURITY.md) for the disclosure policy and supported versions.

## License

MIT
