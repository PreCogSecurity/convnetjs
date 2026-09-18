// Loads the concatenated library exactly the way the browser SpecRunner does
// (via build/convnet.js) and exposes it as a global so the specs can keep
// referencing `convnetjs` the way they do in the browser.
var convnetjs = require('../../../build/convnet.js');
global.convnetjs = convnetjs;