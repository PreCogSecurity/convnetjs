#!/usr/bin/env node
"use strict";

// Concatenates the src/ modules into build/convnet.js in dependency order.
//
// This is a dependency-free replacement for the ant task in compile/build.xml
// (which requires ant + a JVM + the yuicompressor jar). The file order below
// must stay in sync with build.xml so both paths produce identical output.
//
// Usage: node compile/build.js

var fs = require('fs');
var path = require('path');

var srcDir = path.join(__dirname, '..', 'src');
var outFile = path.join(__dirname, '..', 'build', 'convnet.js');

var files = [
  'convnet_init.js',
  'convnet_util.js',
  'convnet_vol.js',
  'convnet_vol_util.js',
  'convnet_layers_dotproducts.js',
  'convnet_layers_pool.js',
  'convnet_layers_input.js',
  'convnet_layers_loss.js',
  'convnet_layers_nonlinearities.js',
  'convnet_layers_dropout.js',
  'convnet_layers_normalization.js',
  'convnet_net.js',
  'convnet_trainers.js',
  'convnet_magicnet.js',
  'convnet_export.js'
];

var parts = files.map(function(f) {
  var p = path.join(srcDir, f);
  if (!fs.existsSync(p)) {
    console.error('Missing source file: ' + p);
    process.exit(1);
  }
  return fs.readFileSync(p, 'utf8');
});

// fixlastline="true" in build.xml guarantees the output ends with a newline.
var output = parts.join('\n') + '\n';
fs.writeFileSync(outFile, output);
console.log('Wrote ' + path.relative(process.cwd(), outFile) + ' (' + output.length + ' bytes)');