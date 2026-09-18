// net_spec.js
//
// Shared, eval()-free parsers for the demo "network definition" textareas.
//
// The demos used to eval() whatever the user typed into the textarea, which
// executed arbitrary JavaScript. Instead, the textarea now holds a JSON
// document with a constrained schema:
//
// {
//   "layer_defs": [ ...layer definition objects... ],
//   "trainer": { ...trainer options... },        // optional
//   "trainer_defs": [ {...}, ... ],              // optional (trainers demo)
//   "legend": [ "name", ... ]                    // optional (trainers demo)
// }
//
// parseNetSpec(text) validates the document against the schema and returns
// {layer_defs, trainer, trainer_defs, legend}. It throws an Error with a
// human-readable message on any validation failure, so the caller can surface
// the problem to the user.
//
// parseBrainSpec(text) validates the schema used by the reinforcement
// learning demo (rldemo.html):
//
// {
//   "num_inputs": 27,
//   "num_actions": 5,
//   "temporal_window": 1,                        // optional
//   "layer_defs": [ ... ],                       // value-function network
//   "tdtrainer_options": { ... },                // optional
//   "opt": { ... }                               // optional Brain options
// }
//
// It returns {num_inputs, num_actions, temporal_window, layer_defs,
// tdtrainer_options, opt}.

(function(global) {
  "use strict";

  var LAYER_TYPES = ['input', 'fc', 'conv', 'pool', 'relu', 'sigmoid', 'tanh',
    'maxout', 'softmax', 'svm', 'regression', 'dropout', 'lrn'];

  var TRAINER_METHODS = ['sgd', 'adam', 'adagrad', 'windowgrad', 'adadelta', 'nesterov'];

  // required fields per layer type (all other fields are optional and are
  // passed through to the layer constructors / trainer untouched)
  var REQUIRED_FIELDS = {
    input: ['out_depth'],
    fc: ['num_neurons'],
    conv: ['sx', 'filters'],
    pool: ['sx'],
    softmax: ['num_classes'],
    svm: ['num_classes'],
    regression: ['num_neurons'],
    relu: [],
    sigmoid: [],
    tanh: [],
    maxout: [],
    dropout: [],
    lrn: []
  };

  function isPlainObject(v) {
    return typeof v === 'object' && v !== null && !(v instanceof Array);
  }

  // Strips // line comments and /* block comments */ from a JSON document,
  // while leaving strings (including "http://..." style URLs) untouched.
  // This lets the demo textareas keep their explanatory comments.
  function stripComments(text) {
    return text.replace(/"(?:\\"|[^"])*"|\/\/.*|\/\*[\s\S]*?\*\//g, function(m) {
      return m.charAt(0) === '"' ? m : '';
    });
  }

  function validateLayerDefs(layer_defs, context) {
    if (!(layer_defs instanceof Array) || layer_defs.length < 2) {
      throw new Error(context + ' must contain a "layer_defs" array with at least 2 layers (input + loss).');
    }

    var out = [];
    for (var i = 0; i < layer_defs.length; i++) {
      var def = layer_defs[i];
      if (!isPlainObject(def)) {
        throw new Error(context + ' layer definition at index ' + i + ' must be a JSON object.');
      }
      if (typeof def.type !== 'string' || LAYER_TYPES.indexOf(def.type) === -1) {
        throw new Error(context + ' layer definition at index ' + i + ' has unknown type "' + def.type +
          '". Allowed types: ' + LAYER_TYPES.join(', ') + '.');
      }
      var required = REQUIRED_FIELDS[def.type];
      for (var j = 0; j < required.length; j++) {
        if (typeof def[required[j]] === 'undefined') {
          throw new Error(context + ' layer definition at index ' + i + ' (type "' + def.type +
            '") is missing required field "' + required[j] + '".');
        }
      }
      out.push(def);
    }

    if (out[0].type !== 'input') {
      throw new Error(context + ': the first layer must be of type "input" to declare the size of the inputs.');
    }
    return out;
  }

  function validateTrainerOptions(trainer, context) {
    if (!isPlainObject(trainer)) {
      throw new Error(context + ' must be a JSON object of trainer options.');
    }
    if (typeof trainer.method !== 'undefined' && TRAINER_METHODS.indexOf(trainer.method) === -1) {
      throw new Error(context + ' has unknown trainer method "' + trainer.method +
        '". Allowed methods: ' + TRAINER_METHODS.join(', ') + '.');
    }
    return trainer;
  }

  function parseNetSpec(text) {
    if (typeof text !== 'string' || text.trim() === '') {
      throw new Error('Network definition is empty. Provide a JSON document with a "layer_defs" array.');
    }

    var spec;
    try {
      spec = JSON.parse(stripComments(text));
    } catch (e) {
      throw new Error('Network definition is not valid JSON: ' + e.message);
    }

    if (!isPlainObject(spec)) {
      throw new Error('Network definition must be a JSON object with a "layer_defs" array.');
    }
    if (typeof spec.layer_defs === 'undefined') {
      throw new Error('Network definition must contain a "layer_defs" array.');
    }

    var layer_defs = validateLayerDefs(spec.layer_defs, 'Network definition');

    var trainer = {};
    if (typeof spec.trainer !== 'undefined') {
      trainer = validateTrainerOptions(spec.trainer, '"trainer"');
    }

    var trainer_defs = [];
    if (typeof spec.trainer_defs !== 'undefined') {
      if (!(spec.trainer_defs instanceof Array) || spec.trainer_defs.length === 0) {
        throw new Error('"trainer_defs" must be a non-empty array of trainer option objects.');
      }
      for (var i = 0; i < spec.trainer_defs.length; i++) {
        trainer_defs.push(validateTrainerOptions(spec.trainer_defs[i], '"trainer_defs" entry at index ' + i));
      }
    }

    var legend = [];
    if (typeof spec.legend !== 'undefined') {
      if (!(spec.legend instanceof Array) || spec.legend.length === 0) {
        throw new Error('"legend" must be a non-empty array of strings.');
      }
      for (var k = 0; k < spec.legend.length; k++) {
        if (typeof spec.legend[k] !== 'string') {
          throw new Error('"legend" entry at index ' + k + ' must be a string.');
        }
        legend.push(spec.legend[k]);
      }
    }

    return {layer_defs: layer_defs, trainer: trainer, trainer_defs: trainer_defs, legend: legend};
  }

  function parseBrainSpec(text) {
    if (typeof text !== 'string' || text.trim() === '') {
      throw new Error('Q-learner specification is empty. Provide a JSON document with "num_inputs", "num_actions" and "layer_defs".');
    }

    var spec;
    try {
      spec = JSON.parse(stripComments(text));
    } catch (e) {
      throw new Error('Q-learner specification is not valid JSON: ' + e.message);
    }

    if (!isPlainObject(spec)) {
      throw new Error('Q-learner specification must be a JSON object.');
    }

    var num_inputs = spec.num_inputs;
    if (typeof num_inputs !== 'number' || !isFinite(num_inputs) || num_inputs <= 0 || Math.floor(num_inputs) !== num_inputs) {
      throw new Error('"num_inputs" must be a positive integer.');
    }
    var num_actions = spec.num_actions;
    if (typeof num_actions !== 'number' || !isFinite(num_actions) || num_actions <= 0 || Math.floor(num_actions) !== num_actions) {
      throw new Error('"num_actions" must be a positive integer.');
    }
    var temporal_window = typeof spec.temporal_window !== 'undefined' ? spec.temporal_window : 1;
    if (typeof temporal_window !== 'number' || !isFinite(temporal_window) || temporal_window <= 0 || Math.floor(temporal_window) !== temporal_window) {
      throw new Error('"temporal_window" must be a positive integer.');
    }

    if (typeof spec.layer_defs === 'undefined') {
      throw new Error('Q-learner specification must contain a "layer_defs" array.');
    }
    var layer_defs = validateLayerDefs(spec.layer_defs, 'Q-learner specification');
    var last = layer_defs[layer_defs.length - 1];
    if (last.type !== 'regression') {
      throw new Error('Q-learner specification: the last layer must be of type "regression" (the value function output).');
    }

    var tdtrainer_options = {};
    if (typeof spec.tdtrainer_options !== 'undefined') {
      tdtrainer_options = validateTrainerOptions(spec.tdtrainer_options, '"tdtrainer_options"');
    }

    var opt = {};
    if (typeof spec.opt !== 'undefined') {
      if (!isPlainObject(spec.opt)) {
        throw new Error('"opt" must be a JSON object of Brain options.');
      }
      opt = spec.opt;
    }
    opt.temporal_window = temporal_window;

    return {
      num_inputs: num_inputs,
      num_actions: num_actions,
      temporal_window: temporal_window,
      layer_defs: layer_defs,
      tdtrainer_options: tdtrainer_options,
      opt: opt
    };
  }

  global.parseNetSpec = parseNetSpec;
  global.parseBrainSpec = parseBrainSpec;
})(window);