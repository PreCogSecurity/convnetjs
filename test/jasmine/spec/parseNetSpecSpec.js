describe("parseNetSpec (demo network definition parser)", function() {

  var parseNetSpec;
  var parseBrainSpec;

  beforeAll(function() {
    // The parser is a browser script that attaches itself to `window`;
    // simulate that environment in Node.
    global.window = {};
    require('../../../demo/js/net_spec.js');
    parseNetSpec = global.window.parseNetSpec;
    parseBrainSpec = global.window.parseBrainSpec;
  });

  it("should parse a valid spec with layer_defs and trainer", function() {
    var spec = parseNetSpec(JSON.stringify({
      layer_defs: [
        {type:'input', out_sx:1, out_sy:1, out_depth:2},
        {type:'fc', num_neurons:6, activation:'tanh'},
        {type:'softmax', num_classes:2}
      ],
      trainer: {learning_rate:0.01, momentum:0.1}
    }));

    expect(spec.layer_defs.length).toEqual(3);
    expect(spec.layer_defs[0].type).toEqual('input');
    expect(spec.layer_defs[1].num_neurons).toEqual(6);
    expect(spec.trainer.learning_rate).toEqual(0.01);
  });

  it("should ignore // and /* */ comments outside of strings", function() {
    var spec = parseNetSpec([
      '{',
      '  // line comment',
      '  /* block',
      '     comment */',
      '  "layer_defs": [',
      '    {"type":"input", "out_depth":2}, // trailing comment',
      '    {"type":"softmax", "num_classes":2}',
      '  ],',
      '  "trainer": {"learning_rate":0.01, "note":"http://example.com/x"}',
      '}'
    ].join('\n'));

    expect(spec.layer_defs.length).toEqual(2);
    expect(spec.trainer.learning_rate).toEqual(0.01);
    // a URL inside a string must not be treated as a comment
    expect(spec.trainer.note).toEqual('http://example.com/x');
  });

  it("should default the trainer to an empty object", function() {
    var spec = parseNetSpec(JSON.stringify({
      layer_defs: [
        {type:'input', out_depth:2},
        {type:'softmax', num_classes:2}
      ]
    }));
    expect(spec.trainer).toEqual({});
  });

  it("should reject empty input", function() {
    expect(function() { parseNetSpec(''); }).toThrow();
    expect(function() { parseNetSpec('   '); }).toThrow();
  });

  it("should reject malformed JSON", function() {
    expect(function() { parseNetSpec('{not json'); }).toThrow();
  });

  it("should reject specs without a layer_defs array", function() {
    expect(function() { parseNetSpec('{}'); }).toThrow();
    expect(function() { parseNetSpec('{"trainer":{}}'); }).toThrow();
  });

  it("should reject unknown layer types", function() {
    expect(function() {
      parseNetSpec(JSON.stringify({
        layer_defs: [
          {type:'input', out_depth:2},
          {type:'warp', num_neurons:5},
          {type:'softmax', num_classes:2}
        ]
      }));
    }).toThrow();
  });

  it("should reject layer defs missing required fields", function() {
    expect(function() {
      parseNetSpec(JSON.stringify({
        layer_defs: [
          {type:'input', out_depth:2},
          {type:'conv', sx:3}, // missing filters
          {type:'softmax', num_classes:2}
        ]
      }));
    }).toThrow();
  });

  it("should reject specs whose first layer is not an input layer", function() {
    expect(function() {
      parseNetSpec(JSON.stringify({
        layer_defs: [
          {type:'fc', num_neurons:5},
          {type:'softmax', num_classes:2}
        ]
      }));
    }).toThrow();
  });

  it("should reject a non-object trainer", function() {
    expect(function() {
      parseNetSpec(JSON.stringify({
        layer_defs: [
          {type:'input', out_depth:2},
          {type:'softmax', num_classes:2}
        ],
        trainer: 'sgd'
      }));
    }).toThrow();
  });

  it("should parse trainer_defs and legend (trainers demo)", function() {
    var spec = parseNetSpec(JSON.stringify({
      layer_defs: [
        {type:'input', out_sx:24, out_sy:24, out_depth:1},
        {type:'fc', num_neurons:20, activation:'relu'},
        {type:'softmax', num_classes:10}
      ],
      trainer_defs: [
        {learning_rate:0.01, method:'sgd', momentum:0.0, batch_size:8, l2_decay:0.001},
        {learning_rate:1.0, method:'adadelta', batch_size:8, l2_decay:0.001}
      ],
      legend: ['sgd', 'adadelta']
    }));

    expect(spec.trainer_defs.length).toEqual(2);
    expect(spec.trainer_defs[0].method).toEqual('sgd');
    expect(spec.trainer_defs[1].method).toEqual('adadelta');
    expect(spec.legend).toEqual(['sgd', 'adadelta']);
  });

  it("should reject unknown trainer methods", function() {
    expect(function() {
      parseNetSpec(JSON.stringify({
        layer_defs: [
          {type:'input', out_depth:2},
          {type:'softmax', num_classes:2}
        ],
        trainer: {method:'gradient-descent'}
      }));
    }).toThrow();
    expect(function() {
      parseNetSpec(JSON.stringify({
        layer_defs: [
          {type:'input', out_depth:2},
          {type:'softmax', num_classes:2}
        ],
        trainer_defs: [{method:'backprop'}]
      }));
    }).toThrow();
  });

  it("should reject a non-array trainer_defs", function() {
    expect(function() {
      parseNetSpec(JSON.stringify({
        layer_defs: [
          {type:'input', out_depth:2},
          {type:'softmax', num_classes:2}
        ],
        trainer_defs: 'sgd'
      }));
    }).toThrow();
  });

  it("should reject a non-string legend entry", function() {
    expect(function() {
      parseNetSpec(JSON.stringify({
        layer_defs: [
          {type:'input', out_depth:2},
          {type:'softmax', num_classes:2}
        ],
        legend: ['sgd', 42]
      }));
    }).toThrow();
  });
});

describe("parseBrainSpec (RL demo Q-learner parser)", function() {

  var parseBrainSpec;

  beforeAll(function() {
    // net_spec.js is a browser script that attaches itself to `window`; the
    // module is cached by require(), so only load it once and reuse the
    // window object created by the first describe block.
    if (typeof global.window === 'undefined' || typeof global.window.parseBrainSpec === 'undefined') {
      global.window = {};
      require('../../../demo/js/net_spec.js');
    }
    parseBrainSpec = global.window.parseBrainSpec;
  });

  var validSpec = {
    num_inputs: 27,
    num_actions: 5,
    temporal_window: 1,
    layer_defs: [
      {type:'input', out_sx:1, out_sy:1, out_depth:59},
      {type:'fc', num_neurons:50, activation:'relu'},
      {type:'regression', num_neurons:5}
    ],
    tdtrainer_options: {learning_rate:0.001, batch_size:64},
    opt: {gamma: 0.7, experience_size: 30000}
  };

  it("should parse a valid brain spec", function() {
    var spec = parseBrainSpec(JSON.stringify(validSpec));
    expect(spec.num_inputs).toEqual(27);
    expect(spec.num_actions).toEqual(5);
    expect(spec.temporal_window).toEqual(1);
    expect(spec.layer_defs.length).toEqual(3);
    expect(spec.layer_defs[2].type).toEqual('regression');
    expect(spec.tdtrainer_options.learning_rate).toEqual(0.001);
    expect(spec.opt.gamma).toEqual(0.7);
    expect(spec.opt.temporal_window).toEqual(1); // injected by the parser
  });

  it("should default temporal_window and tdtrainer_options", function() {
    var s = JSON.parse(JSON.stringify(validSpec));
    delete s.temporal_window;
    delete s.tdtrainer_options;
    var spec = parseBrainSpec(JSON.stringify(s));
    expect(spec.temporal_window).toEqual(1);
    expect(spec.tdtrainer_options).toEqual({});
  });

  it("should reject empty or malformed input", function() {
    expect(function() { parseBrainSpec(''); }).toThrow();
    expect(function() { parseBrainSpec('{nope'); }).toThrow();
  });

  it("should reject non-positive or non-integer num_inputs/num_actions", function() {
    var s = JSON.parse(JSON.stringify(validSpec));
    s.num_inputs = 0;
    expect(function() { parseBrainSpec(JSON.stringify(s)); }).toThrow();
    s.num_inputs = 27.5;
    expect(function() { parseBrainSpec(JSON.stringify(s)); }).toThrow();
    s.num_inputs = 27;
    s.num_actions = -1;
    expect(function() { parseBrainSpec(JSON.stringify(s)); }).toThrow();
  });

  it("should reject a spec whose last layer is not regression", function() {
    var s = JSON.parse(JSON.stringify(validSpec));
    s.layer_defs[2] = {type:'softmax', num_classes:5};
    expect(function() { parseBrainSpec(JSON.stringify(s)); }).toThrow();
  });

  it("should reject unknown layer types inside layer_defs", function() {
    var s = JSON.parse(JSON.stringify(validSpec));
    s.layer_defs[1] = {type:'warp', num_neurons:50};
    expect(function() { parseBrainSpec(JSON.stringify(s)); }).toThrow();
  });
});