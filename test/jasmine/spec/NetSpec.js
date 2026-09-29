describe("Net (layer definition validation and serialization)", function() {

  it("should throw when layer definitions are not an array", function() {
    var net = new convnetjs.Net();
    expect(function() { net.makeLayers('nope'); }).toThrow();
    expect(function() { net.makeLayers({}); }).toThrow();
  });

  it("should throw when fewer than two layers are provided", function() {
    var net = new convnetjs.Net();
    expect(function() { net.makeLayers([{type:'input', out_depth:2}]); }).toThrow();
  });

  it("should throw when the first layer is not an input layer", function() {
    var net = new convnetjs.Net();
    expect(function() {
      net.makeLayers([{type:'fc', num_neurons:5}, {type:'softmax', num_classes:2}]);
    }).toThrow();
  });

  it("should throw on unrecognized layer types", function() {
    var net = new convnetjs.Net();
    expect(function() {
      net.makeLayers([
        {type:'input', out_depth:2},
        {type:'warp', num_neurons:5},
        {type:'softmax', num_classes:2}
      ]);
    }).toThrow();
  });

  it("should throw on unsupported activations", function() {
    var net = new convnetjs.Net();
    expect(function() {
      net.makeLayers([
        {type:'input', out_depth:2},
        {type:'fc', num_neurons:5, activation:'gelu'},
        {type:'softmax', num_classes:2}
      ]);
    }).toThrow();
  });

  it("should desugar activations into separate layers", function() {
    var layer_defs = [];
    layer_defs.push({type:'input', out_depth:2});
    layer_defs.push({type:'fc', num_neurons:5, activation:'relu'});
    layer_defs.push({type:'fc', num_neurons:5, activation:'tanh'});
    layer_defs.push({type:'softmax', num_classes:3});

    var net = new convnetjs.Net();
    net.makeLayers(layer_defs);

    // input, fc, relu, fc, tanh, fc(softmax), softmax
    expect(net.layers.length).toEqual(7);
    expect(net.layers[2].layer_type).toEqual('relu');
    expect(net.layers[4].layer_type).toEqual('tanh');
    expect(net.layers[5].layer_type).toEqual('fc');
    expect(net.layers[6].layer_type).toEqual('softmax');
  });

  it("should return the argmax prediction for a softmax net", function() {
    var layer_defs = [];
    layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});
    layer_defs.push({type:'fc', num_neurons:5, activation:'tanh'});
    layer_defs.push({type:'softmax', num_classes:3});

    var net = new convnetjs.Net();
    net.makeLayers(layer_defs);

    var x = new convnetjs.Vol([0.5, -0.5]);
    net.forward(x);
    var pred = net.getPrediction();
    expect(pred).toBeGreaterThanOrEqual(0);
    expect(pred).toBeLessThan(3);
  });

  it("should round trip through JSON", function() {
    var layer_defs = [];
    layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});
    layer_defs.push({type:'fc', num_neurons:5, activation:'tanh'});
    layer_defs.push({type:'softmax', num_classes:3});

    var net = new convnetjs.Net();
    net.makeLayers(layer_defs);

    var json = JSON.parse(JSON.stringify(net.toJSON()));
    var net2 = new convnetjs.Net();
    net2.fromJSON(json);

    expect(net2.layers.length).toEqual(net.layers.length);
    var x = new convnetjs.Vol([0.3, -0.7]);
    var p1 = net.forward(x);
    var p2 = net2.forward(x);
    for (var i = 0; i < p1.w.length; i++) {
      expect(p1.w[i]).toEqual(p2.w[i]);
    }
  });

  // ---------------------------------------------------------------------------
  // Deserialization is the untrusted-input boundary: model files routinely come
  // from a user upload or a remote URL. A rejected payload must fail loudly
  // and must not leave a half-built network behind.
  // ---------------------------------------------------------------------------

  it("should reject a malformed envelope in fromJSON", function() {
    var net = new convnetjs.Net();
    expect(function() { net.fromJSON(null); }).toThrow();
    expect(function() { net.fromJSON(undefined); }).toThrow();
    expect(function() { net.fromJSON('nope'); }).toThrow();
    expect(function() { net.fromJSON({}); }).toThrow();
    expect(function() { net.fromJSON({layers: 'not an array'}); }).toThrow();
    expect(function() { net.fromJSON({layers: [null]}); }).toThrow();
  });

  it("should reject an unknown layer type with a clear error", function() {
    var net = new convnetjs.Net();
    // previously this produced an opaque
    // "Cannot read properties of undefined (reading 'fromJSON')" TypeError
    var err = null;
    try {
      net.fromJSON({layers: [{layer_type: 'warp'}]});
    } catch (e) {
      err = e;
    }
    expect(err).not.toEqual(null);
    expect(err.message).toContain('warp');
  });

  it("should not resolve a layer constructor through the prototype chain", function() {
    var net = new convnetjs.Net();
    // a naive `known[t]` lookup would find Object.prototype.constructor here
    // and happily instantiate an Object as if it were a layer
    expect(function() {
      net.fromJSON({layers: [{layer_type: 'constructor'}]});
    }).toThrow();
    expect(function() {
      net.fromJSON({layers: [{layer_type: 'toString'}]});
    }).toThrow();
    expect(function() {
      net.fromJSON({layers: [{layer_type: '__proto__'}]});
    }).toThrow();
  });

  it("should leave the existing network intact when a load fails midway", function() {
    var layer_defs = [];
    layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});
    layer_defs.push({type:'fc', num_neurons:5, activation:'tanh'});
    layer_defs.push({type:'softmax', num_classes:3});

    var net = new convnetjs.Net();
    net.makeLayers(layer_defs);
    var good = JSON.parse(JSON.stringify(net.toJSON()));

    // second layer is corrupt, so the load must abort...
    var bad = JSON.parse(JSON.stringify(good));
    bad.layers[1].biases.w = {'0': 'not a number'};
    expect(function() { net.fromJSON(bad); }).toThrow();

    // ...and must not have left a truncated network behind. The defs desugar to
    // input, fc, tanh, fc, softmax.
    expect(net.layers.length).toEqual(5);
    var x = new convnetjs.Vol([0.3, -0.7]);
    expect(net.forward(x).w.length).toEqual(3);
  });

  it("should reject an invalid maxout group_size", function() {
    var net = new convnetjs.Net();
    // 0 would make out_depth Infinity, a negative or fractional value makes it
    // a non-integer; both previously escaped the check entirely
    [0, -2, 1.5, null, 'two'].forEach(function(gs) {
      expect(function() {
        net.makeLayers([
          {type:'input', out_sx:1, out_sy:1, out_depth:4},
          {type:'fc', num_neurons:4, activation:'maxout', group_size: gs},
          {type:'softmax', num_classes:2}
        ]);
      }).toThrow();
    });
  });

  it("should default maxout group_size to 2 and honour an explicit value", function() {
    var net = new convnetjs.Net();
    net.makeLayers([
      {type:'input', out_sx:1, out_sy:1, out_depth:4},
      {type:'fc', num_neurons:4, activation:'maxout'},
      {type:'softmax', num_classes:2}
    ]);
    var maxout = net.layers[2];
    expect(maxout.layer_type).toEqual('maxout');
    expect(maxout.group_size).toEqual(2);
    expect(maxout.out_depth).toEqual(2);

    var net2 = new convnetjs.Net();
    net2.makeLayers([
      {type:'input', out_sx:1, out_sy:1, out_depth:4},
      {type:'fc', num_neurons:4, activation:'maxout', group_size:4},
      {type:'softmax', num_classes:2}
    ]);
    expect(net2.layers[2].group_size).toEqual(4);
    expect(net2.layers[2].out_depth).toEqual(1);
  });

  it("should size switches correctly for a maxout layer restored from JSON", function() {
    // the switch table holds one entry per *output* activation; it used to be
    // allocated with group_size instead, so a restored maxout net poisoned
    // its parameter gradients with NaN on the first training step.
    var net = new convnetjs.Net();
    net.makeLayers([
      {type:'input', out_sx:1, out_sy:1, out_depth:4},
      {type:'fc', num_neurons:4, activation:'maxout', group_size:2},
      {type:'softmax', num_classes:2}
    ]);

    var net2 = new convnetjs.Net();
    net2.fromJSON(JSON.parse(JSON.stringify(net.toJSON())));
    var maxout = net2.layers[2];
    expect(maxout.switches.length).toEqual(
      maxout.out_sx * maxout.out_sy * maxout.out_depth);

    // every switch must be a usable in-range index
    for (var i = 0; i < maxout.switches.length; i++) {
      expect(typeof maxout.switches[i]).toEqual('number');
      expect(isNaN(maxout.switches[i])).toEqual(false);
    }
  });

  it("should train a maxout net after a JSON round trip without NaN gradients", function() {
    var layer_defs = [
      {type:'input', out_sx:1, out_sy:1, out_depth:4},
      {type:'fc', num_neurons:4, activation:'maxout', group_size:2},
      {type:'softmax', num_classes:2}
    ];
    var net = new convnetjs.Net();
    net.makeLayers(layer_defs);

    var net2 = new convnetjs.Net();
    net2.fromJSON(JSON.parse(JSON.stringify(net.toJSON())));

    var trainer = new convnetjs.Trainer(net2, {learning_rate:0.01, l2_decay:0.0});
    var x = new convnetjs.Vol([1, 0, 0, 0]);
    for (var s = 0; s < 5; s++) { trainer.train(x, 1); }

    var params = net2.getParamsAndGrads();
    for (var i = 0; i < params.length; i++) {
      for (var j = 0; j < params[i].grads.length; j++) {
        expect(isNaN(params[i].grads[j])).toEqual(false);
      }
    }
  });

  it("should reject a maxout group_size restored from JSON", function() {
    var net = new convnetjs.Net();
    expect(function() {
      net.fromJSON({layers: [{
        layer_type: 'maxout', out_sx:1, out_sy:1, out_depth:0, group_size: 0
      }]});
    }).toThrow();
  });
});