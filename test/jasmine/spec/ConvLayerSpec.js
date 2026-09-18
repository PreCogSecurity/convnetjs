describe("Convolutional and fully-connected layers (dot products)", function() {

  // Numeric gradient check helper. Uses the sum of absolute values as the
  // denominator so that analytic and numeric gradients that are both ~0 do
  // not produce a NaN relative error.
  function expectGradClose(analytic, numeric, tolerance) {
    var denom = Math.abs(analytic) + Math.abs(numeric);
    if (denom < 1e-12) {
      expect(Math.abs(analytic - numeric)).toBeLessThan(1e-6);
    } else {
      var rel_error = Math.abs(analytic - numeric) / denom;
      expect(rel_error).toBeLessThan(tolerance);
    }
  }

  // Runs a full forward/backward pass and numerically checks the gradient of
  // the loss with respect to the network input, mirroring the pattern used in
  // NeuralNetSpec.js. Note that we use net.backward() directly (and not
  // trainer.train()) so that the weights are not updated between the analytic
  // and the numeric evaluation.
  function checkDataGradient(layer_defs, gti) {
    var net = new convnetjs.Net();
    net.makeLayers(layer_defs);

    var x = new convnetjs.Vol(layer_defs[0].out_sx, layer_defs[0].out_sy, layer_defs[0].out_depth);
    net.forward(x, true);
    net.backward(gti); // computes gradients at all layers, and at x

    var delta = 0.000001;
    for (var i = 0; i < x.w.length; i++) {
      var grad_analytic = x.dw[i];

      var xold = x.w[i];
      x.w[i] += delta;
      var c0 = net.getCostLoss(x, gti);
      x.w[i] -= 2 * delta;
      var c1 = net.getCostLoss(x, gti);
      x.w[i] = xold; // reset

      var grad_numeric = (c0 - c1) / (2 * delta);
      expectGradClose(grad_analytic, grad_numeric, 1e-2);
    }
  }

  it("should produce correctly shaped outputs for a conv layer", function() {
    var layer_defs = [];
    layer_defs.push({type:'input', out_sx:8, out_sy:8, out_depth:3});
    layer_defs.push({type:'conv', sx:3, filters:4, stride:1, pad:1, activation:'tanh'});
    layer_defs.push({type:'softmax', num_classes:4});

    var net = new convnetjs.Net();
    net.makeLayers(layer_defs);

    var conv = net.layers[1];
    expect(conv.layer_type).toEqual('conv');
    expect(conv.out_sx).toEqual(8); // pad 1 keeps spatial size the same
    expect(conv.out_sy).toEqual(8);
    expect(conv.out_depth).toEqual(4);
    expect(conv.filters.length).toEqual(4);
    expect(conv.filters[0].sx).toEqual(3);
    expect(conv.filters[0].sy).toEqual(3);
    expect(conv.filters[0].depth).toEqual(3);

    var x = new convnetjs.Vol(8, 8, 3);
    var out = net.forward(x);
    expect(out.w.length).toEqual(4); // softmax output
    var conv_act = net.layers[1].out_act;
    expect(conv_act.sx).toEqual(8);
    expect(conv_act.sy).toEqual(8);
    expect(conv_act.depth).toEqual(4);
  });

  it("should produce correctly shaped outputs for a strided conv + pool", function() {
    var layer_defs = [];
    layer_defs.push({type:'input', out_sx:16, out_sy:16, out_depth:3});
    layer_defs.push({type:'conv', sx:5, filters:8, stride:2, pad:2, activation:'tanh'});
    layer_defs.push({type:'pool', sx:2, stride:2});
    layer_defs.push({type:'softmax', num_classes:3});

    var net = new convnetjs.Net();
    net.makeLayers(layer_defs);

    // desugared layers: input, conv, tanh, pool, fc, softmax
    // conv with stride 2 and pad 2: floor((16 + 4 - 5)/2 + 1) = floor(8.5) = 8
    expect(net.layers[1].out_sx).toEqual(8);
    expect(net.layers[1].out_sy).toEqual(8);
    expect(net.layers[1].out_depth).toEqual(8);
    // pool with sx 2, stride 2: floor((8 - 2)/2 + 1) = 4
    expect(net.layers[3].layer_type).toEqual('pool');
    expect(net.layers[3].out_sx).toEqual(4);
    expect(net.layers[3].out_sy).toEqual(4);
    expect(net.layers[3].out_depth).toEqual(8);

    var x = new convnetjs.Vol(16, 16, 3);
    var out = net.forward(x);
    expect(out.w.length).toEqual(3); // softmax output
  });

  it("should compute correct gradient at data for a conv net", function() {
    var layer_defs = [];
    layer_defs.push({type:'input', out_sx:4, out_sy:4, out_depth:2});
    layer_defs.push({type:'conv', sx:3, filters:2, stride:1, pad:0, activation:'tanh'});
    layer_defs.push({type:'softmax', num_classes:2});
    checkDataGradient(layer_defs, 1);
  });

  it("should compute correct gradient at data for a conv net with pooling", function() {
    var layer_defs = [];
    layer_defs.push({type:'input', out_sx:6, out_sy:6, out_depth:2});
    layer_defs.push({type:'conv', sx:3, filters:2, stride:1, pad:0, activation:'tanh'});
    layer_defs.push({type:'pool', sx:2, stride:2});
    layer_defs.push({type:'softmax', num_classes:2});
    checkDataGradient(layer_defs, 0);
  });

  it("should produce correctly shaped outputs for a fully connected layer", function() {
    var layer_defs = [];
    layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:4});
    layer_defs.push({type:'fc', num_neurons:7, activation:'tanh'});
    layer_defs.push({type:'softmax', num_classes:3});

    var net = new convnetjs.Net();
    net.makeLayers(layer_defs);

    var fc = net.layers[1];
    expect(fc.layer_type).toEqual('fc');
    expect(fc.out_depth).toEqual(7);
    expect(fc.num_inputs).toEqual(4);
    expect(fc.filters.length).toEqual(7);
    expect(fc.filters[0].w.length).toEqual(4);

    var x = new convnetjs.Vol([0.1, -0.2, 0.3, -0.4]);
    var out = net.forward(x);
    expect(out.w.length).toEqual(3);
  });

  it("should compute correct gradient at data for a fully connected net", function() {
    var layer_defs = [];
    layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:3});
    layer_defs.push({type:'fc', num_neurons:5, activation:'tanh'});
    layer_defs.push({type:'fc', num_neurons:5, activation:'tanh'});
    layer_defs.push({type:'softmax', num_classes:3});
    checkDataGradient(layer_defs, 2);
  });

  it("should serialize and deserialize conv and fc layers", function() {
    var layer_defs = [];
    layer_defs.push({type:'input', out_sx:4, out_sy:4, out_depth:2});
    layer_defs.push({type:'conv', sx:3, filters:2, stride:1, pad:0, activation:'tanh'});
    layer_defs.push({type:'fc', num_neurons:5, activation:'tanh'});
    layer_defs.push({type:'softmax', num_classes:2});

    var net = new convnetjs.Net();
    net.makeLayers(layer_defs);

    var json = JSON.parse(JSON.stringify(net.toJSON()));
    var net2 = new convnetjs.Net();
    net2.fromJSON(json);

    expect(net2.layers.length).toEqual(net.layers.length);
    expect(net2.layers[1].layer_type).toEqual('conv');
    expect(net2.layers[1].out_depth).toEqual(2);
    expect(net2.layers[3].layer_type).toEqual('fc');
    expect(net2.layers[3].out_depth).toEqual(5);

    // forward passes should agree after a round trip
    var x = new convnetjs.Vol(4, 4, 2);
    var p1 = net.forward(x);
    var p2 = net2.forward(x);
    for (var i = 0; i < p1.w.length; i++) {
      expect(p1.w[i]).toEqual(p2.w[i]);
    }
  });
});