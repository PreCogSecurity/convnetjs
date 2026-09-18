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
});