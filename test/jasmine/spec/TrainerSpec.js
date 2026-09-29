describe("Trainers (sgd / adagrad / adadelta / adam / nesterov)", function() {

  var originalRandom;

  beforeEach(function() {
    originalRandom = Math.random;
  });

  afterEach(function() {
    Math.random = originalRandom;
  });

  // Deterministic PRNG so that training runs are reproducible. This is a
  // simple 32-bit LCG; it is only used to seed the data and network
  // initialization inside the specs.
  function makeSeededRandom(seed) {
    var s = seed >>> 0;
    return function() {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 4294967296;
    };
  }

  function makeNet() {
    var layer_defs = [];
    layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});
    layer_defs.push({type:'fc', num_neurons:10, activation:'tanh'});
    layer_defs.push({type:'softmax', num_classes:2});
    var net = new convnetjs.Net();
    net.makeLayers(layer_defs);
    return net;
  }

  // Linearly separable data: class 1 iff x[0] > 0.
  function makeData(n) {
    var data = [];
    var labels = [];
    for (var i = 0; i < n; i++) {
      var x0 = Math.random() * 2 - 1;
      var x1 = Math.random() * 2 - 1;
      data.push(new convnetjs.Vol([x0, x1]));
      labels.push(x0 > 0 ? 1 : 0);
    }
    return {data: data, labels: labels};
  }

  function trainFor(method, extraOpts, iters) {
    Math.random = makeSeededRandom(42); // fixed seed for reproducibility
    var net = makeNet();
    var options = {
      method: method,
      learning_rate: 0.1,
      batch_size: 1,
      l2_decay: 0.0,
      l1_decay: 0.0
    };
    for (var k in extraOpts) {
      if (Object.prototype.hasOwnProperty.call(extraOpts, k)) {
        options[k] = extraOpts[k];
      }
    }
    var trainer = new convnetjs.Trainer(net, options);
    var ds = makeData(20);

    // Compare the mean loss over a leading window against a trailing window
    // rather than a single first and last sample. This is single-example
    // (batch_size 1) training, so any individual step is noisy -- especially
    // for the adaptive methods -- and a two-point comparison passed or failed
    // on the order in which specs happened to consume the RNG stream.
    var window = Math.max(1, Math.floor(iters / 10));
    var head = 0.0;
    var tail = 0.0;
    var losses = [];
    for (var i = 0; i < iters; i++) {
      var ix = i % ds.data.length;
      var stats = trainer.train(ds.data[ix], ds.labels[ix]);
      losses.push(stats.cost_loss);
    }
    for (var a = 0; a < window; a++) {
      head += losses[a];
      tail += losses[iters - window + a];
    }
    return {first_loss: losses[0], last_loss: losses[iters - 1],
            head_loss: head / window, tail_loss: tail / window};
  }

  it("should decrease loss with vanilla SGD", function() {
    var r = trainFor('sgd', {momentum: 0.0}, 200);
    expect(r.tail_loss).toBeLessThan(r.head_loss);
  });

  it("should decrease loss with SGD + momentum", function() {
    var r = trainFor('sgd', {momentum: 0.9}, 200);
    expect(r.tail_loss).toBeLessThan(r.head_loss);
  });

  it("should decrease loss with adagrad", function() {
    var r = trainFor('adagrad', {}, 200);
    expect(r.tail_loss).toBeLessThan(r.head_loss);
  });

  it("should decrease loss with adadelta", function() {
    var r = trainFor('adadelta', {learning_rate: 1.0}, 200);
    expect(r.tail_loss).toBeLessThan(r.head_loss);
  });

  it("should decrease loss with adam", function() {
    var r = trainFor('adam', {}, 200);
    expect(r.tail_loss).toBeLessThan(r.head_loss);
  });

  it("should decrease loss with nesterov", function() {
    var r = trainFor('nesterov', {momentum: 0.9}, 200);
    expect(r.tail_loss).toBeLessThan(r.head_loss);
  });

  it("should report training statistics", function() {
    Math.random = makeSeededRandom(7);
    var net = makeNet();
    var trainer = new convnetjs.SGDTrainer(net, {learning_rate: 0.1, batch_size: 1});
    var ds = makeData(5);

    var stats = trainer.train(ds.data[0], ds.labels[0]);
    expect(stats.cost_loss).toBeGreaterThan(0);
    expect(stats.loss).toBeGreaterThan(0);
    expect(typeof stats.fwd_time).toEqual('number');
    expect(typeof stats.bwd_time).toEqual('number');
    expect(stats.softmax_loss).toEqual(stats.cost_loss); // backwards compatibility
  });

  it("should accumulate gradients over a batch before updating", function() {
    Math.random = makeSeededRandom(3);
    var net = makeNet();
    var trainer = new convnetjs.Trainer(net, {method:'sgd', learning_rate: 0.1, batch_size: 4});
    var ds = makeData(8);

    // snapshot the weights before any update
    var pglist = net.getParamsAndGrads();
    var before = [];
    for (var i = 0; i < pglist.length; i++) {
      before.push(Array.prototype.slice.call(pglist[i].params));
    }

    // 3 of 4 examples in the batch: no update should have happened yet
    for (var k = 0; k < 3; k++) {
      trainer.train(ds.data[k], ds.labels[k]);
    }
    var pglist2 = net.getParamsAndGrads();
    for (var j = 0; j < pglist2.length; j++) {
      var p = pglist2[j].params;
      for (var q = 0; q < p.length; q++) {
        expect(p[q]).toEqual(before[j][q]);
      }
    }

    // 4th example completes the batch: weights must have moved
    trainer.train(ds.data[3], ds.labels[3]);
    var pglist3 = net.getParamsAndGrads();
    var moved = false;
    for (var m = 0; m < pglist3.length; m++) {
      var p3 = pglist3[m].params;
      for (var n = 0; n < p3.length; n++) {
        if (p3[n] !== before[m][n]) { moved = true; }
      }
    }
    expect(moved).toBe(true);
  });
});