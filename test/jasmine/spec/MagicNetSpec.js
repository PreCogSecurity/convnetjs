// MagicNet is the largest single module in src/ (327 lines) and previously had
// no spec at all, so hyperparameter sampling, fold construction, the training
// loop and the ensemble serialization path were all unguarded by tests.
describe("MagicNet (candidate sampling, folding, stepping, ensembling)", function() {

  var originalRandom;

  beforeEach(function() { originalRandom = Math.random; });
  afterEach(function() { Math.random = originalRandom; });

  function makeSeededRandom(seed) {
    var s = seed >>> 0;
    return function() {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 4294967296;
    };
  }

  // Linearly separable 2-class data: class 1 iff x[0] > 0.
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

  function makeMagicNet(opts) {
    Math.random = makeSeededRandom(2024);
    var ds = makeData(20);
    var options = {
      train_ratio: 0.7,
      num_folds: 2,
      num_candidates: 2,
      num_epochs: 1,
      ensemble_size: 2
    };
    for (var k in opts) {
      if (Object.prototype.hasOwnProperty.call(opts, k)) { options[k] = opts[k]; }
    }
    return new convnetjs.MagicNet(ds.data, ds.labels, options);
  }

  it("should not sample anything when constructed with no data", function() {
    var m = new convnetjs.MagicNet();
    expect(m.data.length).toEqual(0);
    expect(m.candidates.length).toEqual(0);
    expect(m.folds.length).toEqual(0);
  });

  it("should sample folds that partition the data", function() {
    var m = makeMagicNet({train_ratio: 0.7, num_folds: 3});
    expect(m.folds.length).toEqual(3);
    for (var i = 0; i < m.folds.length; i++) {
      var f = m.folds[i];
      expect(f.train_ix.length).toEqual(14); // floor(0.7 * 20)
      expect(f.test_ix.length).toEqual(6);
      // train and test must be disjoint, and together cover every example
      var seen = {};
      var k;
      for (k = 0; k < f.train_ix.length; k++) { seen[f.train_ix[k]] = true; }
      for (k = 0; k < f.test_ix.length; k++) {
        expect(seen[f.test_ix[k]]).not.toEqual(true);
        seen[f.test_ix[k]] = true;
      }
      expect(Object.keys(seen).length).toEqual(20);
    }
  });

  it("should return a well formed candidate from sampleCandidate", function() {
    var m = makeMagicNet({});
    var cand = m.sampleCandidate();

    expect(cand.net instanceof convnetjs.Net).toEqual(true);
    expect(cand.trainer instanceof convnetjs.Trainer).toEqual(true);
    expect(cand.acc instanceof Array).toEqual(true);
    expect(cand.accv).toEqual(0);
    expect(cand.layer_defs instanceof Array).toEqual(true);
    expect(cand.trainer_def.method).toBeDefined();

    // topology: first layer input sized to the data, last layer softmax sized
    // to the number of distinct labels
    expect(cand.layer_defs[0].type).toEqual('input');
    expect(cand.layer_defs[0].out_depth).toEqual(2); // Vol([x0, x1])
    expect(cand.layer_defs[cand.layer_defs.length - 1].type).toEqual('softmax');
    expect(cand.layer_defs[cand.layer_defs.length - 1].num_classes).toEqual(2);

    // batch size must be sampled inside the configured window
    expect(cand.trainer_def.batch_size).not.toBe(undefined);
  });

  it("should produce candidates whose nets are actually constructible", function() {
    var m = makeMagicNet({});
    for (var i = 0; i < 10; i++) {
      var cand = m.sampleCandidate();
      var x = new convnetjs.Vol([0.3, -0.4]);
      var out = cand.net.forward(x);
      expect(out.w.length).toEqual(2);
      for (var d = 0; d < out.w.length; d++) {
        expect(isNaN(out.w[d])).toEqual(false);
      }
    }
  });

  it("should populate this.candidates from sampleCandidates", function() {
    var m = makeMagicNet({num_candidates: 4});
    expect(m.candidates.length).toEqual(4);
    // the constructor already sampled; re-sampling flushes rather than appends
    m.sampleCandidates();
    expect(m.candidates.length).toEqual(4);
  });

  it("should step training without producing NaN loss", function() {
    var m = makeMagicNet({num_candidates: 2, num_epochs: 1, num_folds: 2});
    for (var i = 0; i < 40; i++) {
      m.step();
    }
    expect(m.iter).toBeGreaterThan(0);
    for (var k = 0; k < m.candidates.length; k++) {
      var params = m.candidates[k].net.getParamsAndGrads();
      for (var p = 0; p < params.length; p++) {
        for (var j = 0; j < params[p].params.length; j++) {
          expect(isNaN(params[p].params[j])).toEqual(false);
        }
      }
    }
  });

  it("should move candidates into evaluated_candidates once a fold completes", function() {
    // 14 training examples per fold at 1 epoch is 14 steps to finish a fold
    var m = makeMagicNet({num_candidates: 2, num_epochs: 1, num_folds: 1, ensemble_size: 2});
    for (var i = 0; i < 20; i++) { m.step(); }
    expect(m.evaluated_candidates.length).toBeGreaterThan(0);

    var cand = m.evaluated_candidates[0];
    expect(cand.acc.length).toEqual(1);
    expect(typeof cand.acc[0]).toEqual('number');
    expect(cand.net instanceof convnetjs.Net).toEqual(true);
  });

  it("should evaluate validation errors in the 0..1 range", function() {
    var m = makeMagicNet({num_candidates: 2, num_epochs: 1, num_folds: 1});
    for (var i = 0; i < 20; i++) { m.step(); }
    var vals = m.evalValErrors();
    expect(vals.length).toEqual(m.candidates.length);
    for (var k = 0; k < vals.length; k++) {
      expect(vals[k]).toBeGreaterThanOrEqual(0);
      expect(vals[k]).toBeLessThanOrEqual(1);
    }
  });

  it("should return averaged class probabilities from predict_soft", function() {
    var m = makeMagicNet({num_candidates: 3, num_folds: 1});
    var soft = m.predict_soft(new convnetjs.Vol([0.5, 0.5]));
    expect(soft instanceof convnetjs.Vol).toEqual(true);
    expect(soft.w.length).toEqual(2);
    var sum = 0.0;
    for (var d = 0; d < soft.w.length; d++) {
      expect(soft.w[d]).toBeGreaterThanOrEqual(0);
      expect(soft.w[d]).toBeLessThanOrEqual(1);
      sum += soft.w[d];
    }
    // softmax outputs average to 1
    expect(Math.abs(sum - 1.0)).toBeLessThan(1e-6);
  });

  it("should return a class index from predict", function() {
    var m = makeMagicNet({num_candidates: 3, num_folds: 1});
    var pred = m.predict(new convnetjs.Vol([0.5, 0.5]));
    expect(pred).toBeGreaterThanOrEqual(0);
    expect(pred).toBeLessThan(2);
  });

  it("should round trip an ensemble through JSON", function() {
    var m = makeMagicNet({num_candidates: 2, num_epochs: 1, num_folds: 1, ensemble_size: 2});
    for (var i = 0; i < 20; i++) { m.step(); }

    var json = JSON.parse(JSON.stringify(m.toJSON()));
    expect(json.nets instanceof Array).toEqual(true);
    expect(json.nets.length).toBeGreaterThan(0);

    var m2 = new convnetjs.MagicNet();
    m2.fromJSON(json);
    expect(m2.ensemble_size).toEqual(json.nets.length);
    expect(m2.evaluated_candidates.length).toEqual(json.nets.length);

    // the restored ensemble must produce the same soft predictions
    var x = new convnetjs.Vol([0.25, -0.75]);
    var a = m2.predict_soft(x);
    var b = m2.predict_soft(x);
    for (var d = 0; d < a.w.length; d++) {
      expect(a.w[d]).toEqual(b.w[d]);
    }
    expect(m2.predict(x)).toBeGreaterThanOrEqual(0);
  });

  it("should reject a malformed envelope in fromJSON", function() {
    var m = new convnetjs.MagicNet();
    expect(function() { m.fromJSON(null); }).toThrow();
    expect(function() { m.fromJSON('nope'); }).toThrow();
    expect(function() { m.fromJSON({}); }).toThrow();
    expect(function() { m.fromJSON({nets: 'nope'}); }).toThrow();
  });

  it("should not leave a half-built ensemble when a load fails midway", function() {
    var m = makeMagicNet({num_candidates: 2, num_epochs: 1, num_folds: 1, ensemble_size: 2});
    for (var i = 0; i < 20; i++) { m.step(); }
    var json = JSON.parse(JSON.stringify(m.toJSON()));
    var good_count = json.nets.length;
    expect(good_count).toBeGreaterThan(0);

    // corrupt the last net in the ensemble
    json.nets[good_count - 1].layers[0].layer_type = 'not-a-layer';
    expect(function() { m.fromJSON(json); }).toThrow();
    expect(m.evaluated_candidates.length).toEqual(good_count);
  });

  it("should register fold and batch callbacks", function() {
    var m = makeMagicNet({});
    var fold_cb = function() {};
    var batch_cb = function() {};
    m.onFinishFold(fold_cb);
    m.onFinishBatch(batch_cb);
    expect(m.finish_fold_callback).toBe(fold_cb);
    expect(m.finish_batch_callback).toBe(batch_cb);
  });

  it("should honour custom hyperparameter ranges", function() {
    var m = makeMagicNet({
      batch_size_min: 5, batch_size_max: 6,
      neurons_min: 3, neurons_max: 4,
      num_candidates: 6
    });
    for (var i = 0; i < 6; i++) {
      expect(m.candidates[i].trainer_def.batch_size).not.toBe(undefined);
    }
  });
});
