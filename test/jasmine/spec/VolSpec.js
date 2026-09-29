describe("Vol (the basic data structure)", function() {

  it("should construct a 1D volume from an array", function() {
    var v = new convnetjs.Vol([0.5, -1.0, 2.0]);
    expect(v.sx).toEqual(1);
    expect(v.sy).toEqual(1);
    expect(v.depth).toEqual(3);
    expect(v.w.length).toEqual(3);
    expect(v.dw.length).toEqual(3);
    expect(v.w[0]).toEqual(0.5);
    expect(v.w[1]).toEqual(-1.0);
    expect(v.w[2]).toEqual(2.0);
  });

  it("should construct a 3D volume with random or constant values", function() {
    var v = new convnetjs.Vol(2, 3, 4);
    expect(v.sx).toEqual(2);
    expect(v.sy).toEqual(3);
    expect(v.depth).toEqual(4);
    expect(v.w.length).toEqual(24);

    var c = new convnetjs.Vol(2, 2, 2, 0.0);
    for (var i = 0; i < c.w.length; i++) {
      expect(c.w[i]).toEqual(0.0);
    }
  });

  it("should get, set and add values at (x, y, d)", function() {
    var v = new convnetjs.Vol(2, 2, 2, 0.0);
    v.set(1, 0, 1, 3.5);
    expect(v.get(1, 0, 1)).toEqual(3.5);
    v.add(1, 0, 1, 1.5);
    expect(v.get(1, 0, 1)).toEqual(5.0);
    v.set_grad(0, 1, 0, -2.0);
    expect(v.get_grad(0, 1, 0)).toEqual(-2.0);
    v.add_grad(0, 1, 0, 0.5);
    expect(v.get_grad(0, 1, 0)).toEqual(-1.5);
  });

  it("should clone and zero out", function() {
    var v = new convnetjs.Vol([1.0, 2.0, 3.0]);
    var z = v.cloneAndZero();
    expect(z.w.length).toEqual(3);
    for (var i = 0; i < z.w.length; i++) {
      expect(z.w[i]).toEqual(0.0);
    }
    // original must be untouched
    expect(v.w[0]).toEqual(1.0);
  });

  it("should clone values", function() {
    var v = new convnetjs.Vol([1.0, 2.0, 3.0]);
    var c = v.clone();
    expect(c.w[0]).toEqual(1.0);
    expect(c.w[1]).toEqual(2.0);
    expect(c.w[2]).toEqual(3.0);
    c.w[0] = 99.0;
    expect(v.w[0]).toEqual(1.0); // deep copy, not a reference
  });

  it("should add from another volume (optionally scaled)", function() {
    var a = new convnetjs.Vol([1.0, 2.0]);
    var b = new convnetjs.Vol([10.0, 20.0]);
    a.addFrom(b);
    expect(a.w[0]).toEqual(11.0);
    expect(a.w[1]).toEqual(22.0);

    var c = new convnetjs.Vol([1.0, 2.0]);
    c.addFromScaled(b, 0.5);
    expect(c.w[0]).toEqual(6.0);
    expect(c.w[1]).toEqual(12.0);
  });

  it("should set all values to a constant", function() {
    var v = new convnetjs.Vol([1.0, 2.0, 3.0]);
    v.setConst(7.0);
    for (var i = 0; i < v.w.length; i++) {
      expect(v.w[i]).toEqual(7.0);
    }
  });

  it("should round trip through JSON (without gradients)", function() {
    var v = new convnetjs.Vol(2, 2, 3, 0.0);
    for (var i = 0; i < v.w.length; i++) {
      v.w[i] = i * 0.5;
    }
    v.dw[0] = 123.0; // gradients are not serialized

    var v2 = new convnetjs.Vol(0, 0, 0, 0.0);
    v2.fromJSON(JSON.parse(JSON.stringify(v.toJSON())));

    expect(v2.sx).toEqual(2);
    expect(v2.sy).toEqual(2);
    expect(v2.depth).toEqual(3);
    for (var j = 0; j < v.w.length; j++) {
      expect(v2.w[j]).toEqual(v.w[j]);
    }
    expect(v2.dw[0]).toEqual(0.0); // gradients are not backed up
  });

  // ---------------------------------------------------------------------------
  // Vol.fromJSON is the boundary for untrusted model files: a corrupt or
  // hostile payload used to load "successfully" into an empty Vol (because
  // zeros(NaN) returns []) and then return garbage, or coerce non-numeric
  // values straight into the typed array. It must fail loudly instead.
  // ---------------------------------------------------------------------------

  it("should reject fromJSON with a non-object payload", function() {
    var v = new convnetjs.Vol(0, 0, 0, 0.0);
    expect(function() { v.fromJSON(null); }).toThrow();
    expect(function() { v.fromJSON(undefined); }).toThrow();
    expect(function() { v.fromJSON('nope'); }).toThrow();
    expect(function() { v.fromJSON(42); }).toThrow();
  });

  it("should reject fromJSON with missing or non-numeric dimensions", function() {
    var v = new convnetjs.Vol(0, 0, 0, 0.0);
    // missing depth -> previously sx*sy*undefined === NaN -> zeros(NaN) === []
    // which silently produced an empty Vol that "loaded" without error
    expect(function() { v.fromJSON({sx:2, sy:2, w:[1,2,3,4]}); }).toThrow();
    expect(function() { v.fromJSON({sy:2, depth:3, w:[1,2,3,4,5,6]}); }).toThrow();
    // NaN / Infinity / negative / fractional are all rejected
    expect(function() { v.fromJSON({sx:NaN, sy:2, depth:3, w:[1]}); }).toThrow();
    expect(function() { v.fromJSON({sx:Infinity, sy:2, depth:3, w:[1]}); }).toThrow();
    expect(function() { v.fromJSON({sx:-1, sy:2, depth:3, w:[1]}); }).toThrow();
    expect(function() { v.fromJSON({sx:1.5, sy:2, depth:3, w:[1]}); }).toThrow();
    expect(function() { v.fromJSON({sx:'2', sy:2, depth:3, w:[1]}); }).toThrow();
  });

  it("should reject an absurd volume size before allocating it", function() {
    var v = new convnetjs.Vol(0, 0, 0, 0.0);
    // 100000 x 100000 x 100000 would be a ~800TB Float64Array; without the guard
    // this is an out-of-memory kill of the host process.
    var huge = {sx:100000, sy:100000, depth:100000, w:{}};
    expect(function() { v.fromJSON(huge); }).toThrow();
    // dimension product overflow must be caught too
    expect(function() { v.fromJSON({sx:1e308, sy:1e308, depth:4, w:{}}); }).toThrow();
  });

  it("should reject a truncated or missing weight list", function() {
    var v = new convnetjs.Vol(0, 0, 0, 0.0);
    expect(function() { v.fromJSON({sx:2, sy:2, depth:3, w:[1,2,3]}); }).toThrow();
    expect(function() { v.fromJSON({sx:2, sy:2, depth:3, w:{}}); }).toThrow();
    expect(function() { v.fromJSON({sx:2, sy:2, depth:3}); }).toThrow();
  });

  it("should reject non-numeric weights rather than coercing them", function() {
    var v = new convnetjs.Vol(0, 0, 0, 0.0);
    expect(function() { v.fromJSON({sx:1, sy:1, depth:3, w:[1.0, 'two', 3.0]}); }).toThrow();
    expect(function() { v.fromJSON({sx:1, sy:1, depth:3, w:[1.0, null, 3.0]}); }).toThrow();
    expect(function() { v.fromJSON({sx:1, sy:1, depth:2, w:[{}, {}]}); }).toThrow();
    expect(function() { v.fromJSON({sx:1, sy:1, depth:2, w:[NaN, 1.0]}); }).toThrow();
  });

  it("should accept weights in the JSON-stringified typed array shape", function() {
    // JSON.stringify(Float64Array) yields {"0":..,"1":..} rather than a list,
    // which is the shape of every real persisted model file.
    var v = new convnetjs.Vol(0, 0, 0, 0.0);
    v.fromJSON({sx:1, sy:1, depth:3, w:{"0":0.5, "1":-0.5, "2":1.5}});
    expect(v.w.length).toEqual(3);
    expect(v.w[0]).toEqual(0.5);
    expect(v.w[1]).toEqual(-0.5);
    expect(v.w[2]).toEqual(1.5);
  });

  it("should still accept a plain numeric array", function() {
    var v = new convnetjs.Vol(0, 0, 0, 0.0);
    v.fromJSON({sx:1, sy:1, depth:3, w:[0.5, -0.5, 1.5]});
    expect(v.w.length).toEqual(3);
    expect(v.w[2]).toEqual(1.5);
  });
});

describe("Random number utilities", function() {

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

  it("should not blow the stack when Math.random is stubbed to a constant", function() {
    // Stubbing Math.random with a constant is a common way to get reproducible
    // training. The previous implementation recursed on rejected Box-Muller
    // pairs, so a constant RNG (r === 0 every time) exhausted the stack and
    // threw "RangeError: Maximum call stack size exceeded".
    Math.random = function() { return 0.5; };
    var g = convnetjs.randn(0, 1);
    expect(typeof g).toEqual('number');
    expect(isNaN(g)).toEqual(false);
  });

  it("should not blow the stack for an out-of-range stub", function() {
    // returns 1.0 -> u = v = 1 -> r = 2, rejected every time
    Math.random = function() { return 1.0; };
    expect(isNaN(convnetjs.randn(0, 1))).toEqual(false);
  });

  it("should not leak RNG state between independent networks", function() {
    // gaussRandom used to cache the spare Box-Muller deviate in module-level
    // state, so the cached value leaked from one network into the next and
    // reseeding did not make weight initialisation reproducible.
    Math.random = makeSeededRandom(1234);
    var v1 = new convnetjs.Vol(1, 1, 8, undefined);

    Math.random = makeSeededRandom(1234);
    var v2 = new convnetjs.Vol(1, 1, 8, undefined);

    for (var i = 0; i < 8; i++) {
      expect(v1.w[i]).toEqual(v2.w[i]);
    }
  });

  it("should produce reproducible weights for the same seed", function() {
    Math.random = makeSeededRandom(99);
    var a = new convnetjs.Vol(2, 2, 4, undefined);
    Math.random = makeSeededRandom(99);
    var b = new convnetjs.Vol(2, 2, 4, undefined);
    for (var i = 0; i < a.w.length; i++) {
      expect(a.w[i]).toEqual(b.w[i]);
    }
  });

  it("should produce a roughly standard normal distribution", function() {
    // test randn directly: Vol scales its init by sqrt(1/(sx*sy*depth)) to
    // equalize incoming fan-in, so the raw deviate stream is what we want here
    Math.random = makeSeededRandom(7);
    var n = 20000;
    var sum = 0.0, sumsq = 0.0;
    for (var i = 0; i < n; i++) {
      var g = convnetjs.randn(0, 1);
      sum += g;
      sumsq += g * g;
    }
    var mean = sum / n;
    var variance = sumsq / n - mean * mean;
    // loose but meaningful bounds: mean near 0, variance near 1
    expect(Math.abs(mean)).toBeLessThan(0.1);
    expect(Math.abs(variance - 1.0)).toBeLessThan(0.15);
  });

  it("should keep randf and randi inside their bounds", function() {
    Math.random = makeSeededRandom(5);
    for (var i = 0; i < 500; i++) {
      var f = convnetjs.randf(-2, 3);
      expect(f).toBeGreaterThanOrEqual(-2);
      expect(f).toBeLessThan(3);

      var n = convnetjs.randi(3, 9);
      expect(n).toBeGreaterThanOrEqual(3);
      expect(n).toBeLessThan(9);
    }
  });
});