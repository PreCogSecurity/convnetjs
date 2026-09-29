(function(global) {
  "use strict";

  // Random number utilities

  // Marsaglia polar method for standard normal deviates.
  //
  // This used to cache the second Box-Muller deviate in module-level state and
  // recurse to resample rejected pairs. Both were defects:
  //
  //  1. The recursion was unbounded. A rejected pair is not rare (p ~ 21.5%),
  //     but a degenerate or stubbed Math.random -- e.g. anyone seeding
  //     Math.random with a constant to make training reproducible, a mocked RNG,
  //     or a poor PRNG -- yields r == 0 or r > 1 *every* time, recursing until
  //     the engine threw "RangeError: Maximum call stack size exceeded". That
  //     is a crash-on-input hazard in a library that initialises every weight
  //     through this path.
  //  2. The module-level cache was global state, so the Box-Muller "spare"
  //     leaked across unrelated consumers: reseeding Math.random did not give
  //     you a reproducible network, and creating a second network perturbed the
  //     first one's weights. The cached spare also cost a branch on every draw
  //     in the weight-initialisation hot path.
  //
  // Sampling iteratively and returning one deviate per call removes the global
  // state (nets are now independent and seeding works) and removes the
  // unbounded recursion. The retry count is bounded so a degenerate RNG
  // degrades to a zero deviate rather than crashing the caller.
  var MAX_GAUSS_ATTEMPTS = 100;
  var gaussRandom = function() {
    for(var i=0;i<MAX_GAUSS_ATTEMPTS;i++) {
      var u = 2*Math.random()-1;
      var v = 2*Math.random()-1;
      var r = u*u + v*v;
      if(r > 0 && r <= 1) {
        return u * Math.sqrt(-2*Math.log(r)/r);
      }
    }
    // Degenerate RNG: every pair was rejected. Returning 0.0 keeps the caller
    // running with zero-initialised weights instead of throwing.
    return 0.0;
  }
  var randf = function(a, b) { return Math.random()*(b-a)+a; }
  var randi = function(a, b) { return Math.floor(Math.random()*(b-a)+a); }
  var randn = function(mu, std){ return mu+gaussRandom()*std; }

  // Array utilities
  var zeros = function(n) {
    if(typeof(n)==='undefined' || isNaN(n)) { return []; }
    if(typeof ArrayBuffer === 'undefined') {
      // lacking browser support
      var arr = new Array(n);
      for(var i=0;i<n;i++) { arr[i]= 0; }
      return arr;
    } else {
      return new Float64Array(n);
    }
  }

  var arrContains = function(arr, elt) {
    for(var i=0,n=arr.length;i<n;i++) {
      if(arr[i]===elt) return true;
    }
    return false;
  }

  var arrUnique = function(arr) {
    var b = [];
    for(var i=0,n=arr.length;i<n;i++) {
      if(!arrContains(b, arr[i])) {
        b.push(arr[i]);
      }
    }
    return b;
  }

  // return max and min of a given non-empty array.
  var maxmin = function(w) {
    if(w.length === 0) { return {}; } // ... ;s
    var maxv = w[0];
    var minv = w[0];
    var maxi = 0;
    var mini = 0;
    var n = w.length;
    for(var i=1;i<n;i++) {
      if(w[i] > maxv) { maxv = w[i]; maxi = i; } 
      if(w[i] < minv) { minv = w[i]; mini = i; } 
    }
    return {maxi: maxi, maxv: maxv, mini: mini, minv: minv, dv:maxv-minv};
  }

  // create random permutation of numbers, in range [0...n-1]
  var randperm = function(n) {
    var i = n,
        j = 0,
        temp;
    var array = [];
    for(var q=0;q<n;q++)array[q]=q;
    while (i--) {
        j = Math.floor(Math.random() * (i+1));
        temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
    return array;
  }

  // sample from list lst according to probabilities in list probs
  // the two lists are of same size, and probs adds up to 1
  var weightedSample = function(lst, probs) {
    var p = randf(0, 1.0);
    var cumprob = 0.0;
    for(var k=0,n=lst.length;k<n;k++) {
      cumprob += probs[k];
      if(p < cumprob) { return lst[k]; }
    }
  }

  // syntactic sugar function for getting default parameter values
  var getopt = function(opt, field_name, default_value) {
    if(typeof field_name === 'string') {
      // case of single string
      return (typeof opt[field_name] !== 'undefined') ? opt[field_name] : default_value;
    } else {
      // assume we are given a list of string instead
      var ret = default_value;
      for(var i=0;i<field_name.length;i++) {
        var f = field_name[i];
        if (typeof opt[f] !== 'undefined') {
          ret = opt[f]; // overwrite return value
        }
      }
      return ret;
    }
  }

  function assert(condition, message) {
    if (!condition) {
      message = message || "Assertion failed";
      if (typeof Error !== "undefined") {
        throw new Error(message);
      }
      throw message; // Fallback
    }
  }

  // ---- Deserialization guards -------------------------------------------------
  // Net/Vol fromJSON are routinely handed model files that came from a user
  // upload, a URL query string or a third party CDN, i.e. untrusted input. The
  // helpers below make sure malformed or hostile payloads fail loudly and
  // early instead of being silently coerced into a half-built network (which
  // then produces confidently wrong predictions) or into a huge allocation.

  // Upper bound on the number of floats a single Vol may allocate. A Vol is
  // backed by a Float64Array, so `sx * sy * depth` is fully attacker
  // controlled and a value like 1e9 asks for 8GB and takes the process down.
  // 1e8 entries is ~800MB, which is already far past any real filter tensor.
  var MAX_VOL_SIZE = 1e8;

  var isFiniteNumber = function(v) {
    return typeof v === 'number' && isFinite(v);
  }

  // non-negative safe integer check that rejects NaN, Infinity, negatives,
  // fractional and non-numeric values.
  var isNonNegInt = function(v) {
    return isFiniteNumber(v) && Math.floor(v) === v && v >= 0;
  }

  var isPlainish = function(v) {
    return v !== null && typeof v === 'object';
  }

  // Validates the dimensions of a Vol and returns the element count.
  var validateVolDims = function(sx, sy, depth, where) {
    assert(isNonNegInt(sx), 'Error! ' + where + ' sx must be a non-negative integer, got: ' + sx);
    assert(isNonNegInt(sy), 'Error! ' + where + ' sy must be a non-negative integer, got: ' + sy);
    assert(isNonNegInt(depth), 'Error! ' + where + ' depth must be a non-negative integer, got: ' + depth);
    var n = sx * sy * depth;
    assert(isNonNegInt(n), 'Error! ' + where + ' dimensions overflow: ' + sx + '*' + sy + '*' + depth);
    assert(n <= MAX_VOL_SIZE, 'Error! ' + where + ' is too large: ' + n +
      ' elements requested, the limit is ' + MAX_VOL_SIZE);
    return n;
  }

  global.randf = randf;
  global.randi = randi;
  global.randn = randn;
  global.zeros = zeros;
  global.maxmin = maxmin;
  global.randperm = randperm;
  global.weightedSample = weightedSample;
  global.arrUnique = arrUnique;
  global.arrContains = arrContains;
  global.getopt = getopt;
  global.assert = assert;
  global.MAX_VOL_SIZE = MAX_VOL_SIZE;
  global.isFiniteNumber = isFiniteNumber;
  global.isNonNegInt = isNonNegInt;
  global.isPlainish = isPlainish;
  global.validateVolDims = validateVolDims;
  
})(convnetjs);
