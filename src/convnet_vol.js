(function(global) {
  "use strict";

  // Vol is the basic building block of all data in a net.
  // it is essentially just a 3D volume of numbers, with a
  // width (sx), height (sy), and depth (depth).
  // it is used to hold data for all filters, all volumes,
  // all weights, and also stores all gradients w.r.t. 
  // the data. c is optionally a value to initialize the volume
  // with. If c is missing, fills the Vol with random numbers.
  var Vol = function(sx, sy, depth, c) {
    // this is how you check if a variable is an array. Oh, Javascript :)
    if(Object.prototype.toString.call(sx) === '[object Array]') {
      // we were given a list in sx, assume 1D volume and fill it up
      this.sx = 1;
      this.sy = 1;
      this.depth = sx.length;
      // we have to do the following copy because we want to use
      // fast typed arrays, not an ordinary javascript array
      this.w = global.zeros(this.depth);
      this.dw = global.zeros(this.depth);
      for(var i=0;i<this.depth;i++) {
        this.w[i] = sx[i];
      }
    } else {
      // we were given dimensions of the vol
      this.sx = sx;
      this.sy = sy;
      this.depth = depth;
      var n = sx*sy*depth;
      this.w = global.zeros(n);
      this.dw = global.zeros(n);
      if(typeof c === 'undefined') {
        // weight normalization is done to equalize the output
        // variance of every neuron, otherwise neurons with a lot
        // of incoming connections have outputs of larger variance
        var scale = Math.sqrt(1.0/(sx*sy*depth));
        for(var i=0;i<n;i++) { 
          this.w[i] = global.randn(0.0, scale);
        }
      } else {
        for(var i=0;i<n;i++) { 
          this.w[i] = c;
        }
      }
    }
  }

  Vol.prototype = {
    get: function(x, y, d) { 
      var ix=((this.sx * y)+x)*this.depth+d;
      return this.w[ix];
    },
    set: function(x, y, d, v) { 
      var ix=((this.sx * y)+x)*this.depth+d;
      this.w[ix] = v; 
    },
    add: function(x, y, d, v) { 
      var ix=((this.sx * y)+x)*this.depth+d;
      this.w[ix] += v; 
    },
    get_grad: function(x, y, d) { 
      var ix = ((this.sx * y)+x)*this.depth+d;
      return this.dw[ix]; 
    },
    set_grad: function(x, y, d, v) { 
      var ix = ((this.sx * y)+x)*this.depth+d;
      this.dw[ix] = v; 
    },
    add_grad: function(x, y, d, v) { 
      var ix = ((this.sx * y)+x)*this.depth+d;
      this.dw[ix] += v; 
    },
    cloneAndZero: function() { return new Vol(this.sx, this.sy, this.depth, 0.0)},
    clone: function() {
      var V = new Vol(this.sx, this.sy, this.depth, 0.0);
      var n = this.w.length;
      for(var i=0;i<n;i++) { V.w[i] = this.w[i]; }
      return V;
    },
    addFrom: function(V) { for(var k=0;k<this.w.length;k++) { this.w[k] += V.w[k]; }},
    addFromScaled: function(V, a) { for(var k=0;k<this.w.length;k++) { this.w[k] += a*V.w[k]; }},
    setConst: function(a) { for(var k=0;k<this.w.length;k++) { this.w[k] = a; }},

    toJSON: function() {
      // todo: we may want to only save d most significant digits to save space
      var json = {}
      json.sx = this.sx; 
      json.sy = this.sy;
      json.depth = this.depth;
      json.w = this.w;
      return json;
      // we wont back up gradients to save space
    },
    fromJSON: function(json) {
      // This is the deserialization entry point for untrusted model files, so
      // every field is validated before it is used to size an allocation.
      // Previously a missing/NaN dimension silently produced an *empty* Vol
      // (because zeros(NaN) returns []) and non-numeric weights were copied
      // straight into the typed array, so a corrupt file loaded "successfully"
      // and then returned garbage predictions. Fail loudly instead.
      var assert = global.assert;
      var isPlainish = global.isPlainish;
      var isFiniteNumber = global.isFiniteNumber;
      var validateVolDims = global.validateVolDims;

      assert(isPlainish(json), 'Error! Vol.fromJSON expects an object, got: ' + (json === null ? 'null' : typeof json));

      this.sx = json.sx;
      this.sy = json.sy;
      this.depth = json.depth;

      var n = validateVolDims(this.sx, this.sy, this.depth, 'Vol');

      var w = json.w;
      assert(w !== null && typeof w === 'object',
        'Error! Vol.fromJSON requires a "w" array of ' + n + ' numbers.');
      // NOTE: json.w is not necessarily an Array. Vol.toJSON stores a
      // Float64Array, and JSON.stringify() serializes a typed array as an
      // object keyed by stringified index ({"0":..,"1":..}), not as a list --
      // so every model file that has been through a JSON round trip (i.e. all
      // of them in practice) has that shape. Accept both, and index defensively.
      if (typeof w.length === 'number') {
        assert(w.length >= n, 'Error! Vol.fromJSON expected at least ' + n +
          ' weights but got ' + w.length + '. Truncated or corrupt model file?');
      }

      this.w = global.zeros(n);
      this.dw = global.zeros(n);
      // copy over the elements.
      for(var i=0;i<n;i++) {
        assert(isFiniteNumber(w[i]), 'Error! Vol.fromJSON found a missing or non-numeric weight at index ' +
          i + ' (' + w[i] + '). Corrupt model file?');
        this.w[i] = w[i];
      }
    }
  }

  global.Vol = Vol;
})(convnetjs);
