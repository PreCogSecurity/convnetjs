(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience
  var assert = global.assert;
  var isPlainish = global.isPlainish;

  // Net manages a set of layers
  // For now constraints: Simple linear order of layers, first layer input last layer a cost layer
  var Net = function(options) {
    this.layers = [];
  }

  Net.prototype = {
    
    // takes a list of layer definitions and creates the network layer objects
    makeLayers: function(defs) {

      // few checks
      assert(defs instanceof Array, 'Error! layer_defs must be an array of layer definition objects.');
      assert(defs.length >= 2, 'Error! At least one input layer and one loss layer are required.');
      assert(defs[0].type === 'input', 'Error! First layer must be the input layer, to declare size of inputs');

      // desugar layer_defs for adding activation, dropout layers etc
      var desugar = function() {
        var new_defs = [];
        for(var i=0;i<defs.length;i++) {
          var def = defs[i];
          
          if(def.type==='softmax' || def.type==='svm') {
            // add an fc layer here, there is no reason the user should
            // have to worry about this and we almost always want to
            new_defs.push({type:'fc', num_neurons: def.num_classes});
          }

          if(def.type==='regression') {
            // add an fc layer here, there is no reason the user should
            // have to worry about this and we almost always want to
            new_defs.push({type:'fc', num_neurons: def.num_neurons});
          }

          if((def.type==='fc' || def.type==='conv') 
              && typeof(def.bias_pref) === 'undefined'){
            def.bias_pref = 0.0;
            if(typeof def.activation !== 'undefined' && def.activation === 'relu') {
              def.bias_pref = 0.1; // relus like a bit of positive bias to get gradients early
              // otherwise it's technically possible that a relu unit will never turn on (by chance)
              // and will never get any gradient and never contribute any computation. Dead relu.
            }
          }

          new_defs.push(def);

          if(typeof def.activation !== 'undefined') {
            if(def.activation==='relu') { new_defs.push({type:'relu'}); }
            else if (def.activation==='sigmoid') { new_defs.push({type:'sigmoid'}); }
            else if (def.activation==='tanh') { new_defs.push({type:'tanh'}); }
            else if (def.activation==='maxout') {
              // create maxout activation, and pass along group size, if provided
              // NOTE: this used to read `def.group_size !== 'undefined'`, which
              // compares the *value* against the string 'undefined' and so was
              // always true -- a default of group_size=undefined was pushed
              // through and only survived because MaxoutLayer re-applied its own
              // default. An explicit null/0 slipped past and produced NaN or
              // infinite out_depth. Validate properly now.
              var gs = typeof def.group_size !== 'undefined' ? def.group_size : 2;
              assert(global.isNonNegInt(gs) && gs > 0, 'Error! maxout group_size must be a positive integer, got: ' + gs);
              new_defs.push({type:'maxout', group_size:gs});
            }
            else { throw new Error('unsupported activation ' + def.activation); }
          }
          if(typeof def.drop_prob !== 'undefined' && def.type !== 'dropout') {
            new_defs.push({type:'dropout', drop_prob: def.drop_prob});
          }

        }
        return new_defs;
      }
      defs = desugar(defs);

      // create the layers
      this.layers = [];
      for(var i=0;i<defs.length;i++) {
        var def = defs[i];
        if(i>0) {
          var prev = this.layers[i-1];
          def.in_sx = prev.out_sx;
          def.in_sy = prev.out_sy;
          def.in_depth = prev.out_depth;
        }

        switch(def.type) {
          case 'fc': this.layers.push(new global.FullyConnLayer(def)); break;
          case 'lrn': this.layers.push(new global.LocalResponseNormalizationLayer(def)); break;
          case 'dropout': this.layers.push(new global.DropoutLayer(def)); break;
          case 'input': this.layers.push(new global.InputLayer(def)); break;
          case 'softmax': this.layers.push(new global.SoftmaxLayer(def)); break;
          case 'regression': this.layers.push(new global.RegressionLayer(def)); break;
          case 'conv': this.layers.push(new global.ConvLayer(def)); break;
          case 'pool': this.layers.push(new global.PoolLayer(def)); break;
          case 'relu': this.layers.push(new global.ReluLayer(def)); break;
          case 'sigmoid': this.layers.push(new global.SigmoidLayer(def)); break;
          case 'tanh': this.layers.push(new global.TanhLayer(def)); break;
          case 'maxout': this.layers.push(new global.MaxoutLayer(def)); break;
          case 'svm': this.layers.push(new global.SVMLayer(def)); break;
          default: throw new Error('unrecognized layer type ' + def.type);
        }
      }
    },

    // forward prop the network. 
    // The trainer class passes is_training = true, but when this function is
    // called from outside (not from the trainer), it defaults to prediction mode
    forward: function(V, is_training) {
      if(typeof(is_training) === 'undefined') is_training = false;
      var act = this.layers[0].forward(V, is_training);
      for(var i=1;i<this.layers.length;i++) {
        act = this.layers[i].forward(act, is_training);
      }
      return act;
    },

    getCostLoss: function(V, y) {
      this.forward(V, false);
      var N = this.layers.length;
      var loss = this.layers[N-1].backward(y);
      return loss;
    },
    
    // backprop: compute gradients wrt all parameters
    backward: function(y) {
      var N = this.layers.length;
      var loss = this.layers[N-1].backward(y); // last layer assumed to be loss layer
      for(var i=N-2;i>=0;i--) { // first layer assumed input
        this.layers[i].backward();
      }
      return loss;
    },
    getParamsAndGrads: function() {
      // accumulate parameters and gradients for the entire network
      var response = [];
      for(var i=0;i<this.layers.length;i++) {
        var layer_reponse = this.layers[i].getParamsAndGrads();
        for(var j=0;j<layer_reponse.length;j++) {
          response.push(layer_reponse[j]);
        }
      }
      return response;
    },
    getPrediction: function() {
      // this is a convenience function for returning the argmax
      // prediction, assuming the last layer of the net is a softmax
      var S = this.layers[this.layers.length-1];
      assert(S.layer_type === 'softmax', 'getPrediction function assumes softmax as last layer of the net!');

      var p = S.out_act.w;
      var maxv = p[0];
      var maxi = 0;
      for(var i=1;i<p.length;i++) {
        if(p[i] > maxv) { maxv = p[i]; maxi = i;}
      }
      return maxi; // return index of the class with highest class probability
    },
    toJSON: function() {
      var json = {};
      json.layers = [];
      for(var i=0;i<this.layers.length;i++) {
        json.layers.push(this.layers[i].toJSON());
      }
      return json;
    },
    fromJSON: function(json) {
      // Model files are frequently user supplied or fetched from a URL, so this
      // is untrusted-input boundary. Validate up front and build into a
      // local array: a rejected payload must leave `this` untouched rather than
      // leaving a half-built network behind that the caller will happily keep
      // using (and that would then emit confidently wrong predictions).
      assert(isPlainish(json), 'Error! Net.fromJSON expects an object with a "layers" array, got: ' +
        (json === null ? 'null' : typeof json));
      assert(json.layers instanceof Array, 'Error! Net.fromJSON expects json.layers to be an array.');

      var known = {
        'input': global.InputLayer,
        'relu': global.ReluLayer,
        'sigmoid': global.SigmoidLayer,
        'tanh': global.TanhLayer,
        'dropout': global.DropoutLayer,
        'conv': global.ConvLayer,
        'pool': global.PoolLayer,
        'lrn': global.LocalResponseNormalizationLayer,
        'softmax': global.SoftmaxLayer,
        'regression': global.RegressionLayer,
        'fc': global.FullyConnLayer,
        'maxout': global.MaxoutLayer,
        'svm': global.SVMLayer
      };

      var layers = [];
      for(var i=0;i<json.layers.length;i++) {
        var Lj = json.layers[i];
        assert(isPlainish(Lj), 'Error! Net.fromJSON: layer ' + i + ' is not an object.');
        var t = Lj.layer_type;
        var Ctor = Object.prototype.hasOwnProperty.call(known, t) ? known[t] : null;
        // Previously an unrecognized type left L undefined and blew up with an
        // opaque "Cannot read properties of undefined" TypeError.
        assert(Ctor !== null, 'Error! Net.fromJSON: unrecognized layer type ' + JSON.stringify(t) +
          ' at index ' + i + '.');
        var L = new Ctor();
        L.fromJSON(Lj);
        layers.push(L);
      }

      // commit only once every layer deserialized cleanly
      this.layers = layers;
    }
  }
  
  global.Net = Net;
})(convnetjs);
