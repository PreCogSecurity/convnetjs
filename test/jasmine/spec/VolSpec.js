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
});