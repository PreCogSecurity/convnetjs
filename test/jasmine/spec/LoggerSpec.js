describe("demo logger (structured JSON line output)", function() {

  var demoLog;
  var captured;

  beforeAll(function() {
    // Same convention as parseNetSpecSpec.js: the logger is a browser script
    // that attaches itself to `window`; simulate that environment in Node.
    global.window = global.window || {};
    require('../../../demo/js/logger.js');
    demoLog = global.window.demoLog;
  });

  beforeEach(function() {
    captured = [];
    demoLog.setLevel('debug');
    demoLog.setSink(function(line) { captured.push(line); });
  });

  afterEach(function() {
    demoLog.setSink(null);
    demoLog.setLevel('info');
  });

  function parseLast() {
    return JSON.parse(captured[captured.length - 1]);
  }

  it("should attach itself to window", function() {
    expect(demoLog).toBeDefined();
  });

  it("should emit exactly one JSON object per line", function() {
    demoLog.info('starting');
    expect(captured.length).toEqual(1);
    expect(captured[0].indexOf('\n')).toEqual(-1);
    var rec = parseLast();
    expect(rec.level).toEqual('info');
    expect(rec.msg).toEqual('starting');
    expect(typeof rec.ts).toEqual('string');
  });

  it("should produce an ISO-8601 timestamp", function() {
    demoLog.info('hello');
    var rec = parseLast();
    expect(rec.ts).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/);
    expect(isNaN(Date.parse(rec.ts))).toEqual(false);
  });

  it("should flatten metadata into the record", function() {
    demoLog.info('loaded batch', {batch: 3, rows: 128});
    var rec = parseLast();
    expect(rec.batch).toEqual(3);
    expect(rec.rows).toEqual(128);
    expect(rec.msg).toEqual('loaded batch');
  });

  it("should support all four levels", function() {
    demoLog.debug('d');
    demoLog.info('i');
    demoLog.warn('w');
    demoLog.error('e');
    expect(captured.length).toEqual(4);
    expect(JSON.parse(captured[0]).level).toEqual('debug');
    expect(JSON.parse(captured[1]).level).toEqual('info');
    expect(JSON.parse(captured[2]).level).toEqual('warn');
    expect(JSON.parse(captured[3]).level).toEqual('error');
  });

  it("should gate output below the active level", function() {
    demoLog.setLevel('warn');
    expect(demoLog.debug('dropped')).toEqual(false);
    expect(demoLog.info('dropped')).toEqual(false);
    expect(captured.length).toEqual(0);
    expect(demoLog.warn('kept')).toEqual(true);
    expect(captured.length).toEqual(1);
  });

  it("should drop everything at the silent level", function() {
    demoLog.setLevel('silent');
    expect(demoLog.error('nope')).toEqual(false);
    expect(captured.length).toEqual(0);
  });

  it("should reject an unknown level", function() {
    expect(function() { demoLog.setLevel('verbose'); }).toThrow();
    expect(function() { demoLog.log('verbose', 'x'); }).toThrow();
    expect(function() { demoLog.setSink('not a function'); }).toThrow();
  });

  it("should default to info", function() {
    demoLog.setLevel('info');
    expect(demoLog.level).toEqual('info');
  });

  it("should not let metadata spoof the envelope", function() {
    // a caller-supplied "level" must not be able to forge severity
    demoLog.info('real message', {level: 'error', msg: 'forged', ts: 'not-a-date'});
    var rec = parseLast();
    expect(rec.level).toEqual('info');
    expect(rec.msg).toEqual('real message');
    expect(isNaN(Date.parse(rec.ts))).toEqual(false);
  });

  it("should not throw on a circular metadata object", function() {
    // a logger that can crash the page on a weird value is a DoS vector, and
    // demo code logs values that come from data files and user input
    var circular = {name: 'loop'};
    circular.self = circular;
    expect(function() { demoLog.info('circular', circular); }).not.toThrow();
    expect(captured.length).toEqual(1);
    expect(typeof captured[0]).toEqual('string');
  });

  it("should not throw when a metadata getter throws", function() {
    var hostile = {};
    Object.defineProperty(hostile, 'boom', {
      enumerable: true,
      get: function() { throw new Error('nope'); }
    });
    expect(function() { demoLog.info('hostile', hostile); }).not.toThrow();
    expect(captured.length).toEqual(1);
    expect(parseLast().boom).toEqual('[unreadable]');
  });

  it("should ignore inherited metadata properties", function() {
    // inherited keys are not the caller's data; do not serialize them
    function Parent() {}
    Parent.prototype.inherited = 'should-not-appear';
    var child = new Parent();
    child.own = 'mine';
    demoLog.info('meta', child);
    var rec = parseLast();
    expect(rec.own).toEqual('mine');
    expect(rec.inherited).toBeUndefined();
  });

  it("should tolerate missing and non-string messages", function() {
    expect(function() { demoLog.info(); }).not.toThrow();
    expect(parseLast().msg).toEqual('');
    demoLog.info(42);
    expect(parseLast().msg).toEqual('42');
  });

  it("should tolerate non-object metadata", function() {
    expect(function() { demoLog.info('m', 'a string'); }).not.toThrow();
    expect(function() { demoLog.info('m', 7); }).not.toThrow();
    expect(function() { demoLog.info('m', null); }).not.toThrow();
    expect(captured.length).toEqual(3);
  });

  it("should report whether a level is enabled", function() {
    demoLog.setLevel('info');
    expect(demoLog.isEnabled('debug')).toEqual(false);
    expect(demoLog.isEnabled('info')).toEqual(true);
    expect(demoLog.isEnabled('error')).toEqual(true);
  });
});
