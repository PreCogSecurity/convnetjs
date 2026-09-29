// logger.js
//
// Minimal structured logger for the ConvNetJS demos.
//
// The demos previously used raw console.log() for status output, which gives a
// CI smoke test nothing to assert against and gives a human reading a browser
// console no way to filter or correlate messages. This writes one JSON object
// per line instead, so output is both human readable and machine parseable:
//
//   {"ts":"2026-09-29T12:00:00.000Z","level":"info","msg":"starting","batch":0}
//
// Design constraints:
//
//  * No dependencies, ES5 only, no build step -- it is a plain <script> tag in
//    the demo pages, like every other file in demo/js.
//  * It never throws. Demo code logs values that come straight from the
//    network data files and from user input, and a logger that can crash the
//    page (circular object, a toJSON() that throws, a getter that throws) is
//    just a denial of service on the demo.
//  * Callers cannot spoof the envelope. Structured fields are applied before
//    the caller-supplied metadata, so a "level" or "ts" key in the metadata
//    can never override the real one.
//
// Usage from a demo page:
//
//   <script src="js/logger.js"></script>
//   <script>
//     demoLog.setLevel('debug');            // or leave at the default 'info'
//     demoLog.info('starting', {batch: 0});
//   </script>
//
// The active level can also be set before this file loads:
//
//   <script>window.CONVNETJS_LOG_LEVEL = 'debug';</script>
(function(global) {
  "use strict";

  var LEVELS = {debug: 10, info: 20, warn: 30, error: 40, silent: 100};

  var DEFAULT_LEVEL = 'info';

  var currentLevel = DEFAULT_LEVEL;
  var sink = null; // defaults to console.log

  function isValidLevel(level) {
    return Object.prototype.hasOwnProperty.call(LEVELS, level);
  }

  // Resolve the sink lazily so that a page (or a test) can swap console out
  // after this file has loaded, and so we never capture a console reference
  // that does not exist.
  function getSink() {
    if (sink) { return sink; }
    if (typeof console !== 'undefined' && console && typeof console.log === 'function') {
      return function(line) { console.log(line); };
    }
    return null;
  }

  // JSON.stringify throws on circular structures and on objects whose toJSON()
  // throws. Fall back to a string coercion rather than propagating, because a
  // logging call must not be able to take down the page.
  function safeStringify(obj) {
    try {
      return JSON.stringify(obj);
    } catch (e) {
      try {
        return '"' + String(obj).replace(/"/g, '\\"') + '"';
      } catch (e2) {
        return '"[unserializable]"';
      }
    }
  }

  // Build the record. Callers pass arbitrary metadata, so guard against the two
  // ways that metadata can be hostile: not being an object, and carrying
  // enumerable own properties that throw when read.
  function buildRecord(level, msg, meta) {
    var record = {ts: new Date().toISOString(), level: level};

    if (meta !== null && typeof meta === 'object') {
      for (var key in meta) {
        if (Object.prototype.hasOwnProperty.call(meta, key)) {
          try {
            record[key] = meta[key];
          } catch (e) {
            record[key] = '[unreadable]';
          }
        }
      }
    }

    // Canonical fields are written last so metadata cannot spoof them.
    record.ts = new Date().toISOString();
    record.level = level;
    record.msg = (typeof msg === 'undefined') ? '' : String(msg);
    return record;
  }

  var logger = {

    // Current minimum level. Messages below it are dropped.
    get level() { return currentLevel; },

    setLevel: function(level) {
      if (!isValidLevel(level)) {
        throw new Error('Unknown log level "' + level + '". Expected one of: ' +
          Object.keys(LEVELS).join(', ') + '.');
      }
      currentLevel = level;
      return logger;
    },

    // Replace the output sink. Pass null to restore the console.
    setSink: function(fn) {
      if (fn !== null && typeof fn !== 'function') {
        throw new Error('setSink expects a function or null.');
      }
      sink = fn;
      return logger;
    },

    isEnabled: function(level) {
      return LEVELS[level] >= LEVELS[currentLevel];
    },

    // log(level, msg, meta) -- emits one JSON line if the level is enabled.
    log: function(level, msg, meta) {
      if (!isValidLevel(level)) {
        throw new Error('Unknown log level "' + level + '".');
      }
      if (!logger.isEnabled(level)) { return false; }
      var out = getSink();
      if (!out) { return false; }
      out(safeStringify(buildRecord(level, msg, meta)));
      return true;
    },

    debug: function(msg, meta) { return logger.log('debug', msg, meta); },
    info: function(msg, meta) { return logger.log('info', msg, meta); },
    warn: function(msg, meta) { return logger.log('warn', msg, meta); },
    error: function(msg, meta) { return logger.log('error', msg, meta); }
  };

  // Pick up a level configured by the page before this script ran.
  if (typeof global.CONVNETJS_LOG_LEVEL !== 'undefined') {
    if (isValidLevel(global.CONVNETJS_LOG_LEVEL)) {
      currentLevel = global.CONVNETJS_LOG_LEVEL;
    }
  }

  global.demoLog = logger;
})(window);
