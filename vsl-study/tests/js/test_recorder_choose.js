const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const ROOT = path.resolve(__dirname, "../..");
const CORE_JS = path.join(ROOT, "src/vsl_study/recorder/recorder-core.js");
const RECORDER_JS = path.join(ROOT, "src/vsl_study/recorder/recorder.js");
const ENDED_MESSAGE = "This recording session has ended. Open a new recorder from VSL Study.";
const CANCELLED_MESSAGE = "Tab sharing was cancelled. Nothing was recorded. Choose the tab again when you are ready.";
const PREVIEW_MESSAGE = "Preview is muted so it will not echo. Play a moment of the video to confirm the meter moves, then start at the beginning if you can.";
const ELEMENT_IDS = [
  "url",
  "title",
  "limit",
  "open-page",
  "choose",
  "preview",
  "track-state",
  "level",
  "signal-state",
  "start",
  "stop",
  "cancel",
  "clock",
  "save-state",
  "status",
];

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function jsonResponse(body, status) {
  const code = status == null ? 200 : status;
  return {
    ok: code >= 200 && code < 300,
    status: code,
    json: async () => body,
  };
}

function makeTrack(kind, settings) {
  const listeners = {};
  const current = Object.assign(
    { width: 1280, height: 720, frameRate: 10, displaySurface: "browser" },
    settings || {}
  );
  return {
    kind,
    stopped: false,
    lastConstraints: null,
    constraintError: null,
    applyConstraints(constraints) {
      this.lastConstraints = constraints;
      if (this.constraintError) {
        return Promise.reject(this.constraintError);
      }
      if (constraints && constraints.width && constraints.width.ideal) {
        current.width = Math.min(Number(current.width) || constraints.width.ideal, constraints.width.ideal);
      }
      if (constraints && constraints.frameRate && constraints.frameRate.ideal) {
        current.frameRate = Math.min(
          Number(current.frameRate) || constraints.frameRate.ideal,
          constraints.frameRate.ideal
        );
      }
      return Promise.resolve();
    },
    getSettings() {
      return Object.assign({}, current);
    },
    stop() {
      this.stopped = true;
    },
    addEventListener(type, fn) {
      (listeners[type] || (listeners[type] = [])).push(fn);
    },
  };
}

function makeStream(hasAudio, videoSettings) {
  const video = makeTrack("video", videoSettings);
  const audio = hasAudio === false ? null : makeTrack("audio");
  const tracks = audio ? [video, audio] : [video];
  return {
    tracks,
    getTracks() {
      return tracks.slice();
    },
    getAudioTracks() {
      return audio ? [audio] : [];
    },
    getVideoTracks() {
      return [video];
    },
  };
}

function makeEl(id) {
  return {
    id,
    value: "",
    textContent: "",
    disabled: id === "start" || id === "stop",
    style: { width: "" },
    srcObject: null,
    muted: false,
    onclick: null,
    play() {
      return Promise.resolve();
    },
  };
}

async function flush() {
  for (let i = 0; i < 20; i += 1) {
    await Promise.resolve();
  }
}

function createHarness(options) {
  const opts = options || {};
  const deferPlay = Boolean(opts.deferPlay);
  const elements = {};
  ELEMENT_IDS.forEach((id) => {
    elements[id] = makeEl(id);
  });
  elements["track-state"].textContent = "No audio track yet";
  elements["signal-state"].textContent = "No audible signal detected";
  elements.clock.textContent = "Not recording";
  elements["save-state"].textContent = "Nothing saved yet.";

  const fetchCalls = [];
  const displayMediaCalls = [];
  const playCalls = [];
  const audioContexts = [];
  const mediaRecorders = [];
  const intervals = new Map();
  const unhandled = [];
  const documentListeners = {};
  let nextIntervalId = 1;

  elements.preview.play = () => {
    const pending = deferred();
    playCalls.push(pending);
    if (!deferPlay) {
      queueMicrotask(() => pending.resolve());
    }
    return pending.promise;
  };

  function FakeAudioContext() {
    audioContexts.push(this);
    this.closed = false;
  }
  FakeAudioContext.prototype.createMediaStreamSource = function () {
    return { connect() {} };
  };
  FakeAudioContext.prototype.createAnalyser = function () {
    return {
      fftSize: 0,
      frequencyBinCount: 8,
      getByteFrequencyData(arr) {
        arr.fill(0);
      },
    };
  };
  FakeAudioContext.prototype.close = function () {
    this.closed = true;
    return Promise.resolve();
  };

  function FakeMediaRecorder(stream, options) {
    mediaRecorders.push(this);
    this.stream = stream;
    this.options = options || {};
    this.state = "inactive";
    this.start = () => {
      this.state = "recording";
    };
    this.stop = () => {
      this.state = "inactive";
    };
    this.addEventListener = () => {};
  }
  FakeMediaRecorder.isTypeSupported = () => true;

  const navigator = {
    userAgent: "vsl-test",
    vendor: "vsl-test",
    mediaDevices: {
      getDisplayMedia(constraints) {
        const pending = deferred();
        displayMediaCalls.push({ ...pending, constraints });
        return pending.promise;
      },
    },
  };

  const document = {
    getElementById(id) {
      return elements[id] || null;
    },
    visibilityState: "visible",
    addEventListener(type, fn) {
      (documentListeners[type] || (documentListeners[type] = [])).push(fn);
    },
  };

  const sandbox = {
    console,
    URLSearchParams,
    Uint8Array,
    Promise,
    Error,
    JSON,
    Object,
    Number,
    Math,
    Date,
    String,
    Array,
    Boolean,
    parseInt,
    isFinite,
    undefined,
    queueMicrotask,
    setTimeout,
    clearTimeout,
    setInterval(fn, ms) {
      const id = nextIntervalId;
      nextIntervalId += 1;
      intervals.set(id, { fn, ms, id });
      return id;
    },
    clearInterval(id) {
      intervals.delete(id);
    },
    fetch(url, init) {
      const pending = deferred();
      fetchCalls.push({ url, init, done: false, ...pending });
      return pending.promise;
    },
    AudioContext: FakeAudioContext,
    MediaRecorder: FakeMediaRecorder,
    requestAnimationFrame() {
      return 1;
    },
    addEventListener() {},
    open() {},
    navigator,
    location: { search: opts.search || "?g=1" },
    document,
    isSecureContext: opts.isSecureContext !== false,
  };
  sandbox.globalThis = sandbox;
  sandbox.window = sandbox;
  sandbox.self = sandbox;

  const onUnhandled = (err) => {
    unhandled.push(err);
  };
  process.on("unhandledRejection", onUnhandled);

  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(CORE_JS, "utf8"), sandbox, { filename: "recorder-core.js" });
  vm.runInContext(fs.readFileSync(RECORDER_JS, "utf8"), sandbox, { filename: "recorder.js" });

  function pendingFetch(urlPart) {
    return fetchCalls.find((call) => !call.done && String(call.url).includes(urlPart));
  }

  async function resolveFetch(urlPart, body, status) {
    await flush();
    const call = pendingFetch(urlPart);
    assert.ok(call, `expected pending fetch for ${urlPart}`);
    call.done = true;
    call.resolve(jsonResponse(body, status));
    await flush();
    return call;
  }

  async function deliverStaleHeartbeat(message) {
    await resolveFetch("/api/heartbeat", { error: "stale_generation", message: message || ENDED_MESSAGE }, 409);
  }

  async function deliverStalePoll(message) {
    const poll = [...intervals.values()].find((item) => item.ms === 2000);
    assert.ok(poll, "expected session poll interval");
    const running = Promise.resolve().then(() => poll.fn());
    await flush();
    await resolveFetch("/api/session", { error: "stale_generation", message: message || ENDED_MESSAGE }, 409);
    await running;
    await flush();
  }

  async function ackHeartbeat() {
    await resolveFetch("/api/heartbeat", { ok: true }, 200);
  }

  function clickChoose() {
    return Promise.resolve(elements.choose.onclick());
  }

  function dispose() {
    process.off("unhandledRejection", onUnhandled);
  }

  return {
    elements,
    fetchCalls,
    displayMediaCalls,
    playCalls,
    audioContexts,
    mediaRecorders,
    intervals,
    unhandled,
    clickChoose,
    ackHeartbeat,
    deliverStaleHeartbeat,
    deliverStalePoll,
    resolveFetch,
    flush,
    dispose,
  };
}

function assertLockedEnded(harness, message) {
  const els = harness.elements;
  assert.equal(els.status.textContent, message);
  assert.equal(els.choose.disabled, true);
  assert.equal(els.start.disabled, true);
  assert.equal(els.stop.disabled, true);
  assert.equal(els.cancel.disabled, true);
}

function assertAllStopped(stream) {
  const tracks = stream.getTracks();
  assert.ok(tracks.length > 0);
  tracks.forEach((track) => {
    assert.equal(track.stopped, true, `${track.kind} track should be stopped`);
  });
}

test("picker resolving after cancellation stops tracks and keeps the page ended", async () => {
  const harness = createHarness();
  try {
    const chooseDone = harness.clickChoose();
    assert.equal(harness.displayMediaCalls.length, 1);
    await harness.deliverStaleHeartbeat(ENDED_MESSAGE);
    assertLockedEnded(harness, ENDED_MESSAGE);

    const stream = makeStream(true);
    harness.displayMediaCalls[0].resolve(stream);
    await chooseDone;
    await harness.flush();

    assertAllStopped(stream);
    assert.equal(harness.elements.preview.srcObject, null);
    assert.equal(harness.elements["track-state"].textContent, "No audio track yet");
    assertLockedEnded(harness, ENDED_MESSAGE);
    assert.equal(harness.playCalls.length, 0);
    assert.equal(harness.audioContexts.length, 0);
    assert.equal(harness.mediaRecorders.length, 0);
    assert.equal(harness.unhandled.length, 0);
  } finally {
    harness.dispose();
  }
});

test("page ending during preview playback cleans up that stream and skips later setup", async () => {
  const harness = createHarness({ deferPlay: true });
  try {
    await harness.ackHeartbeat();
    const chooseDone = harness.clickChoose();
    const stream = makeStream(true);
    harness.displayMediaCalls[0].resolve(stream);
    await harness.flush();
    assert.equal(harness.playCalls.length, 1);
    assert.equal(harness.elements.preview.srcObject, stream);
    stream.getTracks().forEach((track) => {
      assert.equal(track.stopped, false);
    });

    await harness.deliverStalePoll(ENDED_MESSAGE);
    assertLockedEnded(harness, ENDED_MESSAGE);
    assertAllStopped(stream);
    assert.equal(harness.elements.preview.srcObject, null);

    harness.playCalls[0].resolve();
    await chooseDone;
    await harness.flush();

    assertAllStopped(stream);
    assert.equal(harness.elements.preview.srcObject, null);
    assert.equal(harness.elements["track-state"].textContent, "No audio track yet");
    assertLockedEnded(harness, ENDED_MESSAGE);
    assert.equal(harness.audioContexts.length, 0);
    assert.equal(harness.mediaRecorders.length, 0);
    assert.equal(harness.unhandled.length, 0);
  } finally {
    harness.dispose();
  }
});

test("picker rejection after cancellation keeps the ended message and stays handled", async () => {
  const harness = createHarness();
  try {
    const chooseDone = harness.clickChoose();
    await harness.deliverStaleHeartbeat(ENDED_MESSAGE);
    assertLockedEnded(harness, ENDED_MESSAGE);

    harness.displayMediaCalls[0].reject(Object.assign(new Error("NotAllowedError"), { name: "NotAllowedError" }));
    await chooseDone;
    await harness.flush();

    assertLockedEnded(harness, ENDED_MESSAGE);
    assert.notEqual(harness.elements.status.textContent, CANCELLED_MESSAGE);
    assert.equal(harness.elements.start.disabled, true);
    assert.equal(harness.audioContexts.length, 0);
    assert.equal(harness.unhandled.length, 0);
  } finally {
    harness.dispose();
  }
});

test("an older Choose result cannot replace or stop a newer stream", async () => {
  const harness = createHarness();
  try {
    await harness.ackHeartbeat();
    const first = harness.clickChoose();
    const second = harness.clickChoose();
    assert.equal(harness.displayMediaCalls.length, 2);

    const older = makeStream(true);
    const newer = makeStream(true);
    harness.displayMediaCalls[1].resolve(newer);
    await second;
    await harness.flush();

    assert.equal(harness.elements.preview.srcObject, newer);
    assert.equal(harness.elements.start.disabled, false);
    assert.equal(harness.elements.status.textContent, PREVIEW_MESSAGE);
    assert.equal(harness.audioContexts.length, 1);
    newer.getTracks().forEach((track) => {
      assert.equal(track.stopped, false);
    });

    harness.displayMediaCalls[0].resolve(older);
    await first;
    await harness.flush();

    assertAllStopped(older);
    newer.getTracks().forEach((track) => {
      assert.equal(track.stopped, false);
    });
    assert.equal(harness.elements.preview.srcObject, newer);
    assert.equal(harness.elements.start.disabled, false);
    assert.equal(harness.elements.status.textContent, PREVIEW_MESSAGE);
    assert.equal(harness.audioContexts.length, 1);
    assert.equal(harness.unhandled.length, 0);
  } finally {
    harness.dispose();
  }
});

test("a valid page can select a stream with audio, attach preview, and enable Start", async () => {
  const harness = createHarness();
  try {
    await harness.ackHeartbeat();
    const chooseDone = harness.clickChoose();
    const stream = makeStream(true);
    harness.displayMediaCalls[0].resolve(stream);
    await chooseDone;
    await harness.flush();

    stream.getTracks().forEach((track) => {
      assert.equal(track.stopped, false);
    });
    assert.equal(harness.elements.preview.srcObject, stream);
    assert.equal(harness.elements.preview.muted, true);
    assert.equal(harness.elements["track-state"].textContent, "Audio track is present");
    assert.equal(harness.elements.start.disabled, false);
    assert.equal(harness.elements.status.textContent, PREVIEW_MESSAGE);
    assert.equal(harness.audioContexts.length, 1);
    assert.equal(harness.mediaRecorders.length, 0);
    assert.equal(harness.unhandled.length, 0);
  } finally {
    harness.dispose();
  }
});

test("getDisplayMedia requests study width and frame rate", async () => {
  const harness = createHarness();
  try {
    await harness.ackHeartbeat();
    const chooseDone = harness.clickChoose();
    const video = harness.displayMediaCalls[0].constraints.video;
    assert.equal(video.displaySurface, "browser");
    assert.equal(video.width.ideal, 1280);
    assert.equal(video.width.max, 1280);
    assert.equal(video.frameRate.ideal, 10);
    assert.equal(video.frameRate.max, 10);
    harness.displayMediaCalls[0].resolve(makeStream(true));
    await chooseDone;
  } finally {
    harness.dispose();
  }
});

test("applyConstraints rejection is reported and not labeled as the study profile", async () => {
  const harness = createHarness();
  try {
    await harness.ackHeartbeat();
    const chooseDone = harness.clickChoose();
    const stream = makeStream(true, { width: 3840, height: 1730, frameRate: 30 });
    stream.getVideoTracks()[0].constraintError = Object.assign(new Error("OverconstrainedError"), {
      name: "OverconstrainedError",
    });
    harness.displayMediaCalls[0].resolve(stream);
    await chooseDone;
    await harness.flush();
    assert.equal(harness.elements.start.disabled, false);
    assert.match(harness.elements.status.textContent, /not the study 1280px\/10fps profile/);
    assert.equal(stream.getVideoTracks()[0].stopped, false);
  } finally {
    harness.dispose();
  }
});

test("cancellation during applyConstraints stops the stream and keeps the ended page", async () => {
  const harness = createHarness();
  try {
    const chooseDone = harness.clickChoose();
    const stream = makeStream(true);
    const video = stream.getVideoTracks()[0];
    let release;
    video.applyConstraints = () =>
      new Promise((resolve) => {
        release = resolve;
      });
    harness.displayMediaCalls[0].resolve(stream);
    await harness.flush();
    await harness.deliverStaleHeartbeat(ENDED_MESSAGE);
    assertLockedEnded(harness, ENDED_MESSAGE);
    release();
    await chooseDone;
    await harness.flush();
    assertAllStopped(stream);
    assert.equal(harness.elements.preview.srcObject, null);
    assertLockedEnded(harness, ENDED_MESSAGE);
    assert.equal(harness.unhandled.length, 0);
  } finally {
    harness.dispose();
  }
});
