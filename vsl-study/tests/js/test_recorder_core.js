const { test } = require("node:test");
const assert = require("node:assert/strict");
const crypto = require("crypto");
const path = require("path");

const { CapturePipeline } = require(path.resolve(__dirname, "../../src/vsl_study/recorder/recorder-core.js"));

function sha256(bytes) {
  return Promise.resolve(crypto.createHash("sha256").update(Buffer.from(bytes)).digest("hex"));
}

function blob(text, delayMs) {
  const encoded = Buffer.from(String(text));
  const wait = delayMs || 0;
  return {
    size: encoded.length,
    arrayBuffer() {
      return new Promise((resolve) => {
        const copy = Uint8Array.from(encoded);
        setTimeout(() => resolve(copy.buffer), wait);
      });
    },
  };
}

function pipeline(overrides) {
  const saved = [];
  const opts = Object.assign(
    {
      sha256,
      retryDelay: () => 0,
      stopTimeoutMs: 2000,
      uploadChunk: async (item) => {
        saved.push({
          seq: item.seq,
          last: item.last,
          text: Buffer.from(item.bytes).toString("utf8"),
        });
      },
    },
    overrides || {}
  );
  const pipe = new CapturePipeline(opts);
  pipe._saved = saved;
  return pipe;
}

function mediaText(saved) {
  return saved
    .filter((item) => !item.last)
    .sort((a, b) => a.seq - b.seq)
    .map((item) => item.text)
    .join("");
}

test("slow first chunk keeps original event order", async () => {
  const pipe = pipeline();
  pipe.acceptMedia(blob("A", 80));
  pipe.acceptMedia(blob("B", 0));
  const result = await pipe.finish("user_stop", {
    stopRecorder: async () => {},
    finalizeRemote: async () => ({ process: true, complete: true }),
  });
  assert.equal(mediaText(pipe._saved), "AB");
  assert.deepEqual(
    pipe._saved.map((item) => item.seq),
    [0, 1, 2]
  );
  assert.equal(result.process, true);
  assert.equal(pipe.finalizeCalls, 1);
});

test("final chunk slower than the old 50ms delay is still saved before finalize", async () => {
  const pipe = pipeline();
  pipe.acceptMedia(blob("A", 0));
  const started = Date.now();
  const result = await pipe.finish("user_stop", {
    stopRecorder: async () => {
      pipe.acceptMedia(blob("C", 120));
    },
    finalizeRemote: async () => {
      assert.equal(mediaText(pipe._saved), "AC");
      return { process: true, complete: true };
    },
  });
  assert.ok(Date.now() - started >= 120);
  assert.equal(mediaText(pipe._saved), "AC");
  assert.equal(result.process, true);
  assert.equal(pipe.finalizeCalls, 1);
});

test("upload retries preserve order and do not duplicate bytes", async () => {
  const remainingFails = { 0: 2 };
  const saved = [];
  const pipe = pipeline({
    uploadChunk: async (item) => {
      if ((remainingFails[item.seq] || 0) > 0) {
        remainingFails[item.seq] -= 1;
        const err = new Error("temporary");
        throw err;
      }
      saved.push({
        seq: item.seq,
        last: item.last,
        text: Buffer.from(item.bytes).toString("utf8"),
      });
    },
  });
  pipe.acceptMedia(blob("A", 0));
  pipe.acceptMedia(blob("B", 0));
  await pipe.finish("user_stop", {
    stopRecorder: async () => {},
    finalizeRemote: async () => ({ process: true }),
  });
  assert.equal(mediaText(saved), "AB");
  assert.equal(saved.filter((item) => item.seq === 0).length, 1);
  assert.equal(saved.filter((item) => item.seq === 1).length, 1);
});

test("upload rejection marks incomplete and skips processing", async () => {
  const pipe = pipeline({
    uploadChunk: async () => {
      throw Object.assign(new Error("disk failed"), { code: "upload_failed" });
    },
  });
  pipe.acceptMedia(blob("A", 0));
  const result = await pipe.finish("user_stop", {
    stopRecorder: async () => {},
    finalizeRemote: async () => {
      throw new Error("should not finalize");
    },
    cancelRemote: async () => {},
  });
  assert.equal(result.process, false);
  assert.equal(result.complete, false);
  assert.equal(pipe.incomplete, true);
  assert.equal(pipe.finalizeCalls, 0);
  assert.ok(pipe.cancelCalls >= 1);
});

test("backpressure stops without circular wait or unhandled rejection", async () => {
  const rejections = [];
  const onUnhandled = (err) => rejections.push(err);
  process.on("unhandledRejection", onUnhandled);
  try {
    let finishStarted = false;
    const hang = {
      size: 8,
      arrayBuffer() {
        return new Promise(() => {});
      },
    };
    const pipe = pipeline({ maxPendingBytes: 10 });
    pipe.acceptMedia(hang);
    pipe.acceptMedia(blob("ABCDEFGH", 0));
    assert.equal(pipe.failed, true);
    assert.equal(pipe.failReason, "backpressure");
    finishStarted = true;
    const result = await Promise.race([
      pipe.finish("backpressure", {
        stopRecorder: async () => {},
        finalizeRemote: async () => {
          throw new Error("should not finalize");
        },
        cancelRemote: async () => {},
      }),
      new Promise((_, reject) => setTimeout(() => reject(new Error("circular wait")), 500)),
    ]);
    assert.equal(finishStarted, true);
    assert.equal(result.process, false);
    assert.equal(pipe.finalizeCalls, 0);
    await new Promise((resolve) => setTimeout(resolve, 20));
    assert.equal(rejections.length, 0);
  } finally {
    process.off("unhandledRejection", onUnhandled);
  }
});

test("repeated stop produces one finalization", async () => {
  const pipe = pipeline();
  pipe.acceptMedia(blob("A", 0));
  const hooks = {
    stopRecorder: async () => {},
    finalizeRemote: async () => ({ process: true, complete: true }),
  };
  const first = pipe.finish("user_stop", hooks);
  const second = pipe.finish("user_stop", hooks);
  assert.equal(first, second);
  await first;
  await second;
  assert.equal(pipe.finalizeCalls, 1);
});

test("recorder error and stop sharing follow the intended lifecycle", async () => {
  const errored = pipeline();
  errored.acceptMedia(blob("A", 0));
  errored._fail("recorder_error", "The recorder reported an error.");
  const errResult = await errored.finish("recorder_error", {
    stopRecorder: async () => {},
    finalizeRemote: async () => {
      throw new Error("should not finalize after recorder error");
    },
    cancelRemote: async () => {},
  });
  assert.equal(errResult.process, false);
  assert.equal(errored.finalizeCalls, 0);

  const shared = pipeline();
  shared.acceptMedia(blob("A", 0));
  const shareResult = await shared.finish("stop_sharing", {
    stopRecorder: async () => {},
    finalizeRemote: async (reason) => {
      assert.equal(reason, "stop_sharing");
      return { process: false, complete: false, reason };
    },
  });
  assert.equal(shared.finalizeCalls, 1);
  assert.equal(shareResult.process, false);
  assert.equal(mediaText(shared._saved), "A");
});
