(() => {
  const token = new URLSearchParams(location.search).get("token") || "";
  const Capture = window.VslCapture || {};
  const CapturePipeline = Capture.CapturePipeline;
  const RecorderPage = Capture.RecorderPage;
  const parsePageGeneration = Capture.parsePageGeneration;
  const $ = (id) => document.getElementById(id);
  const STALE_PAGE_MESSAGE = "This recording session has ended. Open a new recorder from VSL Study.";

  const state = {
    stream: null,
    recorder: null,
    recId: null,
    mime: "",
    recording: false,
    audioTrack: false,
    audioDetected: false,
    startedAt: 0,
    timer: null,
    poll: null,
    heartbeat: null,
    pipeline: null,
    meterContext: null,
  };

  const page = new RecorderPage({
    generation: parsePageGeneration(location.search),
    onStopTracks: () => releasePreview(),
    onClearTimers: () => {
      stopHeartbeat();
      if (state.poll) {
        clearInterval(state.poll);
        state.poll = null;
      }
      if (state.timer) {
        clearInterval(state.timer);
        state.timer = null;
      }
    },
    onLockControls: () => lockStaleControls(),
    onStatus: (text) => status(text),
  });

  function headers() {
    const h = { "X-VSL-Token": token };
    if (page.generation != null) h["X-VSL-Generation"] = String(page.generation);
    return h;
  }

  const status = (text) => { $("status").textContent = text; };
  const TYPES = [
    "video/webm;codecs=vp9,opus",
    "video/webm;codecs=vp8,opus",
    "video/webm;codecs=vp8,pcm",
    "video/webm",
  ];

  function pickMime() {
    if (!window.MediaRecorder || !MediaRecorder.isTypeSupported) return "video/webm";
    return TYPES.find((t) => MediaRecorder.isTypeSupported(t)) || "video/webm";
  }

  async function api(path, opts) {
    if (page.ended) {
      const err = new Error(page.message || STALE_PAGE_MESSAGE);
      err.code = "stale_generation";
      err.status = 409;
      throw err;
    }
    const res = await fetch(path, opts);
    const data = await res.json().catch(() => ({}));
    if (data && (data.error === "stale_generation" || data.stale)) {
      page.markEnded(data.message || STALE_PAGE_MESSAGE);
      const err = new Error(page.message);
      err.code = "stale_generation";
      err.status = res.status;
      throw err;
    }
    if (!res.ok) {
      const err = new Error(data.message || data.error || res.statusText);
      err.code = data.error;
      err.status = res.status;
      throw err;
    }
    return data;
  }

  function lockStaleControls() {
    ["open-page", "choose", "start", "stop", "cancel"].forEach((id) => {
      const el = $(id);
      if (el) el.disabled = true;
    });
  }

  $("open-page").onclick = () => {
    const url = ($("url").value || "").trim();
    if (!/^https?:\/\//i.test(url)) {
      status("Enter an http or https address, or open the page yourself.");
      return;
    }
    window.open(url, "_blank", "noopener");
  };

  function stopTracks(stream) {
    if (!stream) return;
    stream.getTracks().forEach((t) => t.stop());
  }

  function abandonChooseStream(stream) {
    stopTracks(stream);
    if (state.stream === stream) {
      state.stream = null;
    }
    const preview = $("preview");
    if (preview && preview.srcObject === stream) {
      preview.srcObject = null;
    }
  }

  function releasePreview() {
    stopTracks(state.stream);
    state.stream = null;
    const preview = $("preview");
    if (preview) preview.srcObject = null;
    if (state.meterContext) {
      state.meterContext.close().catch(() => {});
      state.meterContext = null;
    }
  }

  async function beat() {
    if (!page.canMutate()) return;
    await api("/api/heartbeat", {
      method: "POST",
      headers: { ...headers(), "Content-Type": "application/json" },
      body: JSON.stringify({ id: state.recId, generation: page.generation }),
    });
  }

  function startHeartbeat() {
    if (state.heartbeat) return;
    beat().catch(() => {});
    state.heartbeat = setInterval(() => {
      if (!page.canMutate()) {
        stopHeartbeat();
        return;
      }
      beat().catch((err) => {
        if (!page.handleApiError(err)) {
          /* keep trying until lease expiry or a stale response */
        }
      });
    }, 15000);
  }

  function stopHeartbeat() {
    if (state.heartbeat) {
      clearInterval(state.heartbeat);
      state.heartbeat = null;
    }
  }

  function watchSession() {
    if (state.poll) clearInterval(state.poll);
    state.poll = setInterval(async () => {
      if (!page.canMutate()) {
        if (state.poll) {
          clearInterval(state.poll);
          state.poll = null;
        }
        return;
      }
      try {
        const s = await api("/api/session", { headers: headers() });
        if (page.handleSessionPoll(s)) {
          return;
        }
        if (s.cancelled) {
          page.markEnded("VSL Study closed. Stopping capture.");
        }
      } catch (err) {
        if (page.handleApiError(err)) return;
        if (state.recording) {
          status("Lost the local capture service. Stopping tracks.");
          await abortLocal("service_interrupted");
        }
      }
    }, 2000);
  }

  function setupMeter(stream) {
    state.audioDetected = false;
    $("signal-state").textContent = "No audible signal detected";
    const ctx = new AudioContext();
    state.meterContext = ctx;
    const source = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);
    const data = new Uint8Array(analyser.frequencyBinCount);
    const tick = () => {
      if (!state.stream) {
        ctx.close().catch(() => {});
        if (state.meterContext === ctx) state.meterContext = null;
        return;
      }
      analyser.getByteFrequencyData(data);
      let sum = 0;
      for (let i = 0; i < data.length; i += 1) sum += data[i];
      const avg = sum / data.length / 255;
      $("level").style.width = `${Math.min(100, Math.round(avg * 180))}%`;
      if (avg > 0.02) {
        state.audioDetected = true;
        $("signal-state").textContent = "Audible signal detected";
      }
      requestAnimationFrame(tick);
    };
    tick();
  }

  let chooseOp = 0;

  $("choose").onclick = async () => {
    if (!page.canMutate()) {
      page.markEnded();
      return;
    }
    const op = ++chooseOp;
    status("");
    if (!window.isSecureContext) {
      status("Capture needs a secure local page. This should be http://127.0.0.1 from VSL Study.");
      return;
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
      status("This browser cannot share a tab. Use current Chrome or Edge on Windows.");
      return;
    }
    stopTracks(state.stream);
    const video = { displaySurface: "browser" };
    const opts = {
      video,
      audio: true,
      selfBrowserSurface: "exclude",
      systemAudio: "exclude",
      preferCurrentTab: false,
    };
    let stream;
    try {
      stream = await navigator.mediaDevices.getDisplayMedia(opts);
    } catch (err) {
      if (page.ended || op !== chooseOp) return;
      status("Tab sharing was cancelled. Nothing was recorded. Choose the tab again when you are ready.");
      $("start").disabled = true;
      return;
    }
    if (!page.canMutate() || op !== chooseOp) {
      abandonChooseStream(stream);
      return;
    }
    state.stream = stream;
    $("preview").srcObject = stream;
    $("preview").muted = true;
    await $("preview").play().catch(() => {});
    if (!page.canMutate() || op !== chooseOp) {
      abandonChooseStream(stream);
      return;
    }
    const audioTracks = stream.getAudioTracks();
    state.audioTrack = audioTracks.length > 0;
    $("track-state").textContent = state.audioTrack
      ? "Audio track is present"
      : "No audio track — choose the tab again and turn on sharing that tab’s sound";
    $("start").disabled = !state.audioTrack;
    if (state.audioTrack) setupMeter(stream);
    stream.getVideoTracks().forEach((track) => {
      track.addEventListener("ended", () => {
        if (page.ended) return;
        if (state.recording) requestFinish("stop_sharing");
        else status("Sharing ended. Choose the tab again to record.");
      });
    });
    if (!state.audioTrack) {
      status("No audio track came back. In the picker, select the video tab and enable tab audio. Do not continue without sound.");
    } else {
      status("Preview is muted so it will not echo. Play a moment of the video to confirm the meter moves, then start at the beginning if you can.");
    }
  };

  function clock() {
    if (!state.recording) return;
    const s = Math.floor((Date.now() - state.startedAt) / 1000);
    const m = Math.floor(s / 60);
    $("clock").textContent = `Recording ${String(m).padStart(2, "0")}:${String(s % 60).padStart(2, "0")} — lead-in is part of this timeline`;
    const limit = Number($("limit").value);
    if (limit > 0 && s >= limit * 60) requestFinish("max_duration");
  }

  async function ensureSession() {
    if (state.recId) return;
    if (!page.canMutate()) {
      throw Object.assign(new Error(STALE_PAGE_MESSAGE), { code: "stale_generation" });
    }
    const created = await api("/api/recordings", {
      method: "POST",
      headers: { ...headers(), "Content-Type": "application/json" },
      body: JSON.stringify({
        title: $("title").value,
        source_url: $("url").value,
        mime_type: state.mime,
        max_duration_s: $("limit").value ? Number($("limit").value) * 60 : null,
        generation: page.generation,
      }),
    });
    if (!page.bindRecording(created.id)) {
      throw Object.assign(new Error(STALE_PAGE_MESSAGE), { code: "stale_generation" });
    }
    state.recId = created.id;
  }

  function makePipeline() {
    if (!CapturePipeline) {
      throw new Error("Recorder core did not load.");
    }
    return new CapturePipeline({
      uploadChunk: async (item) => {
        if (!state.recId) throw new Error("Recording session is missing.");
        await api(`/api/recordings/${state.recId}/chunks/${item.seq}`, {
          method: "PUT",
          headers: {
            ...headers(),
            "Content-Type": "application/octet-stream",
            "X-Content-SHA256": item.checksum,
            "X-Last-Chunk": item.last ? "1" : "0",
          },
          body: item.bytes,
        });
      },
      onStatus: (info) => {
        if (info && info.savedThrough != null && !info.last) {
          $("save-state").textContent = `Saved chunks through ${info.savedThrough}.`;
        }
        if (info && info.failed && state.pipeline && !state.pipeline.finishPromise) {
          status(info.message || "Saving failed. Partial media was kept.");
          requestFinish(info.code || "upload_failed");
        }
      },
    });
  }

  function waitForRecorderStop(rec) {
    return new Promise((resolve, reject) => {
      if (!rec || rec.state === "inactive") {
        resolve();
        return;
      }
      rec.addEventListener("stop", () => resolve(), { once: true });
      rec.addEventListener("error", () => {
        reject(Object.assign(new Error("The recorder reported an error. Partial media was kept."), { code: "recorder_error" }));
      }, { once: true });
      try {
        rec.stop();
      } catch (err) {
        resolve();
      }
    });
  }

  function requestFinish(reason) {
    const pipeline = state.pipeline;
    if (!pipeline) {
      return abortLocal(reason);
    }
    return pipeline.finish(reason, {
      stopRecorder: () => waitForRecorderStop(state.recorder),
      finalizeRemote: (stopReason) => finalizeRemote(stopReason),
      cancelRemote: (stopReason) => cancelRemote(stopReason),
    }).then((result) => {
      afterFinish(result, reason);
      state.recId = null;
      state.pipeline = null;
      state.recorder = null;
      return result;
    }).catch((err) => {
      if (page.handleApiError(err)) return;
      status((err && err.message) || String(err));
      releasePreview();
      restoreChooser();
    });
  }

  async function finalizeRemote(reason) {
    if (!state.recId) {
      throw Object.assign(new Error("Recording session is missing."), { code: "not_found" });
    }
    return api(`/api/recordings/${state.recId}/finalize`, {
      method: "POST",
      headers: { ...headers(), "Content-Type": "application/json" },
      body: JSON.stringify({
        stop_reason: reason,
        mime_type: state.mime,
        audio_track: state.audioTrack,
        audio_detected: state.audioDetected,
        title: $("title").value,
        browser: { userAgent: navigator.userAgent, vendor: navigator.vendor },
        generation: page.generation,
      }),
    });
  }

  async function cancelRemote(reason) {
    if (!state.recId || reason === "app_closed") return;
    try {
      await api(`/api/recordings/${state.recId}/cancel`, {
        method: "POST",
        headers: { ...headers(), "Content-Type": "application/json" },
        body: JSON.stringify({ reason, generation: page.generation }),
      });
    } catch (err) {
      /* session may already be gone */
    }
  }

  function afterFinish(result, reason) {
    state.recording = false;
    $("stop").disabled = true;
    $("cancel").disabled = true;
    if (state.timer) clearInterval(state.timer);
    releasePreview();
    const process = result && result.process;
    if (process) {
      status("Saved. Transcribing and screenshots continue in the VSL Study window. Open the evidence from there when it finishes.");
    } else if (reason === "user_stop" || reason === "max_duration") {
      status((result && result.error && result.error.message) || "Partial recording saved. It was not treated as a complete VSL, so it was not analyzed automatically.");
      restoreChooser();
    } else {
      status((result && result.error && result.error.message) || "Recording stopped without a complete capture. Partial media already saved was kept.");
      restoreChooser();
    }
  }

  function restoreChooser() {
    if (page.ended) {
      lockStaleControls();
      return;
    }
    $("choose").disabled = false;
    $("start").disabled = true;
    $("stop").disabled = true;
    $("cancel").disabled = false;
    $("clock").textContent = "Not recording";
  }

  $("start").onclick = async () => {
    if (!page.canMutate()) {
      page.markEnded();
      return;
    }
    if (!state.stream || !state.audioTrack) {
      status("Choose a tab with audio first.");
      return;
    }
    try {
      state.mime = pickMime();
      await ensureSession();
      watchSession();
      state.pipeline = makePipeline();
      const rec = new MediaRecorder(state.stream, { mimeType: state.mime, videoBitsPerSecond: 2_500_000 });
      state.recorder = rec;
      rec.ondataavailable = (ev) => {
        if (!state.pipeline || page.ended || !page.canMutate()) return;
        try {
          state.pipeline.acceptMedia(ev.data);
        } catch (err) {
          status(err.message || String(err));
          requestFinish("upload_failed");
        }
      };
      rec.onerror = () => {
        if (state.pipeline) {
          state.pipeline._fail("recorder_error", "The recorder reported an error. Partial media was kept if anything was saved.");
        }
        requestFinish("recorder_error");
      };
      rec.start(3000);
      state.recording = true;
      state.startedAt = Date.now();
      $("start").disabled = true;
      $("choose").disabled = true;
      $("stop").disabled = false;
      $("clock").textContent = "Recording 00:00 — lead-in is part of this timeline";
      state.timer = setInterval(clock, 250);
      status("Return to the video tab and play it normally. Come back here to stop.");
    } catch (err) {
      status(err.message || String(err));
    }
  };

  async function abortLocal(reason) {
    const pipeline = state.pipeline;
    if (pipeline && pipeline.finishPromise) {
      return pipeline.finishPromise;
    }
    if (pipeline) {
      pipeline._fail(reason || "cancelled", "Recording stopped. Partial media already saved was kept.");
      return requestFinish(reason);
    }
    state.recording = false;
    try {
      if (state.recorder && state.recorder.state !== "inactive") state.recorder.stop();
    } catch (err) {
      /* ignore */
    }
    releasePreview();
    await cancelRemote(reason);
    restoreChooser();
  }

  async function cancelPage() {
    if (page.ended) return;
    $("cancel").disabled = true;
    $("stop").disabled = true;
    if (state.timer) clearInterval(state.timer);
    state.recording = false;
    try {
      if (state.pipeline) {
        await requestFinish("client_cancel");
      } else {
        await abortLocal("client_cancel");
      }
    } catch (err) {
      status(err.message || String(err));
    }
    try {
      await api("/api/page/cancel", {
        method: "POST",
        headers: { ...headers(), "Content-Type": "application/json" },
        body: JSON.stringify({ reason: "client_cancel", generation: page.generation }),
      });
    } catch (err) {
      /* desktop may already have reset */
    }
    stopHeartbeat();
    if (state.poll) {
      clearInterval(state.poll);
      state.poll = null;
    }
    restoreChooser();
    $("cancel").disabled = true;
    status("Cancelled. You can close this tab. VSL Study is ready for another recording or a local file.");
  }

  $("stop").onclick = () => {
    if (!page.canMutate()) {
      page.markEnded();
      return;
    }
    requestFinish("user_stop");
  };
  $("cancel").onclick = () => cancelPage();
  document.addEventListener("visibilitychange", () => {
    if (page.ended) return;
    if (document.visibilityState === "visible") beat().catch((err) => page.handleApiError(err));
  });
  window.addEventListener("pagehide", () => {
    releasePreview();
  });
  if (page.generation == null) {
    page.markEnded(STALE_PAGE_MESSAGE);
  } else {
    startHeartbeat();
    watchSession();
  }
})();
