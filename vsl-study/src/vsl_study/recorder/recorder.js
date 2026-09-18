(() => {
  const token = new URLSearchParams(location.search).get("token") || "";
  const headers = () => ({ "X-VSL-Token": token });
  const $ = (id) => document.getElementById(id);

  const state = {
    stream: null,
    recorder: null,
    recId: null,
    seq: 0,
    mime: "",
    recording: false,
    stopping: false,
    finalized: false,
    audioTrack: false,
    audioDetected: false,
    startedAt: 0,
    timer: null,
    poll: null,
    pending: 0,
    pendingBytes: 0,
    maxPendingBytes: 12 * 1024 * 1024,
    queue: [],
    uploading: false,
  };

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

  async function sha256(buf) {
    const hash = await crypto.subtle.digest("SHA-256", buf);
    return [...new Uint8Array(hash)].map((b) => b.toString(16).padStart(2, "0")).join("");
  }

  async function api(path, opts) {
    const res = await fetch(path, opts);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const err = new Error(data.message || data.error || res.statusText);
      err.code = data.error;
      err.status = res.status;
      throw err;
    }
    return data;
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

  function watchSession() {
    if (state.poll) clearInterval(state.poll);
    state.poll = setInterval(async () => {
      try {
        const s = await api("/api/session", { headers: headers() });
        if (s.cancelled) {
          status("VSL Study closed. Stopping capture.");
          await abortLocal("app_closed");
        }
      } catch (err) {
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
    const source = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);
    const data = new Uint8Array(analyser.frequencyBinCount);
    const tick = () => {
      if (!state.stream) {
        ctx.close().catch(() => {});
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

  $("choose").onclick = async () => {
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
      status("Tab sharing was cancelled. Nothing was recorded.");
      $("start").disabled = true;
      return;
    }
    state.stream = stream;
    $("preview").srcObject = stream;
    $("preview").muted = true;
    await $("preview").play().catch(() => {});
    const audioTracks = stream.getAudioTracks();
    state.audioTrack = audioTracks.length > 0;
    $("track-state").textContent = state.audioTrack
      ? "Audio track is present"
      : "No audio track — choose the tab again and turn on sharing that tab’s sound";
    $("start").disabled = !state.audioTrack;
    if (state.audioTrack) setupMeter(stream);
    stream.getVideoTracks().forEach((track) => {
      track.addEventListener("ended", () => {
        if (state.recording) finish("stop_sharing");
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
    if (limit > 0 && s >= limit * 60) finish("max_duration");
  }

  async function ensureSession() {
    if (state.recId) return;
    const created = await api("/api/recordings", {
      method: "POST",
      headers: { ...headers(), "Content-Type": "application/json" },
      body: JSON.stringify({
        title: $("title").value,
        source_url: $("url").value,
        mime_type: state.mime,
        max_duration_s: $("limit").value ? Number($("limit").value) * 60 : null,
      }),
    });
    state.recId = created.id;
  }

  async function pump() {
    if (state.uploading) return;
    state.uploading = true;
    try {
      while (state.queue.length) {
        if (state.pendingBytes > state.maxPendingBytes && state.recording) {
          status("Saving cannot keep up. Stopping so media is not dropped silently.");
          await finish("upload_failed");
          break;
        }
        const item = state.queue[0];
        let attempt = 0;
        while (attempt < 5) {
          try {
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
            state.pendingBytes -= item.bytes.byteLength;
            state.queue.shift();
            $("save-state").textContent = `Saved chunks through ${item.seq}.`;
            break;
          } catch (err) {
            if (err.code === "conflict" || err.code === "finalized") throw err;
            attempt += 1;
            if (attempt >= 5) throw err;
            await new Promise((r) => setTimeout(r, 400 * attempt));
          }
        }
      }
    } finally {
      state.uploading = false;
    }
  }

  async function enqueue(blob, last) {
    const buf = await blob.arrayBuffer();
    const bytes = new Uint8Array(buf);
    if (bytes.byteLength === 0 && !last) return;
    const checksum = await sha256(bytes);
    const seq = state.seq;
    state.seq += 1;
    state.pendingBytes += bytes.byteLength;
    if (state.pendingBytes > state.maxPendingBytes * 1.5) {
      throw new Error("Too much unsaved media in memory.");
    }
    state.queue.push({ seq, bytes, checksum, last });
    pump();
  }

  $("start").onclick = async () => {
    if (!state.stream || !state.audioTrack) {
      status("Choose a tab with audio first.");
      return;
    }
    try {
      state.mime = pickMime();
      await ensureSession();
      watchSession();
      state.seq = 0;
      state.finalized = false;
      state.stopping = false;
      const rec = new MediaRecorder(state.stream, { mimeType: state.mime, videoBitsPerSecond: 2_500_000 });
      state.recorder = rec;
      rec.ondataavailable = (ev) => {
        if (ev.data && ev.data.size) enqueue(ev.data, false).catch((err) => status(err.message));
      };
      rec.onerror = () => status("The recorder reported an error. Partial media was kept if anything was saved.");
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

  async function waitForUploads() {
    const start = Date.now();
    while ((state.queue.length || state.uploading) && Date.now() - start < 120000) {
      await pump();
      await new Promise((r) => setTimeout(r, 50));
    }
    if (state.queue.length) throw new Error("Not all chunks were acknowledged.");
  }

    async function finish(reason) {
    if (state.stopping || state.finalized) return;
    state.stopping = true;
    state.recording = false;
    $("stop").disabled = true;
    if (state.timer) clearInterval(state.timer);
    status("Saving…");
    const rec = state.recorder;
    await new Promise((resolve) => {
      if (!rec || rec.state === "inactive") {
        resolve();
        return;
      }
      rec.addEventListener("stop", () => resolve(), { once: true });
      try {
        rec.stop();
      } catch (err) {
        resolve();
      }
      setTimeout(resolve, 4000);
    });
    await new Promise((r) => setTimeout(r, 50));
    try {
      await waitForUploads();
      if (state.seq > 0) await enqueue(new Blob([new Uint8Array(0)]), true);
      await waitForUploads();
    } catch (err) {
      status(`Saving failed. Partial file was kept. ${err.message}`);
      stopTracks(state.stream);
      return;
    }
    status("Validating…");
    try {
      const result = await api(`/api/recordings/${state.recId}/finalize`, {
        method: "POST",
        headers: { ...headers(), "Content-Type": "application/json" },
        body: JSON.stringify({
          stop_reason: reason,
          mime_type: state.mime,
          audio_track: state.audioTrack,
          audio_detected: state.audioDetected,
          title: $("title").value,
          browser: { userAgent: navigator.userAgent, vendor: navigator.vendor },
        }),
      });
      state.finalized = true;
      stopTracks(state.stream);
      if (result.process) {
        status("Saved. Transcribing and screenshots continue in the VSL Study window. Open the evidence from there when it finishes.");
      } else {
        status("Partial recording saved. It was not treated as a complete VSL, so it was not analyzed automatically.");
      }
    } catch (err) {
      status(err.message || String(err));
      stopTracks(state.stream);
    }
  }

  async function abortLocal(reason) {
    if (state.finalized) return;
    state.recording = false;
    try {
      if (state.recorder && state.recorder.state !== "inactive") state.recorder.stop();
    } catch (err) {
      /* ignore */
    }
    stopTracks(state.stream);
    if (state.recId && reason !== "app_closed") {
      try {
        await api(`/api/recordings/${state.recId}/cancel`, { method: "POST", headers: headers(), body: "{}", });
      } catch (err) {
        /* ignore */
      }
    }
    state.stopping = false;
  }

  $("stop").onclick = () => finish("user_stop");
  window.addEventListener("beforeunload", () => {
    if (state.recording) {
      stopTracks(state.stream);
    }
  });
})();
