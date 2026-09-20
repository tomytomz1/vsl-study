(function (root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  root.VslCapture = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function defaultSha256(bytes) {
    var subtle = globalThis.crypto && globalThis.crypto.subtle;
    if (!subtle) {
      return Promise.reject(new Error("SHA-256 is not available."));
    }
    var view = bytes.buffer
      ? bytes
      : new Uint8Array(bytes);
    return subtle.digest("SHA-256", view).then(function (hash) {
      return Array.from(new Uint8Array(hash))
        .map(function (b) {
          return b.toString(16).padStart(2, "0");
        })
        .join("");
    });
  }

  function CaptureError(code, message) {
    var err = new Error(message);
    err.code = code;
    return err;
  }

  function CapturePipeline(options) {
    options = options || {};
    this.uploadChunk = options.uploadChunk;
    this.sha256 = options.sha256 || defaultSha256;
    this.retryDelay = options.retryDelay || function (attempt) {
      return 400 * attempt;
    };
    this.maxPendingBytes = options.maxPendingBytes || 12 * 1024 * 1024;
    this.stopTimeoutMs = options.stopTimeoutMs || 30000;
    this.onStatus = options.onStatus || function () {};
    this.nextSeq = 0;
    this.outstandingBytes = 0;
    this.chain = Promise.resolve();
    this.failed = false;
    this.failReason = null;
    this.failError = null;
    this.finishing = false;
    this.finishPromise = null;
    this.uploaded = [];
    this.ackedSeqs = [];
    this.finalizeCalls = 0;
    this.cancelCalls = 0;
    this.incomplete = false;
    this.preparing = 0;
    this.mediaClosed = false;
  }

  CapturePipeline.prototype.acceptMedia = function (blob) {
    if (this.failed || this.mediaClosed) {
      return;
    }
    if (!blob || !blob.size) {
      return;
    }
    if (this.outstandingBytes + blob.size > this.maxPendingBytes) {
      this._fail("backpressure", "Saving cannot keep up. Partial media already saved was kept.");
      return;
    }
    var seq = this.nextSeq;
    this.nextSeq += 1;
    this.outstandingBytes += blob.size;
    this.preparing += 1;
    this.onStatus({ accepted: seq, outstandingBytes: this.outstandingBytes, preparing: this.preparing });
    this._enqueueWork(seq, blob, false);
  };

  CapturePipeline.prototype._enqueueWork = function (seq, blob, last) {
    var self = this;
    this.chain = this.chain
      .then(function () {
        return self._runItem(seq, blob, last);
      })
      .catch(function (err) {
        self._fail((err && err.code) || "upload_failed", (err && err.message) || String(err));
      });
  };

  CapturePipeline.prototype._runItem = function (seq, blob, last) {
    var self = this;
    if (self.failed) {
      return Promise.resolve();
    }
    var size = blob && blob.size ? blob.size : 0;
    return Promise.resolve()
      .then(function () {
        if (!blob || !blob.size) {
          return new Uint8Array(0);
        }
        return blob.arrayBuffer().then(function (buf) {
          return buf instanceof Uint8Array ? buf : new Uint8Array(buf);
        });
      })
      .then(function (bytes) {
        if (!last) {
          self.preparing = Math.max(0, self.preparing - 1);
        }
        if (self.failed) {
          return;
        }
        return self.sha256(bytes).then(function (checksum) {
          if (self.failed) {
            return false;
          }
          return self._uploadWithRetry(seq, bytes, checksum, last).then(function () {
            return true;
          });
        });
      })
      .then(function (uploaded) {
        if (!uploaded || self.failed) {
          return;
        }
        self.outstandingBytes = Math.max(0, self.outstandingBytes - size);
        self.ackedSeqs.push(seq);
        self.onStatus({ savedThrough: seq, last: last, outstandingBytes: self.outstandingBytes });
      });
  };

  CapturePipeline.prototype._uploadWithRetry = function (seq, bytes, checksum, last) {
    var self = this;
    var attempt = 0;
    function once() {
      return Promise.resolve()
        .then(function () {
          return self.uploadChunk({ seq: seq, bytes: bytes, checksum: checksum, last: last });
        })
        .then(function () {
          self.uploaded.push({ seq: seq, byteLength: bytes.byteLength, last: last });
        })
        .catch(function (err) {
          if (err && (err.code === "conflict" || err.code === "finalized" || err.code === "cancelled" || err.fatal)) {
            throw err;
          }
          attempt += 1;
          if (attempt >= 5) {
            throw err;
          }
          var wait = self.retryDelay(attempt);
          return new Promise(function (resolve) {
            setTimeout(resolve, wait);
          }).then(once);
        });
    }
    return once();
  };

  CapturePipeline.prototype._fail = function (code, message) {
    if (this.failed) {
      return;
    }
    this.failed = true;
    this.incomplete = true;
    this.failReason = code;
    this.failError = CaptureError(code, message);
    this.onStatus({ failed: true, code: code, message: message });
  };

  CapturePipeline.prototype.waitUntilIdle = function () {
    var self = this;
    return this.chain.then(function () {
      if (self.failed) {
        throw self.failError;
      }
    });
  };

  CapturePipeline.prototype.finish = function (reason, hooks) {
    if (this.finishPromise) {
      return this.finishPromise;
    }
    this.finishing = true;
    this.finishPromise = this._finish(reason, hooks || {});
    return this.finishPromise;
  };

  CapturePipeline.prototype._awaitStop = function (stopRecorder) {
    var self = this;
    var timer;
    return new Promise(function (resolve, reject) {
      timer = setTimeout(function () {
        reject(CaptureError("stop_timeout", "The recorder did not stop in time. Partial media was kept."));
      }, self.stopTimeoutMs);
      Promise.resolve()
        .then(stopRecorder)
        .then(resolve)
        .catch(reject);
    }).finally(function () {
      if (timer) {
        clearTimeout(timer);
      }
    });
  };

  CapturePipeline.prototype._emptyBlob = function () {
    return {
      size: 0,
      arrayBuffer: function () {
        return Promise.resolve(new Uint8Array(0).buffer);
      },
    };
  };

  CapturePipeline.prototype._finish = function (reason, hooks) {
    var self = this;
    var stopRecorder = hooks.stopRecorder;
    var finalizeRemote = hooks.finalizeRemote;
    var cancelRemote = hooks.cancelRemote;
    var failCodes = { upload_failed: true, backpressure: true, recorder_error: true, stop_timeout: true };
    var finalizeReasons = { user_stop: true, max_duration: true, trailing_silence: true, stop_sharing: true };
    var processReasons = { user_stop: true, max_duration: true, trailing_silence: true };

    function cancelAndStop() {
      self.cancelCalls += 1;
      var tasks = [];
      if (stopRecorder) {
        tasks.push(
          Promise.resolve()
            .then(stopRecorder)
            .catch(function () {})
        );
      }
      if (cancelRemote) {
        tasks.push(
          Promise.resolve()
            .then(function () {
              return cancelRemote(self.failReason || reason);
            })
            .catch(function () {})
        );
      }
      return Promise.all(tasks).then(function () {
        return {
          process: false,
          complete: false,
          reason: self.failReason || reason,
          error: self.failError,
        };
      });
    }

    if (self.failed && failCodes[self.failReason]) {
      return cancelAndStop();
    }

    return Promise.resolve()
      .then(function () {
        if (!stopRecorder) {
          self.mediaClosed = true;
          return;
        }
        return self._awaitStop(stopRecorder).then(function () {
          self.mediaClosed = true;
        });
      })
      .then(function () {
        if (self.failed) {
          return cancelAndStop();
        }
        var lastSeq = self.nextSeq;
        self.nextSeq += 1;
        self._enqueueWork(lastSeq, self._emptyBlob(), true);
        return self.waitUntilIdle();
      })
      .then(function (early) {
        if (early && early.process === false) {
          return early;
        }
        if (self.failed) {
          return cancelAndStop();
        }
        if (!finalizeReasons[reason] || !finalizeRemote) {
          if (!finalizeReasons[reason] && cancelRemote) {
            return cancelAndStop();
          }
          return {
            process: Boolean(processReasons[reason]),
            complete: Boolean(processReasons[reason]),
            reason: reason,
          };
        }
        self.finalizeCalls += 1;
        return Promise.resolve(finalizeRemote(reason));
      })
      .catch(function (err) {
        self._fail((err && err.code) || "upload_failed", (err && err.message) || String(err));
        return cancelAndStop();
      });
  };

  function parsePageGeneration(search) {
    var params;
    try {
      params = new URLSearchParams(search || "");
    } catch (err) {
      return null;
    }
    var raw = params.get("g");
    if (raw == null || raw === "") {
      return null;
    }
    var value = Number(raw);
    if (!isFinite(value) || value < 1 || Math.floor(value) !== value) {
      return null;
    }
    return value;
  }

  var STALE_PAGE_MESSAGE = "This recording session has ended. Open a new recorder from VSL Study.";

  function RecorderPage(options) {
    options = options || {};
    this.generation = options.generation != null ? options.generation : null;
    this.ended = false;
    this.recId = null;
    this.message = "";
    this.tracksStopped = 0;
    this.timersCleared = 0;
    this.controlsLocked = false;
    this.unhandled = [];
    this.onStopTracks = options.onStopTracks || function () {};
    this.onClearTimers = options.onClearTimers || function () {};
    this.onLockControls = options.onLockControls || function () {};
    this.onStatus = options.onStatus || function () {};
  }

  RecorderPage.prototype.canMutate = function () {
    return !this.ended && this.generation != null;
  };

  RecorderPage.prototype.markEnded = function (message) {
    if (this.ended) {
      return { already: true, message: this.message };
    }
    this.ended = true;
    this.controlsLocked = true;
    this.message = message || STALE_PAGE_MESSAGE;
    this.timersCleared += 1;
    this.tracksStopped += 1;
    try {
      this.onClearTimers();
    } catch (err) {
      this.unhandled.push(err);
    }
    try {
      this.onStopTracks();
    } catch (err) {
      this.unhandled.push(err);
    }
    try {
      this.onLockControls();
    } catch (err) {
      this.unhandled.push(err);
    }
    try {
      this.onStatus(this.message);
    } catch (err) {
      this.unhandled.push(err);
    }
    return { already: false, message: this.message };
  };

  RecorderPage.prototype.bindRecording = function (id) {
    if (!this.canMutate()) {
      return false;
    }
    this.recId = id;
    return true;
  };

  RecorderPage.prototype.handleApiError = function (err) {
    var code = err && err.code;
    if (code === "stale_generation" || code === "superseded") {
      this.markEnded((err && err.message) || STALE_PAGE_MESSAGE);
      return true;
    }
    return false;
  };

  RecorderPage.prototype.handleSessionPoll = function (payload) {
    payload = payload || {};
    if (payload.cancelled || payload.stale) {
      this.markEnded(payload.message || STALE_PAGE_MESSAGE);
      return true;
    }
    if (payload.current_generation != null && Number(payload.current_generation) !== Number(this.generation)) {
      this.markEnded(STALE_PAGE_MESSAGE);
      return true;
    }
    return false;
  };

  var STUDY_CAPTURE_PROFILE = {
    id: "study-1280-10",
    targetWidth: 1280,
    targetFrameRate: 10,
    videoBitsPerSecond: 1200000,
  };

  function studyDisplayMediaOptions() {
    return {
      video: {
        displaySurface: "browser",
        width: { max: STUDY_CAPTURE_PROFILE.targetWidth, ideal: STUDY_CAPTURE_PROFILE.targetWidth },
        frameRate: { max: STUDY_CAPTURE_PROFILE.targetFrameRate, ideal: STUDY_CAPTURE_PROFILE.targetFrameRate },
      },
      audio: { suppressLocalAudioPlayback: false },
      preferCurrentTab: true,
      selfBrowserSurface: "exclude",
      surfaceSwitching: "exclude",
      monitorTypeSurfaces: "exclude",
    };
  }

  function studyTrackConstraints() {
    return {
      width: { max: STUDY_CAPTURE_PROFILE.targetWidth, ideal: STUDY_CAPTURE_PROFILE.targetWidth },
      frameRate: { max: STUDY_CAPTURE_PROFILE.targetFrameRate, ideal: STUDY_CAPTURE_PROFILE.targetFrameRate },
    };
  }

  function readTrackSettings(track) {
    try {
      return track && typeof track.getSettings === "function" ? track.getSettings() || {} : {};
    } catch (err) {
      return {};
    }
  }

  function settingsMatchProfile(settings, profile) {
    var width = Number(settings && settings.width);
    var fps = Number(settings && settings.frameRate);
    if (!Number.isFinite(width) || width > profile.targetWidth + 2) {
      return false;
    }
    if (!Number.isFinite(fps) || fps > profile.targetFrameRate + 0.51) {
      return false;
    }
    return true;
  }

  function applyStudyVideoConstraints(track, canMutate, profile) {
    profile = profile || STUDY_CAPTURE_PROFILE;
    var report = {
      profileId: profile.id,
      requested: {
        width: profile.targetWidth,
        frameRate: profile.targetFrameRate,
        videoBitsPerSecond: profile.videoBitsPerSecond,
      },
      before: readTrackSettings(track),
      after: null,
      constraintApplied: false,
      constraintError: null,
      matchesProfile: false,
      stale: false,
    };
    if (!track) {
      report.constraintError = "No video track";
      report.after = {};
      return Promise.resolve(report);
    }
    if (typeof track.applyConstraints !== "function") {
      report.constraintError = "applyConstraints is not supported";
      report.after = report.before;
      report.matchesProfile = settingsMatchProfile(report.after, profile);
      return Promise.resolve(report);
    }
    return Promise.resolve()
      .then(function () {
        return track.applyConstraints(studyTrackConstraints());
      })
      .then(function () {
        if (typeof canMutate === "function" && !canMutate()) {
          report.stale = true;
          report.after = readTrackSettings(track);
          return report;
        }
        report.constraintApplied = true;
        report.after = readTrackSettings(track);
        report.matchesProfile = settingsMatchProfile(report.after, profile);
        return report;
      })
      .catch(function (err) {
        report.constraintError = String((err && (err.message || err.name)) || err);
        report.after = readTrackSettings(track);
        report.matchesProfile = settingsMatchProfile(report.after, profile);
        return report;
      });
  }

  var TRAILING_SILENCE_MS = 180000;

  function trailingSilenceShouldStop(opts) {
    opts = opts || {};
    if (!opts.recording || opts.finishing) {
      return false;
    }
    if (!opts.audioHeard) {
      return false;
    }
    if (opts.lastAudibleAt == null) {
      return false;
    }
    var now = opts.now != null ? opts.now : Date.now();
    var threshold = opts.thresholdMs != null ? Number(opts.thresholdMs) : TRAILING_SILENCE_MS;
    return now - Number(opts.lastAudibleAt) >= threshold;
  }

  function averageFrequencyLevel(data) {
    if (!data || !data.length) {
      return 0;
    }
    var sum = 0;
    for (var i = 0; i < data.length; i += 1) {
      sum += data[i];
    }
    return sum / data.length / 255;
  }

  return {
    CapturePipeline: CapturePipeline,
    CaptureError: CaptureError,
    RecorderPage: RecorderPage,
    parsePageGeneration: parsePageGeneration,
    STUDY_CAPTURE_PROFILE: STUDY_CAPTURE_PROFILE,
    studyDisplayMediaOptions: studyDisplayMediaOptions,
    applyStudyVideoConstraints: applyStudyVideoConstraints,
    settingsMatchProfile: settingsMatchProfile,
    trailingSilenceShouldStop: trailingSilenceShouldStop,
    averageFrequencyLevel: averageFrequencyLevel,
    TRAILING_SILENCE_MS: TRAILING_SILENCE_MS,
  };
});
