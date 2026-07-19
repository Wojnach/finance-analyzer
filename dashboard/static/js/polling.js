/*
 * polling.js — interval registry, visibility-aware.
 *
 * Each task: { name, intervalMs, fn, lastFiredAt }. Tasks pause when
 * `document.visibilityState === "hidden"` and fire once on resume.
 * View modules should `register()` on mount and `unregister()` on unmount
 * so closed views don't keep polling.
 *
 * `fn()` should return a truthy value when its run produced at least one
 * non-null fetch result — that's the signal LAST_REFRESH stamps on. A
 * falsy/omitted return is treated as "nothing refreshed" (see each view's
 * poll callback: `return d != null;`).
 *
 * Track-6 spec: per-section cadences, no aggregate ticker.
 */

import * as state from "./state.js";

const _tasks = new Map(); // name -> { intervalMs, fn, timer, runningP }
let _paused = false;       // global pause via UI toggle
let _hidden = false;       // tab not visible

/**
 * Register a polling task. Fires once immediately, then every intervalMs.
 * Calling register() again with the same name updates the interval/fn
 * (and resets the schedule). Returns an unregister function.
 */
export function register(name, intervalMs, fn) {
  unregister(name);
  const task = { intervalMs, fn, timer: null, runningP: null, gen: 0 };
  _tasks.set(name, task);
  _fire(name); // initial fire
  return () => unregister(name);
}

export function unregister(name) {
  const task = _tasks.get(name);
  if (!task) return;
  if (task.timer) clearTimeout(task.timer);
  _tasks.delete(name);
}

export function unregisterAll() {
  for (const name of [..._tasks.keys()]) unregister(name);
}

export function setPaused(paused) {
  _paused = !!paused;
  state.set(state.Slots.PAUSED, _paused);
  if (!_paused && !_hidden) {
    // Resume — fire each task once
    for (const name of _tasks.keys()) _fire(name);
  }
}

export function isPaused() { return _paused; }

/** Fire a specific task immediately (e.g. pull-to-refresh). */
export function fireNow(name) {
  if (_tasks.has(name)) _fire(name);
}

/** Fire all tasks immediately. */
export function fireAll() {
  for (const name of _tasks.keys()) _fire(name);
}

// ---------------------------------------------------------------------------
// Visibility hook
// ---------------------------------------------------------------------------

if (typeof document !== "undefined") {
  document.addEventListener("visibilitychange", () => {
    _hidden = document.visibilityState === "hidden";
    if (!_hidden && !_paused) {
      // Page resumed — fire each task once and re-schedule
      for (const name of _tasks.keys()) _fire(name);
    }
  });
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

function _fire(name) {
  const task = _tasks.get(name);
  if (!task) return;

  if (task.timer) {
    clearTimeout(task.timer);
    task.timer = null;
  }

  // Sequence token for this run. If a newer _fire() lands (manual refresh,
  // visibility resume, ...) before this one resolves, its token goes stale —
  // a stale run must not stamp LAST_REFRESH or reschedule the timer, since
  // whichever run holds the current token already did (or will).
  const myGen = ++task.gen;

  const run = async () => {
    if (_paused || _hidden) {
      // Skip while suppressed — no rescheduling here. Resume hooks will refire.
      return;
    }
    let ok = false;
    try {
      task.runningP = Promise.resolve(task.fn());
      ok = await task.runningP;
      // "last refresh Xs ago" must reflect the last SUCCESS (task.fn()
      // returns a falsy/null result on failure), not merely the last
      // attempt, and only from the latest-issued run for this task.
      if (ok && task.gen === myGen) {
        state.set(state.Slots.LAST_REFRESH, Date.now());
      }
    } catch (e) {
      console.warn(`polling task "${name}" threw`, e);
    } finally {
      task.runningP = null;
      // Reschedule — only the latest-issued run arms the next timer, else
      // overlapping runs would each arm their own and the task free-runs
      // faster than intervalMs.
      if (_tasks.get(name) === task && task.gen === myGen) {
        task.timer = setTimeout(() => _fire(name), task.intervalMs);
      }
    }
  };
  // Run async without awaiting — return immediately to caller.
  void run();
}
