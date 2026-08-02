//! Process-wide interrupt flag for graceful shutdown of training runs.
//!
//! A signal handler runs in a background thread and sets [`INTERRUPT_REQUESTED`]
//! when SIGINT or SIGTERM arrives. The training loop polls
//! [`interrupt_requested`] at every step boundary; on the first interrupt it
//! breaks out cleanly so the caller can save a partial checkpoint. A second
//! interrupt within ~2 seconds bypasses the save and exits immediately with
//! exit code 130 — the operator escape hatch.
//!
//! The handler is only installed on the **training** path; serving and
//! interactive inference inherit the default Ctrl-C behaviour (process abort).
//!
//! On non-Unix targets [`install_training_signal_handler`] is a no-op and
//! [`interrupt_requested`] always returns `false`, so the rest of the codebase
//! does not have to special-case Windows.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Exit code used when training was interrupted but a partial checkpoint was
/// saved successfully. Matches the shell convention `128 + SIGINT(2)`.
pub const INTERRUPTED_EXIT_CODE: i32 = 130;

/// Window in milliseconds in which a second interrupt aborts immediately
/// without saving.
const SECOND_INTERRUPT_GRACE_MS: u64 = 2_000;

static INTERRUPT_REQUESTED: AtomicBool = AtomicBool::new(false);
/// Wall-clock millis (since [`StartMonotonic`]) when the first interrupt
/// arrived. `0` means "no interrupt yet" — we offset by `+1` when storing so
/// that an interrupt arriving at process millisecond 0 is still
/// distinguishable from the sentinel.
static FIRST_INTERRUPT_AT_MS_PLUS_ONE: AtomicU64 = AtomicU64::new(0);

/// True if a SIGINT or SIGTERM has been observed and the training loop should
/// stop at the next step boundary.
pub fn interrupt_requested() -> bool {
    INTERRUPT_REQUESTED.load(Ordering::SeqCst)
}

/// Install a signal handler thread that flips [`interrupt_requested`] on
/// SIGINT or SIGTERM. Idempotent — calling more than once is a no-op.
///
/// On non-Unix targets this is a no-op so the binary still builds on Windows.
#[cfg(unix)]
pub fn install_training_signal_handler() -> anyhow::Result<()> {
    use signal_hook::consts::signal::{SIGINT, SIGTERM};
    use signal_hook::iterator::Signals;
    use std::sync::Once;

    static INSTALL: Once = Once::new();
    let mut err: Option<anyhow::Error> = None;
    INSTALL.call_once(|| match install_handler_thread() {
        Ok(()) => {}
        Err(e) => err = Some(e),
    });
    if let Some(e) = err {
        return Err(e);
    }

    fn install_handler_thread() -> anyhow::Result<()> {
        let mut signals = Signals::new([SIGINT, SIGTERM])?;
        let start = std::time::Instant::now();
        std::thread::spawn(move || {
            for sig in signals.forever() {
                let now_ms = start.elapsed().as_millis() as u64;
                // Reserve 0 as the "no interrupt yet" sentinel.
                let now_marker = now_ms.saturating_add(1);
                let previous = FIRST_INTERRUPT_AT_MS_PLUS_ONE.swap(now_marker, Ordering::SeqCst);
                if previous == 0 {
                    // First interrupt — request a clean stop at the next
                    // step boundary.
                    INTERRUPT_REQUESTED.store(true, Ordering::SeqCst);
                    eprintln!(
                        "\nInterrupt received (signal {sig}); saving checkpoint at the next step boundary. Send the signal again within {}s to abort immediately.",
                        SECOND_INTERRUPT_GRACE_MS / 1000
                    );
                } else {
                    let previous_ms = previous.saturating_sub(1);
                    let elapsed_ms = now_ms.saturating_sub(previous_ms);
                    if elapsed_ms <= SECOND_INTERRUPT_GRACE_MS {
                        eprintln!(
                            "Second interrupt within {}s — exiting immediately without saving.",
                            SECOND_INTERRUPT_GRACE_MS / 1000
                        );
                        std::process::exit(INTERRUPTED_EXIT_CODE);
                    } else {
                        // The previous interrupt is stale (e.g. the loop is
                        // still inside the slow save). Treat this as a fresh
                        // request.
                        INTERRUPT_REQUESTED.store(true, Ordering::SeqCst);
                    }
                }
            }
        });
        Ok(())
    }

    Ok(())
}

#[cfg(not(unix))]
pub fn install_training_signal_handler() -> anyhow::Result<()> {
    // Non-Unix targets keep the default Ctrl-C behaviour; the training loop
    // will simply never observe an interrupt request from this module.
    Ok(())
}

/// Request a clean stop at the next training step boundary, exactly as a
/// SIGINT would. Used by the HTTP stop path and by tests; the signal handler
/// thread sets the same flag directly (it also records the timestamp that
/// drives the second-interrupt escape hatch, which this function deliberately
/// does not touch — a programmatic stop must never abort the process).
pub fn request_interrupt() {
    INTERRUPT_REQUESTED.store(true, Ordering::SeqCst);
}

/// Clear the interrupt flag and the first-interrupt timestamp.
///
/// [`INTERRUPT_REQUESTED`] is process-global and the HTTP server is
/// long-lived, so every server-initiated training run must call this before
/// entering its step loop. Without it, one interrupted or stopped run leaves
/// the flag set forever and every later run on the same process dies at its
/// first step boundary, silently, until the process restarts.
pub fn reset_interrupt() {
    INTERRUPT_REQUESTED.store(false, Ordering::SeqCst);
    FIRST_INTERRUPT_AT_MS_PLUS_ONE.store(0, Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_and_reset_toggle_the_interrupt_flag() {
        reset_interrupt();
        assert!(!interrupt_requested());
        request_interrupt();
        assert!(interrupt_requested());
        reset_interrupt();
        assert!(!interrupt_requested());
    }
}
