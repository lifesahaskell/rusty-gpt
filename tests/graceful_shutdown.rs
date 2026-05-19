//! Integration test for Sprint 1 / T2: SIGINT during a MiniGPT training run
//! saves a partial checkpoint at `<checkpoint>.interrupted-step-<N>.mpk` and
//! exits with code 130.

#![cfg(unix)]

use nix::sys::signal::{Signal, kill};
use nix::unistd::Pid;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Substring printed at the very start of `train_model`'s MiniGPT branch. By
/// the time this line hits the subprocess's stdout, `run_training_demo` has
/// already installed the SIGINT/SIGTERM handler — so it's a reliable barrier
/// for "the handler is now live, you may signal me."
const TRAINING_STARTED_MARKER: &str = "Training minigpt model";

#[test]
#[ignore = "signal delivery is flaky under cargo test; run manually when changing runtime_signals"]
fn sigint_during_training_saves_interrupted_checkpoint() {
    let tmp = PathBuf::from("checkpoints").join(format!(
        "rusty-gpt-graceful-shutdown-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).expect("create tmp checkpoint dir");
    let checkpoint = tmp.join("mini_gpt");

    // Many train steps give the test enough room to interrupt even with the
    // tiny test-only model shape below, while each step is still fast enough
    // to reach the next signal-poll boundary quickly.
    let mut child = Command::new(env!("CARGO_BIN_EXE_rusty-gpt"))
        .args([
            "--model",
            "minigpt",
            "--input",
            "tests/fixtures/input.txt",
            "--block-size",
            "8",
            "--batch-size",
            "1",
            "--embed-dim",
            "8",
            "--num-heads",
            "2",
            "--num-layers",
            "1",
            "--checkpoint",
        ])
        .arg(&checkpoint)
        .env("RUSTY_GPT_TRAIN_STEPS", "100000")
        .env("RUSTY_GPT_BPE_TOKENIZER", "tests/fixtures/tokenizer.json")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn rusty-gpt training subprocess");

    let child_pid = Pid::from_raw(child.id() as i32);

    // Stream stdout line-by-line in a background thread so we can detect
    // `TRAINING_STARTED_MARKER` in real time. The full buffer is preserved
    // so we can include it in failure messages.
    let stdout = child.stdout.take().expect("child stdout");
    let stdout_buf = Arc::new(Mutex::new(String::new()));
    let training_started = Arc::new(AtomicBool::new(false));
    let stdout_buf_clone = stdout_buf.clone();
    let training_started_clone = training_started.clone();
    let stdout_thread = thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines().map_while(Result::ok) {
            if line.contains(TRAINING_STARTED_MARKER) {
                training_started_clone.store(true, Ordering::SeqCst);
            }
            let mut buf = stdout_buf_clone.lock().unwrap();
            buf.push_str(&line);
            buf.push('\n');
        }
    });

    let stderr = child.stderr.take().expect("child stderr");
    let stderr_buf = Arc::new(Mutex::new(String::new()));
    let stderr_buf_clone = stderr_buf.clone();
    let stderr_thread = thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines().map_while(Result::ok) {
            let mut buf = stderr_buf_clone.lock().unwrap();
            buf.push_str(&line);
            buf.push('\n');
        }
    });

    // Wait for the marker (60s ceiling — debug-build CI startup can be
    // surprisingly slow). Polling sleep keeps this race-free.
    let marker_deadline = Instant::now() + Duration::from_secs(60);
    while !training_started.load(Ordering::SeqCst) {
        if Instant::now() >= marker_deadline {
            let _ = child.kill();
            let _ = stdout_thread.join();
            let _ = stderr_thread.join();
            let stdout_text = stdout_buf.lock().unwrap().clone();
            let stderr_text = stderr_buf.lock().unwrap().clone();
            panic!(
                "subprocess never printed {TRAINING_STARTED_MARKER:?} within 60s\nstdout:\n{stdout_text}\nstderr:\n{stderr_text}"
            );
        }
        thread::sleep(Duration::from_millis(100));
    }

    kill(child_pid, Signal::SIGINT).expect("send SIGINT to child");

    // Give the handler time to save the checkpoint and exit.
    let exit_deadline = Instant::now() + Duration::from_secs(120);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() >= exit_deadline => {
                let _ = child.kill();
                panic!("training subprocess did not exit within 120s of SIGINT");
            }
            Ok(None) => thread::sleep(Duration::from_millis(250)),
            Err(e) => panic!("failed to poll child status: {e}"),
        }
    };

    let _ = stdout_thread.join();
    let _ = stderr_thread.join();
    let stdout_text = stdout_buf.lock().unwrap().clone();
    let stderr_text = stderr_buf.lock().unwrap().clone();

    assert_eq!(
        Some(130),
        status.code(),
        "expected exit code 130 (interrupted + saved), got {code:?}\nstdout:\n{stdout_text}\nstderr:\n{stderr_text}",
        code = status.code()
    );

    let interrupted_files: Vec<PathBuf> = fs::read_dir(&tmp)
        .expect("read tmp dir")
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("mini_gpt.interrupted-step-") && n.ends_with(".mpk"))
                .unwrap_or(false)
        })
        .collect();

    assert!(
        !interrupted_files.is_empty(),
        "expected at least one mini_gpt.interrupted-step-*.mpk under {tmp:?}, found {entries:?}",
        tmp = tmp,
        entries = fs::read_dir(&tmp)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
            .collect::<Vec<_>>()
    );

    let sidecar_file = interrupted_files[0].with_extension("metadata.json");
    assert!(
        sidecar_file.exists(),
        "expected metadata sidecar at {sidecar_file:?}"
    );
    let sidecar = fs::read_to_string(&sidecar_file).expect("read sidecar");
    let interrupted_marker = sidecar.replace(' ', "");
    assert!(
        interrupted_marker.contains("\"interrupted\":true"),
        "metadata sidecar should mark interrupted=true; got: {sidecar}"
    );
    assert!(
        interrupted_marker.contains("\"interrupted_at_step\":"),
        "metadata sidecar should include interrupted_at_step; got: {sidecar}"
    );

    let _ = fs::remove_dir_all(&tmp);
}
