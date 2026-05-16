//! Integration test for Sprint 1 / T2: SIGINT during a MiniGPT training run
//! saves a partial checkpoint at `<checkpoint>.interrupted-step-<N>.mpk` and
//! exits with code 130.

#![cfg(unix)]

use nix::sys::signal::{Signal, kill};
use nix::unistd::Pid;
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

#[test]
fn sigint_during_training_saves_interrupted_checkpoint() {
    let tmp = std::env::temp_dir().join(format!(
        "rusty-gpt-graceful-shutdown-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).expect("create tmp checkpoint dir");
    let checkpoint = tmp.join("mini_gpt");

    // 1000 train_steps is plenty of room — each MiniGPT step on CPU takes
    // ~15+s, so we are guaranteed to interrupt mid-run when we kill after
    // the first observed step boundary log.
    let mut child = Command::new(env!("CARGO_BIN_EXE_rusty-gpt"))
        .args([
            "--model",
            "minigpt",
            "--input",
            "tests/fixtures/input.txt",
            "--checkpoint",
        ])
        .arg(&checkpoint)
        .env("RUSTY_GPT_TRAIN_STEPS", "1000")
        .env("RUSTY_GPT_BPE_TOKENIZER", "tests/fixtures/tokenizer.json")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn rusty-gpt training subprocess");

    let child_pid = Pid::from_raw(child.id() as i32);

    // Wait for training to actually start (handler is installed at the top of
    // run_training_demo, before the "Training" log line is emitted). Once we
    // observe that line, the signal handler is guaranteed to be live.
    let mut stdout = child.stdout.take().expect("child stdout");
    let stdout_buf = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let stdout_buf_clone = stdout_buf.clone();
    let stdout_thread = thread::spawn(move || {
        use std::io::Read;
        let mut local = String::new();
        let _ = stdout.read_to_string(&mut local);
        *stdout_buf_clone.lock().unwrap() = local;
    });

    let mut stderr = child.stderr.take().expect("child stderr");
    let stderr_buf = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let stderr_buf_clone = stderr_buf.clone();
    let stderr_thread = thread::spawn(move || {
        use std::io::Read;
        let mut local = String::new();
        let _ = stderr.read_to_string(&mut local);
        *stderr_buf_clone.lock().unwrap() = local;
    });

    // Wait long enough for training to have started (tokenizer + data load
    // can take a few seconds; first step alone is ~15s on CPU). 8s lands
    // mid-first-step and gives the handler ample time to install.
    thread::sleep(Duration::from_secs(8));
    kill(child_pid, Signal::SIGINT).expect("send SIGINT to child");

    // Give the handler time to save the checkpoint and exit.
    let deadline = Instant::now() + Duration::from_secs(120);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                panic!("training subprocess did not exit within 120s of SIGINT");
            }
            Ok(None) => thread::sleep(Duration::from_millis(250)),
            Err(e) => panic!("failed to poll child status: {e}"),
        }
    };

    // Drain pipe readers so we can include their contents in failure messages.
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
