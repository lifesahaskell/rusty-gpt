//! Integration test for Sprint 3 / T2: `--resume-from` continues MiniGPT
//! training from a saved checkpoint's `completed_steps`, performing only the
//! remaining steps to reach the absolute `--train-steps` target while keeping
//! the step counter continuous. A mismatched model shape on resume fails fast
//! with the strict metadata loader's diff-style error.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Shared tiny-model training flags so the fresh run and the resume run build
/// the same shape (the strict loader rejects any mismatch).
const SHAPE_ARGS: &[&str] = &[
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
];

fn run_training(extra: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_rusty-gpt"))
        .args(SHAPE_ARGS)
        .args(extra)
        .env("RUSTY_GPT_BPE_TOKENIZER", "tests/fixtures/tokenizer.json")
        .output()
        .expect("spawn rusty-gpt training subprocess")
}

/// Pull the 0-indexed `step` value out of every `training_progress` JSON line.
/// serde_json is not a dev-dependency here, so parse the compact JSON by hand.
fn logged_step_indices(stdout: &str) -> Vec<usize> {
    stdout
        .lines()
        .filter(|line| line.contains(r#""event":"training_progress""#))
        .filter_map(|line| {
            let marker = r#""step":"#;
            let start = line.find(marker)? + marker.len();
            let digits: String = line[start..]
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            digits.parse().ok()
        })
        .collect()
}

fn completed_steps_in_sidecar(sidecar: &Path) -> usize {
    let raw = fs::read_to_string(sidecar).expect("read metadata sidecar");
    let compact = raw.replace([' ', '\n'], "");
    let marker = r#""completed_steps":"#;
    let start = compact
        .find(marker)
        .expect("sidecar records completed_steps")
        + marker.len();
    let digits: String = compact[start..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().expect("completed_steps is an integer")
}

#[test]
fn resume_continues_step_count_and_produces_loadable_checkpoint() {
    // N = 2. Fresh run trains 2 steps, resume run targets 2N = 4, so it must
    // run exactly N = 2 more steps (absolute indices 2 and 3).
    const N: usize = 2;

    let tmp = PathBuf::from("checkpoints").join(format!("rusty-gpt-resume-{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).expect("create tmp checkpoint dir");
    let checkpoint = tmp.join("mini_gpt");
    let sidecar = tmp.join("mini_gpt.metadata.json");

    // --- Fresh run: N steps ---
    let fresh = run_training(&[
        "--train-steps",
        "2",
        "--log-format",
        "json",
        "--checkpoint",
        checkpoint.to_str().unwrap(),
    ]);
    assert!(
        fresh.status.success(),
        "fresh training failed: {}",
        String::from_utf8_lossy(&fresh.stderr)
    );
    assert_eq!(
        N,
        completed_steps_in_sidecar(&sidecar),
        "fresh run should record completed_steps == N"
    );

    // --- Resume run: absolute target 2N ---
    let resumed = run_training(&[
        "--train-steps",
        "4",
        "--log-format",
        "json",
        "--checkpoint",
        checkpoint.to_str().unwrap(),
        "--resume-from",
        checkpoint.to_str().unwrap(),
    ]);
    let stdout = String::from_utf8_lossy(&resumed.stdout);
    let stderr = String::from_utf8_lossy(&resumed.stderr);
    assert!(
        resumed.status.success(),
        "resume training failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );

    // Weights were loaded through the strict loader before training.
    assert!(
        stdout.contains(r#""event":"checkpoint_loaded""#),
        "resume run should load the checkpoint before training; stdout:\n{stdout}"
    );

    // The resumed run performs exactly N more steps, continuing the counter:
    // absolute (0-indexed) steps 2 and 3 — never re-running steps 0 or 1.
    // Step index 2 is the (N+1)-th training step, the resume boundary.
    let steps = logged_step_indices(&stdout);
    assert_eq!(
        vec![N, N + 1],
        steps,
        "resume should log absolute steps [N, 2N-1] and no earlier steps"
    );
    assert_eq!(
        Some(&N),
        steps.first(),
        "first logged resumed step is index N (the (N+1)-th step)"
    );

    // Absolute target reached — completed_steps is 2N, not N + 2N.
    assert_eq!(
        2 * N,
        completed_steps_in_sidecar(&sidecar),
        "resume should reach the absolute --train-steps target"
    );

    // --- Final checkpoint loads and generates ---
    let mut child = Command::new(env!("CARGO_BIN_EXE_rusty-gpt"))
        .args(SHAPE_ARGS)
        .args([
            "--interactive-generate",
            "--load-checkpoint",
            "--checkpoint",
            checkpoint.to_str().unwrap(),
        ])
        .env("RUSTY_GPT_BPE_TOKENIZER", "tests/fixtures/tokenizer.json")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn interactive generation");
    // A prompt then an empty line to exit the loop cleanly.
    child
        .stdin
        .take()
        .expect("interactive stdin")
        .write_all(b"The\n\n")
        .expect("write interactive prompt");
    let gen_output = child
        .wait_with_output()
        .expect("await interactive generation");
    let gen_stdout = String::from_utf8_lossy(&gen_output.stdout);
    assert!(
        gen_output.status.success(),
        "loading + generating from the resumed checkpoint failed: {}",
        String::from_utf8_lossy(&gen_output.stderr)
    );
    assert!(
        gen_stdout.contains("Loaded minigpt checkpoint"),
        "resumed checkpoint should load for generation; stdout:\n{gen_stdout}"
    );

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn resume_with_mismatched_shape_fails_with_strict_loader_error() {
    let tmp = PathBuf::from("checkpoints")
        .join(format!("rusty-gpt-resume-mismatch-{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).expect("create tmp checkpoint dir");
    let checkpoint = tmp.join("mini_gpt");

    // Fresh run with embed-dim 8.
    let fresh = run_training(&[
        "--train-steps",
        "2",
        "--checkpoint",
        checkpoint.to_str().unwrap(),
    ]);
    assert!(
        fresh.status.success(),
        "fresh training failed: {}",
        String::from_utf8_lossy(&fresh.stderr)
    );

    // Resume asking for embed-dim 16 — a shape mismatch the strict metadata
    // loader must reject with its diff-style expected/found message.
    let mismatch = Command::new(env!("CARGO_BIN_EXE_rusty-gpt"))
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
            "16",
            "--num-heads",
            "2",
            "--num-layers",
            "1",
            "--train-steps",
            "4",
            "--checkpoint",
            checkpoint.to_str().unwrap(),
            "--resume-from",
            checkpoint.to_str().unwrap(),
        ])
        .env("RUSTY_GPT_BPE_TOKENIZER", "tests/fixtures/tokenizer.json")
        .output()
        .expect("spawn mismatched resume");

    assert!(
        !mismatch.status.success(),
        "resume with a mismatched shape must fail"
    );
    let stderr = String::from_utf8_lossy(&mismatch.stderr);
    assert!(
        stderr.contains("model_shape.embed_dim expected 16, found 8"),
        "expected a diff-style shape-mismatch error; stderr:\n{stderr}"
    );

    let _ = fs::remove_dir_all(&tmp);
}
