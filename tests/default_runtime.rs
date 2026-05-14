use std::process::Command;

#[test]
fn default_runtime_uses_cpu_and_does_not_touch_cuda() {
    let output = Command::new(env!("CARGO_BIN_EXE_rusty-gpt"))
        .args(["--input", "tests/fixtures/input.txt"])
        .env("RUSTY_GPT_TRAIN_STEPS", "1")
        .env("RUSTY_GPT_BPE_TOKENIZER", "tests/fixtures/tokenizer.json")
        .output()
        .expect("binary should run");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}{stderr}");

    assert!(
        output.status.success(),
        "binary failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        stdout,
        stderr
    );
    assert!(stdout.contains("Configured app: backend=cpu"));
    assert!(stdout.contains("Prepared runtime batch:"));
    assert!(stdout.contains("Configured app: backend=cpu, model=minigpt"));
    assert!(stdout.contains("minigpt forward pass:"));
    assert!(!combined.contains("libcuda"));
    assert!(!combined.contains("RecvError"));
    assert!(!combined.contains("panicked"));
}
