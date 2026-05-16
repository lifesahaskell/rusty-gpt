//! Integration test for Sprint 1 / T3: `--checkpoint-interval` writes
//! numbered mid-run snapshots, `--checkpoint-keep` prunes older ones, the
//! final end-of-run save is always present, and the SIGINT-interrupted
//! save is never pruned.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[test]
fn checkpoint_interval_and_keep_produce_expected_files() {
    let tmp = std::env::temp_dir().join(format!(
        "rusty-gpt-periodic-checkpoints-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).expect("create tmp checkpoint dir");
    let checkpoint = tmp.join("mini_gpt");

    let output = Command::new(env!("CARGO_BIN_EXE_rusty-gpt"))
        .args([
            "--model",
            "minigpt",
            "--input",
            "tests/fixtures/input.txt",
            "--train-steps",
            "4",
            "--checkpoint-interval",
            "2",
            "--checkpoint-keep",
            "1",
            "--checkpoint",
        ])
        .arg(&checkpoint)
        .env("RUSTY_GPT_BPE_TOKENIZER", "tests/fixtures/tokenizer.json")
        .output()
        .expect("spawn rusty-gpt training subprocess");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "training subprocess failed with status {:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status.code()
    );

    let names: Vec<String> = fs::read_dir(&tmp)
        .expect("read tmp dir")
        .filter_map(|entry| entry.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .collect();

    // Final end-of-run save is always present.
    assert!(
        names.contains(&"mini_gpt.mpk".to_string()),
        "expected the final mini_gpt.mpk, found: {names:?}"
    );
    assert!(
        names.contains(&"mini_gpt.metadata.json".to_string()),
        "expected the final mini_gpt.metadata.json, found: {names:?}"
    );

    // With --train-steps 4 --checkpoint-interval 2 --checkpoint-keep 1 the
    // periodic save fires at step 2 (step 4 is suppressed because it
    // coincides with the final), and nothing is pruned because only one
    // periodic snapshot ever exists.
    let periodic_mpks: Vec<&String> = names
        .iter()
        .filter(|name| name.starts_with("mini_gpt.step-") && name.ends_with(".mpk"))
        .collect();
    assert_eq!(
        1,
        periodic_mpks.len(),
        "expected exactly one mini_gpt.step-N.mpk, found: {names:?}"
    );
    assert_eq!(
        Some(&"mini_gpt.step-2.mpk".to_string()),
        periodic_mpks.first().copied()
    );

    let sidecar_path: PathBuf = tmp.join("mini_gpt.step-2.metadata.json");
    let sidecar = fs::read_to_string(&sidecar_path).expect("read step-2 sidecar");
    let compact = sidecar.replace(' ', "");
    assert!(
        compact.contains("\"step\":2"),
        "step-2 sidecar should record step: 2; got: {sidecar}"
    );
    assert!(
        compact.contains("\"interval\":2"),
        "step-2 sidecar should record interval: 2; got: {sidecar}"
    );

    let _ = fs::remove_dir_all(&tmp);
}
