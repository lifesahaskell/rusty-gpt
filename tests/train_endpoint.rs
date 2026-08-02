//! `POST /api/train` behaviour, driven through the real router with a fake
//! training runner.
//!
//! These live in their own test binary on purpose: they set the
//! process-global `runtime_signals` interrupt flag, which would race the
//! training-loop unit tests if they shared a process. Within this binary they
//! serialize on [`RUN_LOCK`] for the same reason.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use axum::body::{Body, to_bytes};
use axum::http::{HeaderMap, Method, Request, StatusCode, header};
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use rusty_gpt::model::MiniGpt;
use rusty_gpt::observability::{EventLogger, LogFormat, RuntimeEvent};
use rusty_gpt::runtime_signals;
use rusty_gpt::server::{
    ServerLimits, ServerProvenance, ServerState, TrainRequest, TrainRunRecord, TrainRunStatus,
    TrainingRunOutcome, TrainingRunner, router_with_limits,
};
use rusty_gpt::tokenizer::RuntimeTokenizer;
use tower::ServiceExt;

type TestBackend = NdArray<f32, i64>;

static RUN_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Stands in for `ServerTrainingRunner`, mirroring the two behaviours the HTTP
/// layer depends on: it polls the interrupt flag at every step boundary, and
/// it emits the same `training_progress` / `checkpoint_saved` events the real
/// training loop does.
struct FakeRunner {
    device: NdArrayDevice,
    /// Wall-clock cost per step, so a test can interrupt a run in flight.
    step_delay: Duration,
    /// Vocab size of the model handed back, so a test can prove the swap.
    trained_vocab_size: usize,
    runs_started: Arc<AtomicUsize>,
}

impl TrainingRunner<TestBackend> for FakeRunner {
    fn run(
        &self,
        request: &TrainRequest,
        logger: &EventLogger,
    ) -> Result<TrainingRunOutcome<TestBackend>, String> {
        self.runs_started.fetch_add(1, Ordering::SeqCst);
        let mut steps_completed = 0;
        for step in 0..request.train_steps {
            if runtime_signals::interrupt_requested() {
                return Ok(TrainingRunOutcome {
                    model: MiniGpt::<TestBackend>::new(7, 8, 1, 6, 2, &self.device).into(),
                    steps_completed,
                    interrupted: true,
                });
            }
            std::thread::sleep(self.step_delay);
            steps_completed = step + 1;
            logger.log(RuntimeEvent::TrainingProgress {
                backend: "cpu".to_string(),
                model: "minigpt".to_string(),
                step,
                total_steps: request.train_steps,
                training_loss: 2.0,
                value_loss: 3.0,
                value_perplexity: 20.0,
                elapsed_ms: 1,
                tokens_per_second: 1.0,
                steps_per_second: 1.0,
                step_ms_mean: 1.0,
                learning_rate: request.learning_rate,
                aux_loss: None,
            });
        }
        logger.log(RuntimeEvent::CheckpointSaved {
            path: "/srv/rusty-gpt/checkpoints/mini_gpt.mpk".to_string(),
            elapsed_ms: 1,
        });
        logger.log(RuntimeEvent::TrainingCompleted {
            backend: "cpu".to_string(),
            model: "minigpt".to_string(),
            total_steps: request.train_steps,
            elapsed_ms: 1,
            final_value_loss: 3.0,
            final_perplexity: 20.0,
        });

        Ok(TrainingRunOutcome {
            model: MiniGpt::<TestBackend>::new(self.trained_vocab_size, 8, 1, 6, 2, &self.device)
                .into(),
            steps_completed,
            interrupted: false,
        })
    }
}

fn runs_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rusty-gpt-train-endpoint-{}-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn state_without_training(limits: ServerLimits) -> Arc<ServerState<TestBackend>> {
    let device = NdArrayDevice::Cpu;
    Arc::new(ServerState::new_with_limits(
        MiniGpt::<TestBackend>::new(7, 8, 1, 6, 2, &device),
        RuntimeTokenizer::char_from_text("abcdefg"),
        device,
        EventLogger::stdout(LogFormat::Plain),
        ServerProvenance::fresh(),
        limits,
    ))
}

fn state_with_runner(
    limits: ServerLimits,
    dir: PathBuf,
    runner: FakeRunner,
) -> Arc<ServerState<TestBackend>> {
    let device = NdArrayDevice::Cpu;
    Arc::new(
        ServerState::new_with_limits(
            MiniGpt::<TestBackend>::new(7, 8, 1, 6, 2, &device),
            RuntimeTokenizer::char_from_text("abcdefg"),
            device,
            EventLogger::stdout(LogFormat::Plain),
            ServerProvenance::fresh(),
            limits,
        )
        .with_training_runner(Arc::new(runner), dir),
    )
}

fn fast_runner(runs_started: Arc<AtomicUsize>) -> FakeRunner {
    FakeRunner {
        device: NdArrayDevice::Cpu,
        step_delay: Duration::from_millis(1),
        trained_vocab_size: 5,
        runs_started,
    }
}

fn slow_runner(runs_started: Arc<AtomicUsize>) -> FakeRunner {
    FakeRunner {
        device: NdArrayDevice::Cpu,
        step_delay: Duration::from_millis(20),
        trained_vocab_size: 5,
        runs_started,
    }
}

fn train_body(train_steps: usize) -> serde_json::Value {
    serde_json::json!({
        "train_steps": train_steps,
        "learning_rate": 1e-4,
        "checkpoint_interval": 100,
        "eval_interval": 100
    })
}

async fn post_json(
    state: Arc<ServerState<TestBackend>>,
    limits: ServerLimits,
    uri: &str,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value, HeaderMap) {
    let response = router_with_limits::<TestBackend>(limits)
        .with_state(state)
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri(uri)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, body, headers)
}

async fn get_json(
    state: Arc<ServerState<TestBackend>>,
    limits: ServerLimits,
    uri: &str,
) -> (StatusCode, serde_json::Value) {
    let response = router_with_limits::<TestBackend>(limits)
        .with_state(state)
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri(uri)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    // An unmatched route answers 404 with an empty body, which is not JSON.
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
    )
}

async fn get_status(
    state: &Arc<ServerState<TestBackend>>,
    limits: ServerLimits,
    run_id: &str,
) -> (StatusCode, serde_json::Value) {
    get_json(
        Arc::clone(state),
        limits,
        &format!("/train/{run_id}/status"),
    )
    .await
}

/// Poll the status endpoint itself — over HTTP, the way a UI does — until the
/// run reports `want`.
async fn poll_status_until(
    state: &Arc<ServerState<TestBackend>>,
    limits: ServerLimits,
    run_id: &str,
    want: &str,
) -> serde_json::Value {
    let mut last = serde_json::Value::Null;
    for _ in 0..600 {
        let (status, body) = get_status(state, limits, run_id).await;
        assert_eq!(StatusCode::OK, status, "status endpoint must stay readable");
        if body["status"] == want {
            return body;
        }
        last = body;
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("timed out waiting for status {want}; last body: {last}");
}

/// Poll the shared run status until `predicate` holds. Fails the test rather
/// than hanging if the run never gets there.
async fn wait_for(
    state: &Arc<ServerState<TestBackend>>,
    what: &str,
    predicate: impl Fn(&TrainRunRecord) -> bool,
) -> TrainRunRecord {
    for _ in 0..600 {
        if let Some(record) = state.training_run()
            && predicate(&record)
        {
            return record;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!(
        "timed out waiting for {what}; last record: {:?}",
        state.training_run()
    );
}

fn is_terminal(record: &TrainRunRecord) -> bool {
    record.status != TrainRunStatus::Running
}

#[tokio::test]
async fn interrupted_run_does_not_poison_the_next_run() {
    let _guard = RUN_LOCK.lock().await;
    let dir = runs_dir("reset");
    let limits = ServerLimits::default();
    let starts = Arc::new(AtomicUsize::new(0));
    let state = state_with_runner(limits, dir.clone(), slow_runner(Arc::clone(&starts)));

    let (status, body, _) =
        post_json(Arc::clone(&state), limits, "/train", train_body(1_000)).await;
    assert_eq!(StatusCode::ACCEPTED, status);
    let first_run_id = body["run_id"].as_str().unwrap().to_string();

    // Interrupt only once the loop is actually running, otherwise the
    // start-of-run reset would swallow the request.
    wait_for(&state, "the first run to make progress", |record| {
        record.steps_completed >= 1
    })
    .await;
    runtime_signals::request_interrupt();
    let interrupted = wait_for(&state, "the first run to stop", is_terminal).await;
    assert_eq!(TrainRunStatus::Interrupted, interrupted.status);
    assert_eq!(first_run_id, interrupted.run_id);
    assert!(runtime_signals::interrupt_requested());

    // The flag is still set process-wide. Without the reset at the top of each
    // run, this second run would die at its first step boundary.
    let (status, body, _) = post_json(Arc::clone(&state), limits, "/train", train_body(3)).await;
    assert_eq!(StatusCode::ACCEPTED, status);
    let second_run_id = body["run_id"].as_str().unwrap().to_string();
    assert_ne!(first_run_id, second_run_id);

    let completed = wait_for(&state, "the second run to finish", is_terminal).await;
    assert_eq!(
        TrainRunStatus::Completed,
        completed.status,
        "second run must not inherit the previous run's interrupt"
    );
    assert_eq!(3, completed.steps_completed);
    assert_eq!(2, starts.load(Ordering::SeqCst));

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn second_train_request_conflicts_with_the_active_run() {
    let _guard = RUN_LOCK.lock().await;
    let dir = runs_dir("conflict");
    let limits = ServerLimits::default();
    let starts = Arc::new(AtomicUsize::new(0));
    let state = state_with_runner(limits, dir.clone(), slow_runner(Arc::clone(&starts)));

    let (status, body, _) =
        post_json(Arc::clone(&state), limits, "/train", train_body(1_000)).await;
    assert_eq!(StatusCode::ACCEPTED, status);
    let active_run_id = body["run_id"].as_str().unwrap().to_string();
    wait_for(&state, "the run to make progress", |record| {
        record.steps_completed >= 1
    })
    .await;

    let (status, body, headers) =
        post_json(Arc::clone(&state), limits, "/train", train_body(5)).await;

    assert_eq!(StatusCode::CONFLICT, status);
    assert_eq!(
        serde_json::json!({"error":"run_in_progress","run_id":active_run_id}),
        body
    );
    assert!(headers.contains_key(header::RETRY_AFTER));
    assert_eq!(1, starts.load(Ordering::SeqCst), "no second run may start");

    runtime_signals::request_interrupt();
    wait_for(&state, "the run to stop", is_terminal).await;
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn generate_is_unavailable_while_a_run_is_active() {
    let _guard = RUN_LOCK.lock().await;
    let dir = runs_dir("generate-503");
    let limits = ServerLimits {
        rate_limit_rps: 0,
        ..ServerLimits::default()
    };
    let starts = Arc::new(AtomicUsize::new(0));
    let state = state_with_runner(limits, dir.clone(), slow_runner(Arc::clone(&starts)));

    let (status, _, _) = post_json(Arc::clone(&state), limits, "/train", train_body(1_000)).await;
    assert_eq!(StatusCode::ACCEPTED, status);
    wait_for(&state, "the run to make progress", |record| {
        record.steps_completed >= 1
    })
    .await;

    let (status, body, headers) = post_json(
        Arc::clone(&state),
        limits,
        "/generate",
        serde_json::json!({"prompt":"ab","max_tokens":1,"temperature":1.0,"top_k":null}),
    )
    .await;

    assert_eq!(StatusCode::SERVICE_UNAVAILABLE, status);
    assert_eq!(
        serde_json::json!({"error":"training_in_progress","retry_after_seconds":30}),
        body
    );
    assert_eq!(
        Some("30"),
        headers
            .get(header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
    );

    runtime_signals::request_interrupt();
    wait_for(&state, "the run to stop", is_terminal).await;

    // Generation works again once the run is over.
    let (status, _, _) = post_json(
        Arc::clone(&state),
        limits,
        "/generate",
        serde_json::json!({"prompt":"ab","max_tokens":1,"temperature":1.0,"top_k":null}),
    )
    .await;
    assert_eq!(StatusCode::OK, status);

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn completed_run_writes_its_manifest_and_swaps_the_served_model() {
    let _guard = RUN_LOCK.lock().await;
    let dir = runs_dir("manifest");
    let limits = ServerLimits::default();
    let starts = Arc::new(AtomicUsize::new(0));
    let state = state_with_runner(limits, dir.clone(), fast_runner(Arc::clone(&starts)));

    let (status, body, _) = post_json(Arc::clone(&state), limits, "/train", train_body(3)).await;
    assert_eq!(StatusCode::ACCEPTED, status);
    let run_id = body["run_id"].as_str().unwrap().to_string();

    let record = wait_for(&state, "the run to finish", is_terminal).await;
    assert_eq!(TrainRunStatus::Completed, record.status);

    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join(format!("run-{run_id}.json"))).expect("manifest must exist"),
    )
    .unwrap();

    assert_eq!(run_id, manifest["run_id"]);
    assert_eq!("completed", manifest["status"]);
    assert_eq!(3, manifest["steps_completed"]);
    assert_eq!(3, manifest["total_steps"]);
    assert_eq!(3.0, manifest["value_loss"]);
    assert_eq!(
        serde_json::json!(["mini_gpt.mpk"]),
        manifest["checkpoints"],
        "checkpoints are recorded as basenames so the status endpoint cannot leak paths"
    );
    assert_eq!(
        train_body(3)["train_steps"],
        manifest["request"]["train_steps"]
    );
    assert!(manifest["started_at_unix"].as_u64().unwrap() > 0);
    assert!(manifest["ended_at_unix"].as_u64().unwrap() > 0);
    assert!(manifest.get("error").is_none(), "no error key on success");

    // The freshly trained model is now the served one.
    let (_, info) = get_json(Arc::clone(&state), limits, "/info").await;
    assert_eq!(5, info["vocab_size"]);

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn train_rejects_requests_above_the_configured_caps() {
    let _guard = RUN_LOCK.lock().await;
    let dir = runs_dir("caps");
    let limits = ServerLimits {
        max_train_steps: 10,
        max_train_learning_rate: 1e-3,
        ..ServerLimits::default()
    };
    let starts = Arc::new(AtomicUsize::new(0));
    let state = state_with_runner(limits, dir.clone(), fast_runner(Arc::clone(&starts)));

    let (status, body, _) = post_json(Arc::clone(&state), limits, "/train", train_body(11)).await;
    assert_eq!(StatusCode::BAD_REQUEST, status);
    assert_eq!(
        serde_json::json!({"error":"train_steps_out_of_range","max_allowed":10,"requested":11}),
        body
    );

    let (status, body, _) = post_json(
        Arc::clone(&state),
        limits,
        "/train",
        serde_json::json!({
            "train_steps": 5,
            "learning_rate": 1.0,
            "checkpoint_interval": 0,
            "eval_interval": 0
        }),
    )
    .await;
    assert_eq!(StatusCode::BAD_REQUEST, status);
    assert_eq!("learning_rate_out_of_range", body["error"]);

    assert_eq!(0, starts.load(Ordering::SeqCst), "no run may start");
    assert!(state.training_run().is_none());
    assert!(!dir.exists(), "rejected requests write no manifest");
}

#[tokio::test]
async fn status_polls_a_run_through_to_completion() {
    let _guard = RUN_LOCK.lock().await;
    let dir = runs_dir("status-completed");
    let limits = ServerLimits::default();
    let starts = Arc::new(AtomicUsize::new(0));
    let state = state_with_runner(limits, dir.clone(), fast_runner(Arc::clone(&starts)));

    let (status, body, _) = post_json(Arc::clone(&state), limits, "/train", train_body(2)).await;
    assert_eq!(StatusCode::ACCEPTED, status);
    let run_id = body["run_id"].as_str().unwrap().to_string();

    let final_status = poll_status_until(&state, limits, &run_id, "completed").await;

    assert_eq!(run_id, final_status["run_id"]);
    assert_eq!(2, final_status["steps_completed"]);
    assert_eq!(2, final_status["total_steps"]);
    assert_eq!(3.0, final_status["value_loss"], "final metrics populate");
    assert_eq!(2.0, final_status["training_loss"]);
    assert!(final_status["ended_at_unix"].as_u64().unwrap() > 0);
    assert!(
        final_status["eta_seconds"].is_null(),
        "a finished run has nothing left to wait for"
    );
    assert_eq!(
        serde_json::json!(["mini_gpt.mpk"]),
        final_status["checkpoints"],
        "basenames only — the status endpoint discloses no paths"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn status_of_a_running_run_reports_eta_from_reported_throughput() {
    let _guard = RUN_LOCK.lock().await;
    let dir = runs_dir("status-eta");
    let limits = ServerLimits::default();
    let starts = Arc::new(AtomicUsize::new(0));
    let state = state_with_runner(limits, dir.clone(), slow_runner(Arc::clone(&starts)));

    let (status, body, _) =
        post_json(Arc::clone(&state), limits, "/train", train_body(1_000)).await;
    assert_eq!(StatusCode::ACCEPTED, status);
    let run_id = body["run_id"].as_str().unwrap().to_string();

    wait_for(&state, "the run to report progress", |record| {
        record.steps_completed >= 1
    })
    .await;
    let (status, running) = get_status(&state, limits, &run_id).await;

    assert_eq!(StatusCode::OK, status);
    assert_eq!("running", running["status"]);
    // The fake runner reports 1 step/s, so the ETA is just the steps left.
    let steps_completed = running["steps_completed"].as_u64().unwrap();
    assert_eq!(1.0, running["steps_per_second"]);
    assert_eq!(1_000 - steps_completed, running["eta_seconds"]);

    runtime_signals::request_interrupt();
    wait_for(&state, "the run to stop", is_terminal).await;
    let (_, stopped) = get_status(&state, limits, &run_id).await;
    assert_eq!("interrupted", stopped["status"]);
    assert!(stopped["eta_seconds"].is_null());

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn status_of_an_older_run_is_served_from_its_manifest() {
    let _guard = RUN_LOCK.lock().await;
    let dir = runs_dir("status-older");
    let limits = ServerLimits::default();
    let starts = Arc::new(AtomicUsize::new(0));
    let state = state_with_runner(limits, dir.clone(), fast_runner(Arc::clone(&starts)));

    let (_, body, _) = post_json(Arc::clone(&state), limits, "/train", train_body(2)).await;
    let first_run_id = body["run_id"].as_str().unwrap().to_string();
    poll_status_until(&state, limits, &first_run_id, "completed").await;

    // The in-memory slot holds one run; starting a second one overwrites it.
    let (_, body, _) = post_json(Arc::clone(&state), limits, "/train", train_body(3)).await;
    let second_run_id = body["run_id"].as_str().unwrap().to_string();
    assert_ne!(first_run_id, second_run_id);
    poll_status_until(&state, limits, &second_run_id, "completed").await;
    assert_eq!(
        second_run_id,
        state.training_run().unwrap().run_id,
        "the live slot has moved on to the second run"
    );

    let (status, older) = get_status(&state, limits, &first_run_id).await;

    assert_eq!(StatusCode::OK, status, "an older run is still readable");
    assert_eq!(first_run_id, older["run_id"]);
    assert_eq!("completed", older["status"]);
    assert_eq!(2, older["steps_completed"], "the first run's own progress");

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn status_of_an_unknown_run_is_not_found() {
    let _guard = RUN_LOCK.lock().await;
    let dir = runs_dir("status-404");
    let limits = ServerLimits::default();
    let starts = Arc::new(AtomicUsize::new(0));
    let state = state_with_runner(limits, dir.clone(), fast_runner(Arc::clone(&starts)));

    let missing = "6f1b6e6a-0000-4000-8000-000000000000";
    let (status, body) = get_status(&state, limits, missing).await;

    assert_eq!(StatusCode::NOT_FOUND, status);
    assert_eq!(
        serde_json::json!({"error":"run_not_found","run_id":missing}),
        body
    );

    // The id becomes part of a filename, and axum percent-decodes path params
    // — `%2f` really does arrive as a `/`. Anything that is not a run id this
    // server minted is refused before it reaches the filesystem.
    for (hostile, decoded) in [
        ("not-a-uuid", "not-a-uuid"),
        ("..%2f..%2fetc%2fpasswd", "../../etc/passwd"),
        ("%2e%2e%2fsecrets", "../secrets"),
    ] {
        let (status, body) = get_status(&state, limits, hostile).await;
        assert_eq!(StatusCode::NOT_FOUND, status, "rejected: {hostile}");
        assert_eq!(
            serde_json::json!({"error":"run_not_found","run_id":decoded}),
            body,
            "the handler rejected it, not the router: {hostile}"
        );
    }

    assert_eq!(0, starts.load(Ordering::SeqCst));
    assert!(!dir.exists(), "reading status creates nothing");
}

#[tokio::test]
async fn train_is_unavailable_without_a_runner() {
    let limits = ServerLimits::default();
    let state = state_without_training(limits);

    let (status, body, _) = post_json(state, limits, "/train", train_body(5)).await;

    assert_eq!(StatusCode::SERVICE_UNAVAILABLE, status);
    assert_eq!("training_unavailable", body["error"]);
}
