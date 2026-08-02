//! `POST /api/train` — kick off a MiniGPT training run in the background;
//! `GET /api/train/{run_id}/status` — the polling endpoint that reads it back;
//! `DELETE /api/train/{run_id}` — stop it at the next step boundary.
//!
//! The HTTP layer owns run bookkeeping only: admission control (one run at a
//! time), the run manifest on disk, and the shared status other handlers read.
//! The training itself is behind the [`TrainingRunner`] trait so this module
//! stays free of backend/autodiff plumbing — the binary supplies the real
//! implementation (`runtime_training::ServerTrainingRunner`), tests supply a
//! fake.
//!
//! Lifecycle of one run:
//!
//! 1. Handler validates the payload against [`ServerLimits`], takes the
//!    single run slot, writes `checkpoints/runs/run-<uuid>.json`, and returns
//!    `202 Accepted` with the `run_id`.
//! 2. A `spawn_blocking` task resets the process-global interrupt flag, then
//!    calls the runner with a JSON [`EventLogger`] whose sink parses each
//!    rendered event and writes progress through to the manifest.
//! 3. On a clean finish the freshly trained model replaces the served one.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::extract::{Path as PathParam, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};

use super::{ServedModel, ServerLimits, SharedServerState};
use crate::observability::{EventLogger, LogFormat};
use crate::runtime_signals;

/// Directory holding one `run-<uuid>.json` manifest per training run.
pub const DEFAULT_RUNS_DIR: &str = "checkpoints/runs";

/// Route-local body cap. The payload is a handful of numbers plus an optional
/// checkpoint name; anything larger is not a train request.
pub(super) const TRAIN_REQUEST_BODY_LIMIT_BYTES: usize = 4096;

/// `Retry-After` advertised by `POST /api/generate` while a run is active.
/// ponytail: a fixed guess. Estimating it from step throughput needs the
/// status endpoint's data model, which is a follow-up task.
pub(super) const TRAINING_RETRY_AFTER_SECONDS: u64 = 30;

/// Request body of `POST /api/train`.
///
/// `resume_from` is named to match the `--resume-from` CLI flag and
/// `RUSTY_GPT_RESUME_FROM`: a checkpoint path without `.mpk`, resolved and
/// confined to `checkpoints/` by the runner exactly like the CLI does.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct TrainRequest {
    pub train_steps: usize,
    pub learning_rate: f64,
    pub checkpoint_interval: usize,
    pub eval_interval: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resume_from: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct TrainAccepted {
    pub run_id: String,
}

/// Body of `GET /api/train/{run_id}/status`: the run manifest verbatim, plus
/// the one field that is a function of it rather than stored in it. Flattened
/// on purpose — the manifest field names are already the contract, and a
/// second set of wire names for the same data would only drift.
#[derive(Debug, Serialize)]
pub struct TrainStatusResponse {
    #[serde(flatten)]
    pub record: TrainRunRecord,
    /// Seconds of work left at the last reported throughput. `null` unless the
    /// run is still going and has reported at least one progress event.
    pub eta_seconds: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainRunStatus {
    Running,
    Completed,
    /// The step loop stopped early on SIGINT/SIGTERM (or a programmatic
    /// [`runtime_signals::request_interrupt`]). A partial checkpoint was saved.
    Interrupted,
    Failed,
}

/// On-disk shape of `checkpoints/runs/run-<uuid>.json`, and the in-memory
/// status other handlers read. Serialized as-is — treat the field names as a
/// contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainRunRecord {
    pub run_id: String,
    pub status: TrainRunStatus,
    pub request: TrainRequest,
    pub started_at_unix: u64,
    pub ended_at_unix: Option<u64>,
    /// Steps completed as of the last `training_progress` event (absolute, so
    /// it continues across a `resume_from` boundary). `0` until the first one.
    pub steps_completed: usize,
    /// Absolute step target of the run, echoed from the training events.
    pub total_steps: usize,
    pub training_loss: Option<f64>,
    pub value_loss: Option<f64>,
    /// Throughput as of the last `training_progress` event — the single source
    /// the status endpoint's `eta_seconds` is derived from, so the API and the
    /// logs can never disagree about how fast a run is going. `default` so
    /// manifests written before this field existed still deserialize.
    #[serde(default)]
    pub steps_per_second: Option<f64>,
    /// Every checkpoint the run wrote, in the order the `checkpoint_saved`
    /// events arrived: periodic `.step-N.mpk` snapshots first, then the final
    /// (or `.interrupted-step-N.mpk`) save.
    pub checkpoints: Vec<String>,
    /// Set only when `status` is `failed`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl TrainRunRecord {
    fn started(run_id: String, request: TrainRequest) -> Self {
        Self {
            run_id,
            status: TrainRunStatus::Running,
            total_steps: request.train_steps,
            request,
            started_at_unix: unix_now(),
            ended_at_unix: None,
            steps_completed: 0,
            training_loss: None,
            value_loss: None,
            steps_per_second: None,
            checkpoints: Vec::new(),
            error: None,
        }
    }

    fn is_running(&self) -> bool {
        self.status == TrainRunStatus::Running
    }
}

/// What a [`TrainingRunner`] hands back to the HTTP layer.
pub struct TrainingRunOutcome<B: Backend> {
    /// The trained model, ready to serve. Swapped in only when
    /// `interrupted` is false.
    pub model: ServedModel<B>,
    pub steps_completed: usize,
    pub interrupted: bool,
}

/// The training backend behind `POST /api/train`.
///
/// Implementations run synchronously on a blocking thread and must poll
/// [`runtime_signals::interrupt_requested`] at step boundaries (the shared
/// training loop already does) so a SIGINT stops the run and saves a partial
/// checkpoint.
pub trait TrainingRunner<B: Backend>: Send + Sync + 'static {
    fn run(
        &self,
        request: &TrainRequest,
        logger: &EventLogger,
    ) -> Result<TrainingRunOutcome<B>, String>;
}

/// Run bookkeeping owned by `ServerState`. Only one run may be active; the
/// slot then keeps that run's final record so a status endpoint can read it
/// (older runs are on disk under [`DEFAULT_RUNS_DIR`]).
pub struct TrainingState<B: Backend> {
    runner: Option<Arc<dyn TrainingRunner<B>>>,
    handle: RunHandle,
}

impl<B: Backend> Default for TrainingState<B> {
    fn default() -> Self {
        Self {
            runner: None,
            handle: RunHandle::new(PathBuf::from(DEFAULT_RUNS_DIR)),
        }
    }
}

impl<B: Backend> TrainingState<B> {
    pub(super) fn with_runner(runner: Arc<dyn TrainingRunner<B>>, runs_dir: PathBuf) -> Self {
        Self {
            runner: Some(runner),
            handle: RunHandle::new(runs_dir),
        }
    }

    pub(super) fn current(&self) -> Option<TrainRunRecord> {
        self.handle.snapshot()
    }

    pub(super) fn is_running(&self) -> bool {
        self.handle
            .snapshot()
            .map(|record| record.is_running())
            .unwrap_or(false)
    }
}

/// Shared, write-through handle to the current run record.
#[derive(Clone)]
struct RunHandle {
    runs_dir: PathBuf,
    record: Arc<Mutex<Option<TrainRunRecord>>>,
}

impl RunHandle {
    fn new(runs_dir: PathBuf) -> Self {
        Self {
            runs_dir,
            record: Arc::new(Mutex::new(None)),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Option<TrainRunRecord>> {
        // A panicking training task must not wedge the endpoint: the record is
        // a plain data struct, so the worst a poisoned lock hides is a
        // half-updated progress field.
        self.record
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn snapshot(&self) -> Option<TrainRunRecord> {
        self.lock().clone()
    }

    /// Mutate the current record and write the manifest back to disk.
    /// Manifest write failures are reported but never abort a run in flight.
    ///
    /// The write stays inside the critical section on purpose: anyone who
    /// observes a status through [`RunHandle::snapshot`] must find a manifest
    /// on disk that agrees with it. Writing after the guard drops lets a
    /// reader see `completed` in memory and then read a truncated file.
    fn update(&self, mutate: impl FnOnce(&mut TrainRunRecord)) {
        let mut guard = self.lock();
        let Some(record) = guard.as_mut() else {
            return;
        };
        mutate(record);
        if let Err(err) = write_manifest(&self.runs_dir, record) {
            eprintln!(
                "training run {}: failed to update manifest: {err}",
                record.run_id
            );
        }
    }
}

fn manifest_path(runs_dir: &Path, run_id: &str) -> PathBuf {
    runs_dir.join(format!("run-{run_id}.json"))
}

fn write_manifest(runs_dir: &Path, record: &TrainRunRecord) -> std::io::Result<()> {
    std::fs::create_dir_all(runs_dir)?;
    let body = serde_json::to_vec_pretty(record).map_err(std::io::Error::other)?;
    std::fs::write(manifest_path(runs_dir, &record.run_id), body)
}

/// `Ok(None)` means no such run; a manifest that exists but cannot be read is
/// a server fault, not a 404 — during an incident "no such run" for a run
/// whose file is plainly on disk is the wrong thing to tell an operator.
fn read_manifest(runs_dir: &Path, run_id: &str) -> Result<Option<TrainRunRecord>, TrainError> {
    let body = match std::fs::read(manifest_path(runs_dir, run_id)) {
        Ok(body) => body,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(TrainError::Internal(format!(
                "failed to read run manifest: {err}"
            )));
        }
    };
    serde_json::from_slice(&body)
        .map(Some)
        .map_err(|err| TrainError::Internal(format!("run manifest could not be parsed: {err}")))
}

/// Work left over throughput, both as the training loop last reported them.
/// `None` once the run is terminal (nothing left to wait for) or before the
/// first progress event has given us a rate.
fn eta_seconds(record: &TrainRunRecord) -> Option<u64> {
    if !record.is_running() {
        return None;
    }
    let steps_per_second = record.steps_per_second.filter(|rate| *rate > 0.0)?;
    let remaining = record.total_steps.saturating_sub(record.steps_completed);
    Some((remaining as f64 / steps_per_second).round() as u64)
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

/// `POST /api/train`.
///
/// SECURITY: unauthenticated by design for now — anyone who can reach this
/// route can spend the box's GPU/CPU and overwrite `checkpoints/mini_gpt`.
/// Bind to localhost only (the `--server-addr` default) until auth lands.
pub(super) async fn train<B>(
    State(state): State<SharedServerState<B>>,
    Json(request): Json<TrainRequest>,
) -> Result<(StatusCode, Json<TrainAccepted>), TrainError>
where
    B: Backend + Send + Sync + 'static,
    B::Device: Send + Sync + 'static,
{
    validate_request(&request, state.limits)?;
    let runner = state
        .training
        .runner
        .clone()
        .ok_or(TrainError::TrainingUnavailable)?;

    let handle = state.training.handle.clone();
    let run_id = uuid::Uuid::new_v4().to_string();
    let record = {
        // Admission control and slot capture must be one critical section, or
        // two simultaneous requests both see "idle" and both start a run.
        let mut guard = handle.lock();
        if let Some(active) = guard.as_ref().filter(|record| record.is_running()) {
            return Err(TrainError::RunInProgress {
                run_id: active.run_id.clone(),
            });
        }
        let record = TrainRunRecord::started(run_id.clone(), request.clone());
        *guard = Some(record.clone());
        record
    };
    // The manifest is a contract other tooling reads; if we cannot write it,
    // fail the request rather than run a job nobody can observe.
    if let Err(err) = write_manifest(&handle.runs_dir, &record) {
        *handle.lock() = None;
        return Err(TrainError::Internal(format!(
            "failed to write run manifest: {err}"
        )));
    }

    let task_state = Arc::clone(&state);
    let task_run_id = run_id.clone();
    // Training is CPU/GPU-bound and fully synchronous — it must not sit on an
    // async worker thread.
    tokio::task::spawn_blocking(move || {
        run_training(task_state, runner, handle, request, task_run_id);
    });

    Ok((StatusCode::ACCEPTED, Json(TrainAccepted { run_id })))
}

/// `GET /api/train/{run_id}/status`.
///
/// The in-memory slot only ever holds the most recently started run, so an
/// older run's status comes from the manifest [`train`] already writes on
/// every progress update — same data, no second source of truth.
///
/// Not rate limited: the limiter is invoked from inside `POST /api/generate`
/// rather than as middleware, so a read-only route is exempt by construction.
/// A UI polling this every second must not burn the generate budget.
pub(super) async fn status<B>(
    State(state): State<SharedServerState<B>>,
    PathParam(run_id): PathParam<String>,
) -> Result<Json<TrainStatusResponse>, TrainError>
where
    B: Backend + Send + Sync + 'static,
{
    // Run ids are UUIDs this server minted, and this one is about to become
    // part of a filename. Rejecting anything else settles the traversal
    // question up front instead of relying on the `run-` prefix to blunt it.
    if uuid::Uuid::parse_str(&run_id).is_err() {
        return Err(TrainError::RunNotFound { run_id });
    }

    let handle = &state.training.handle;
    let live = handle.snapshot().filter(|record| record.run_id == run_id);
    let record = match live {
        Some(record) => Some(record),
        None => read_manifest(&handle.runs_dir, &run_id)?,
    };
    let Some(record) = record else {
        return Err(TrainError::RunNotFound { run_id });
    };

    Ok(Json(TrainStatusResponse {
        eta_seconds: eta_seconds(&record),
        record,
    }))
}

/// `DELETE /api/train/{run_id}` — stop the active run at its next step
/// boundary, exactly as a SIGINT would: the loop finishes the step it is on,
/// saves an `.interrupted-step-<N>` checkpoint, and the run lands on
/// [`TrainRunStatus::Interrupted`]. There is no separate "stopped" state — a
/// programmatic stop and a signal are the same thing to the training loop.
///
/// Anything that is not the currently running run — unknown ID, an ID that
/// belongs to an earlier run, or the active run after it already finished —
/// is a `404`. Stopping is idempotent for free: the run stays `running` until
/// the background task reaches its next boundary, so a repeat `DELETE` in
/// that window re-sets an already-set flag and answers `202` again.
///
/// SECURITY: unauthenticated, like the rest of `/api/train`. The `run_id`
/// match is not authorization — it only keeps a stale client from stopping a
/// run it never started.
pub(super) async fn stop_train<B>(
    State(state): State<SharedServerState<B>>,
    PathParam(run_id): PathParam<String>,
) -> StatusCode
where
    B: Backend,
{
    let stoppable = state
        .training
        .current()
        .is_some_and(|record| record.run_id == run_id && record.is_running());
    if !stoppable {
        return StatusCode::NOT_FOUND;
    }

    runtime_signals::request_interrupt();
    StatusCode::ACCEPTED
}

fn run_training<B>(
    state: SharedServerState<B>,
    runner: Arc<dyn TrainingRunner<B>>,
    handle: RunHandle,
    request: TrainRequest,
    run_id: String,
) where
    B: Backend + Send + Sync + 'static,
    B::Device: Send + Sync + 'static,
{
    // The interrupt flag is process-global and this server outlives any single
    // run: without this reset, one stopped run would kill every later run at
    // its first step boundary. See `runtime_signals::reset_interrupt`.
    runtime_signals::reset_interrupt();

    let progress = handle.clone();
    // Same capture-and-parse shape as the generate-lifecycle logging test:
    // render events as JSON, read the fields back off each line.
    let logger = EventLogger::with_sink(LogFormat::Json, move |line| {
        apply_event_line(&progress, &line);
        // Keep the operator's console informed; the run's events are JSON even
        // when the server itself logs plain text.
        println!("{line}");
    });

    match runner.run(&request, &logger) {
        Ok(outcome) => {
            if outcome.interrupted {
                handle.update(|record| {
                    record.status = TrainRunStatus::Interrupted;
                    record.ended_at_unix = Some(unix_now());
                    record.steps_completed = outcome.steps_completed;
                });
                return;
            }
            // Never hold the model lock across training — swap only here, once
            // the new weights are final.
            state.replace_model(outcome.model);
            handle.update(|record| {
                record.status = TrainRunStatus::Completed;
                record.ended_at_unix = Some(unix_now());
                record.steps_completed = outcome.steps_completed;
            });
        }
        Err(err) => {
            eprintln!("training run {run_id} failed: {err}");
            handle.update(|record| {
                record.status = TrainRunStatus::Failed;
                record.ended_at_unix = Some(unix_now());
                record.error = Some(err);
            });
        }
    }
}

/// Fold one rendered JSON event line into the run record. Unknown or
/// unparseable lines are ignored — the log is a progress feed, not a protocol.
fn apply_event_line(handle: &RunHandle, line: &str) {
    let Ok(event) = serde_json::from_str::<serde_json::Value>(line) else {
        return;
    };
    match event["event"].as_str() {
        Some("training_progress") => handle.update(|record| {
            // `step` is the 0-indexed step just finished.
            if let Some(step) = event["step"].as_u64() {
                record.steps_completed = step as usize + 1;
            }
            if let Some(total) = event["total_steps"].as_u64() {
                record.total_steps = total as usize;
            }
            record.training_loss = event["training_loss"].as_f64();
            record.value_loss = event["value_loss"].as_f64();
            record.steps_per_second = event["steps_per_second"].as_f64();
        }),
        Some("training_completed") => handle.update(|record| {
            if let Some(total) = event["total_steps"].as_u64() {
                record.total_steps = total as usize;
            }
            if let Some(loss) = event["final_value_loss"].as_f64() {
                record.value_loss = Some(loss);
            }
        }),
        Some("checkpoint_saved") => handle.update(|record| {
            // Basename only. This record is meant to be served over HTTP by
            // the status endpoint, and `/api/health` already establishes that
            // the API never discloses absolute paths. Every checkpoint lives
            // in `checkpoints/`, so the name is enough to find it.
            if let Some(name) = event["path"]
                .as_str()
                .map(Path::new)
                .and_then(Path::file_name)
                .and_then(|name| name.to_str())
            {
                record.checkpoints.push(name.to_string());
            }
        }),
        _ => {}
    }
}

fn validate_request(request: &TrainRequest, limits: ServerLimits) -> Result<(), TrainError> {
    if request.train_steps == 0 || request.train_steps > limits.max_train_steps {
        return Err(TrainError::TrainStepsOutOfRange {
            max_allowed: limits.max_train_steps,
            requested: request.train_steps,
        });
    }
    if !request.learning_rate.is_finite()
        || request.learning_rate <= 0.0
        || request.learning_rate > limits.max_train_learning_rate
    {
        return Err(TrainError::LearningRateOutOfRange {
            max_allowed: limits.max_train_learning_rate,
            requested: request.learning_rate,
        });
    }
    Ok(())
}

#[derive(Debug, Serialize)]
#[serde(tag = "error", rename_all = "snake_case")]
enum TrainErrorBody {
    TrainStepsOutOfRange {
        max_allowed: usize,
        requested: usize,
    },
    LearningRateOutOfRange {
        max_allowed: f64,
        requested: f64,
    },
    RunInProgress {
        run_id: String,
    },
    RunNotFound {
        run_id: String,
    },
    TrainingUnavailable {
        message: String,
    },
    Internal {
        message: String,
    },
}

#[derive(Debug)]
pub enum TrainError {
    TrainStepsOutOfRange {
        max_allowed: usize,
        requested: usize,
    },
    LearningRateOutOfRange {
        max_allowed: f64,
        requested: f64,
    },
    /// A run is already active. Only one at a time, process-wide.
    RunInProgress {
        run_id: String,
    },
    /// No live run and no manifest for this id.
    RunNotFound {
        run_id: String,
    },
    /// The server was started without a training runner (e.g. `--serve
    /// --model moe-gpt`).
    TrainingUnavailable,
    Internal(String),
}

impl TrainError {
    fn status(&self) -> StatusCode {
        match self {
            Self::TrainStepsOutOfRange { .. } | Self::LearningRateOutOfRange { .. } => {
                StatusCode::BAD_REQUEST
            }
            Self::RunInProgress { .. } => StatusCode::CONFLICT,
            Self::RunNotFound { .. } => StatusCode::NOT_FOUND,
            Self::TrainingUnavailable => StatusCode::SERVICE_UNAVAILABLE,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn body(self) -> TrainErrorBody {
        match self {
            Self::TrainStepsOutOfRange {
                max_allowed,
                requested,
            } => TrainErrorBody::TrainStepsOutOfRange {
                max_allowed,
                requested,
            },
            Self::LearningRateOutOfRange {
                max_allowed,
                requested,
            } => TrainErrorBody::LearningRateOutOfRange {
                max_allowed,
                requested,
            },
            Self::RunInProgress { run_id } => TrainErrorBody::RunInProgress { run_id },
            Self::RunNotFound { run_id } => TrainErrorBody::RunNotFound { run_id },
            Self::TrainingUnavailable => TrainErrorBody::TrainingUnavailable {
                message: "training is not enabled on this server".to_string(),
            },
            Self::Internal(message) => TrainErrorBody::Internal { message },
        }
    }
}

impl IntoResponse for TrainError {
    fn into_response(self) -> Response {
        let status = self.status();
        let retry_after = matches!(self, Self::TrainingUnavailable | Self::RunInProgress { .. })
            .then_some(TRAINING_RETRY_AFTER_SECONDS);
        let mut response = (status, Json(self.body())).into_response();
        if let Some(seconds) = retry_after
            && let Ok(value) = HeaderValue::from_str(&seconds.to_string())
        {
            response.headers_mut().insert(header::RETRY_AFTER, value);
        }
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> TrainRequest {
        TrainRequest {
            train_steps: 10,
            learning_rate: 1e-4,
            checkpoint_interval: 5,
            eval_interval: 5,
            resume_from: None,
        }
    }

    #[test]
    fn rejects_train_steps_above_cap() {
        let limits = ServerLimits {
            max_train_steps: 9,
            ..ServerLimits::default()
        };

        let err = validate_request(&request(), limits).expect_err("10 > 9 must be rejected");

        assert_eq!(StatusCode::BAD_REQUEST, err.status());
        assert_eq!(
            serde_json::json!({"error":"train_steps_out_of_range","max_allowed":9,"requested":10}),
            serde_json::to_value(err.body()).unwrap()
        );
    }

    #[test]
    fn rejects_zero_train_steps() {
        let request = TrainRequest {
            train_steps: 0,
            ..request()
        };

        let err = validate_request(&request, ServerLimits::default())
            .expect_err("zero steps must be rejected");

        assert_eq!(StatusCode::BAD_REQUEST, err.status());
    }

    #[test]
    fn rejects_non_positive_and_non_finite_learning_rates() {
        for rate in [0.0, -1e-4, f64::NAN, f64::INFINITY] {
            let request = TrainRequest {
                learning_rate: rate,
                ..request()
            };
            assert!(
                validate_request(&request, ServerLimits::default()).is_err(),
                "learning_rate {rate} must be rejected"
            );
        }
    }

    #[test]
    fn rejects_learning_rate_above_cap() {
        let limits = ServerLimits {
            max_train_learning_rate: 1e-5,
            ..ServerLimits::default()
        };

        let err = validate_request(&request(), limits).expect_err("1e-4 > 1e-5 must be rejected");

        assert_eq!(StatusCode::BAD_REQUEST, err.status());
    }

    #[test]
    fn accepts_a_request_exactly_at_both_caps() {
        let limits = ServerLimits {
            max_train_steps: 10,
            max_train_learning_rate: 1e-4,
            ..ServerLimits::default()
        };

        assert!(validate_request(&request(), limits).is_ok());
    }

    #[test]
    fn manifest_path_is_run_id_scoped() {
        assert_eq!(
            PathBuf::from("checkpoints/runs/run-abc.json"),
            manifest_path(Path::new(DEFAULT_RUNS_DIR), "abc")
        );
    }

    #[test]
    fn progress_events_fold_into_the_record() {
        let handle = RunHandle::new(
            std::env::temp_dir().join(format!("rusty-gpt-run-events-{}", std::process::id())),
        );
        *handle.lock() = Some(TrainRunRecord::started("abc".to_string(), request()));

        apply_event_line(
            &handle,
            r#"{"event":"training_progress","step":4,"total_steps":10,"training_loss":2.5,"value_loss":3.5,"steps_per_second":2.0}"#,
        );
        apply_event_line(
            &handle,
            r#"{"event":"checkpoint_saved","path":"/srv/rusty-gpt/checkpoints/mini_gpt.step-5.mpk","elapsed_ms":1}"#,
        );
        apply_event_line(&handle, "not json at all");

        let record = handle.snapshot().unwrap();
        assert_eq!(5, record.steps_completed);
        assert_eq!(10, record.total_steps);
        assert_eq!(Some(2.5), record.training_loss);
        assert_eq!(Some(3.5), record.value_loss);
        assert_eq!(Some(2.0), record.steps_per_second);
        assert_eq!(
            vec!["mini_gpt.step-5.mpk"],
            record.checkpoints,
            "checkpoint paths are recorded as basenames, never absolute paths"
        );

        let _ = std::fs::remove_dir_all(&handle.runs_dir);
    }

    fn running_record(steps_completed: usize, steps_per_second: Option<f64>) -> TrainRunRecord {
        TrainRunRecord {
            steps_completed,
            steps_per_second,
            ..TrainRunRecord::started("abc".to_string(), request())
        }
    }

    #[test]
    fn eta_is_remaining_steps_over_reported_throughput() {
        // 10 total, 4 done, 2 steps/s => 3s left.
        assert_eq!(Some(3), eta_seconds(&running_record(4, Some(2.0))));
    }

    #[test]
    fn eta_is_absent_without_usable_throughput() {
        assert_eq!(None, eta_seconds(&running_record(4, None)));
        assert_eq!(
            None,
            eta_seconds(&running_record(4, Some(0.0))),
            "a zero rate would divide to infinity"
        );
    }

    #[test]
    fn eta_is_absent_once_the_run_is_terminal() {
        for status in [
            TrainRunStatus::Completed,
            TrainRunStatus::Interrupted,
            TrainRunStatus::Failed,
        ] {
            let record = TrainRunRecord {
                status,
                ..running_record(4, Some(2.0))
            };
            assert_eq!(None, eta_seconds(&record), "{status:?} has nothing left");
        }
    }

    #[test]
    fn status_response_carries_manifest_field_names_plus_eta() {
        let response = TrainStatusResponse {
            eta_seconds: Some(3),
            record: running_record(4, Some(2.0)),
        };

        let body = serde_json::to_value(&response).unwrap();

        assert_eq!("abc", body["run_id"]);
        assert_eq!("running", body["status"]);
        assert_eq!(4, body["steps_completed"]);
        assert_eq!(10, body["total_steps"]);
        assert_eq!(2.0, body["steps_per_second"]);
        assert_eq!(3, body["eta_seconds"]);
        assert_eq!(10, body["request"]["train_steps"]);
    }

    #[test]
    fn manifests_written_before_steps_per_second_existed_still_load() {
        let legacy = r#"{
            "run_id":"abc","status":"completed",
            "request":{"train_steps":10,"learning_rate":0.0001,"checkpoint_interval":5,"eval_interval":5},
            "started_at_unix":1,"ended_at_unix":2,"steps_completed":10,"total_steps":10,
            "training_loss":2.5,"value_loss":3.5,"checkpoints":[]
        }"#;

        let record: TrainRunRecord = serde_json::from_str(legacy).unwrap();

        assert_eq!(None, record.steps_per_second);
    }

    #[test]
    fn missing_manifest_is_not_found_rather_than_an_error() {
        let dir = std::env::temp_dir().join(format!(
            "rusty-gpt-run-missing-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));

        let found = read_manifest(&dir, "does-not-exist").expect("a missing file is not a fault");

        assert!(found.is_none());
    }
}
