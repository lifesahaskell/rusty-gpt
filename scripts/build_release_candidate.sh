#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="rusty-gpt"
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT_DIR/Cargo.toml" | head -n 1)"
RC_ID="${RC_ID:-$(date -u +%Y%m%d%H%M%S)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$ROOT_DIR/target/release-candidates}"
PACKAGE_NAME="$APP_NAME-$VERSION-rc.$RC_ID"
BUILD_DIR="$ARTIFACT_ROOT/$PACKAGE_NAME"
PACKAGE_PATH="$ARTIFACT_ROOT/$PACKAGE_NAME.tar.gz"
RUSTY_GPT_BACKEND="${RUSTY_GPT_BACKEND:-cpu}"
RELEASE_SMOKE_CHECK="${RELEASE_SMOKE_CHECK:-1}"

CARGO_FEATURE_ARGS=()
case "$RUSTY_GPT_BACKEND" in
  cpu)
    ;;
  cuda)
    CARGO_FEATURE_ARGS=(--features cuda)
    ;;
  *)
    echo "Unsupported RUSTY_GPT_BACKEND '$RUSTY_GPT_BACKEND'; expected cpu or cuda." >&2
    exit 2
    ;;
esac

case "$RELEASE_SMOKE_CHECK" in
  1|true|0|false)
    ;;
  *)
    echo "Unsupported RELEASE_SMOKE_CHECK '$RELEASE_SMOKE_CHECK'; expected 1, true, 0, or false." >&2
    exit 2
    ;;
esac

if command -v git >/dev/null 2>&1; then
  GIT_COMMIT="$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || true)"
else
  GIT_COMMIT=""
fi

cd "$ROOT_DIR"
cargo build --release "${CARGO_FEATURE_ARGS[@]}" --bin "$APP_NAME"

cd "$ROOT_DIR/mini-gpt-ui"
npm run build

rm -rf "$BUILD_DIR" "$PACKAGE_PATH"
mkdir -p "$BUILD_DIR/bin" "$BUILD_DIR/ui" "$BUILD_DIR/scripts"

install -m 755 "$ROOT_DIR/target/release/rusty-gpt" "$BUILD_DIR/bin/rusty-gpt"
cp -R "$ROOT_DIR/mini-gpt-ui/dist/." "$BUILD_DIR/ui/"

cat > "$BUILD_DIR/scripts/run_api.sh" <<'RUN_API'
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

exec "$ARTIFACT_DIR/bin/rusty-gpt" --serve "$@"
RUN_API
chmod +x "$BUILD_DIR/scripts/run_api.sh"

SMOKE_CHECK_STATUS="skipped"
if [[ "$RELEASE_SMOKE_CHECK" == "1" || "$RELEASE_SMOKE_CHECK" == "true" ]]; then
  "$BUILD_DIR/bin/rusty-gpt" \
    --input "$ROOT_DIR/tests/fixtures/input.txt" \
    --model trivial \
    --backend cpu \
    --train-steps 1 \
    > "$BUILD_DIR/smoke-check.log" \
    2>&1
  SMOKE_CHECK_STATUS="passed"
fi

cat > "$BUILD_DIR/README.md" <<README
# $PACKAGE_NAME

Release candidate artifact for $APP_NAME.

- API binary: \`bin/rusty-gpt\`
- UI static bundle: \`ui/\`
- API launcher: \`scripts/run_api.sh\`
- Manifest: \`manifest.txt\`
- CPU smoke-check log: \`smoke-check.log\` when \`smoke_check=passed\`

Run the API:

\`\`\`bash
./scripts/run_api.sh --input /path/to/input.txt --server-addr 127.0.0.1:8787
\`\`\`

This artifact was built with \`RUSTY_GPT_BACKEND=$RUSTY_GPT_BACKEND\`.

- CPU artifacts are built without the \`cuda\` Cargo feature and should be used with \`--backend cpu\`.
- CUDA artifacts are built with \`--features cuda\`; they can still run CPU smoke checks, but \`--backend cuda\` requires a compatible NVIDIA driver on the host.

Serve \`ui/\` with any static file server and proxy \`/api\` to the API server.

CUDA API smoke after unpacking on a GPU host:

\`\`\`bash
./scripts/run_api.sh --backend cuda --input /path/to/input.txt --server-addr 127.0.0.1:8787
curl http://127.0.0.1:8787/api/info
\`\`\`
README

cat > "$BUILD_DIR/manifest.txt" <<MANIFEST
name=$APP_NAME
version=$VERSION
rc_id=$RC_ID
git_commit=$GIT_COMMIT
built_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
backend=$RUSTY_GPT_BACKEND
smoke_check=$SMOKE_CHECK_STATUS
api_binary=bin/rusty-gpt
ui_dir=ui
MANIFEST

tar -C "$ARTIFACT_ROOT" -czf "$PACKAGE_PATH" "$PACKAGE_NAME"

echo "$PACKAGE_PATH"
