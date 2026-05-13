use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

const DATA_DIR: &str = "data";

#[derive(Debug, PartialEq, Eq)]
struct Config {
    repo: PathBuf,
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    let config = parse_args(env::args().skip(1))?;
    let output_path = output_path(&config.repo, config.output.as_deref())?;
    let files = collect_source_files(&config.repo)?;
    let corpus = concatenate_files(&config.repo, &files)?;

    fs::create_dir_all(DATA_DIR).context("failed to create data directory")?;
    fs::write(&output_path, corpus)
        .with_context(|| format!("failed to write source corpus to {:?}", output_path))?;

    println!(
        "Wrote {} source files from {:?} to {:?}",
        files.len(),
        config.repo,
        output_path
    );

    Ok(())
}

fn parse_args<I, S>(args: I) -> Result<Config>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let args: Vec<String> = args
        .into_iter()
        .map(|arg| arg.as_ref().to_string())
        .collect();
    let mut repo = None;
    let mut output = None;
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--repo" => {
                let value = args
                    .get(index + 1)
                    .context("--repo requires a repository path")?;
                repo = Some(PathBuf::from(value));
                index += 2;
            }
            "--output" => {
                let value = args
                    .get(index + 1)
                    .context("--output requires a .txt file name")?;
                output = Some(PathBuf::from(value));
                index += 2;
            }
            "-h" | "--help" => {
                bail!("Usage: collect-source --repo <path> [--output <file.txt>]");
            }
            other => bail!("unsupported argument: {other}"),
        }
    }

    Ok(Config {
        repo: repo.context("--repo is required")?,
        output,
    })
}

fn output_path(repo: &Path, requested: Option<&Path>) -> Result<PathBuf> {
    let file_name = match requested {
        Some(path) => path
            .file_name()
            .context("--output must include a file name")?
            .to_os_string(),
        None => {
            let repo_name = repo
                .file_name()
                .and_then(|name| name.to_str())
                .filter(|name| !name.is_empty())
                .unwrap_or("repository");
            format!("{repo_name}.txt").into()
        }
    };

    let path = PathBuf::from(DATA_DIR).join(file_name);
    if path.extension().and_then(|extension| extension.to_str()) != Some("txt") {
        bail!("--output must end in .txt");
    }

    Ok(path)
}

fn collect_source_files(repo: &Path) -> Result<Vec<PathBuf>> {
    if !repo.is_dir() {
        bail!("repository path is not a directory: {:?}", repo);
    }

    let mut files = Vec::new();
    visit_dir(repo, &mut files)?;
    files.sort();
    Ok(files)
}

fn visit_dir(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("failed to read directory {:?}", dir))? {
        let entry = entry.with_context(|| format!("failed to read entry in {:?}", dir))?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .with_context(|| format!("failed to read file type for {:?}", path))?;

        if file_type.is_dir() {
            if should_skip_dir(&path) {
                continue;
            }
            visit_dir(&path, files)?;
        } else if file_type.is_file() && is_source_file(&path) {
            files.push(path);
        }
    }

    Ok(())
}

fn concatenate_files(repo: &Path, files: &[PathBuf]) -> Result<String> {
    let mut output = String::new();
    for path in files {
        let relative = path.strip_prefix(repo).unwrap_or(path);
        let contents = fs::read_to_string(path)
            .with_context(|| format!("failed to read source file {:?}", path))?;

        output.push_str("\n--- FILE: ");
        output.push_str(&relative.to_string_lossy());
        output.push_str(" ---\n");
        output.push_str(&contents);
        if !contents.ends_with('\n') {
            output.push('\n');
        }
    }

    Ok(output)
}

fn should_skip_dir(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(
            ".git"
                | ".hg"
                | ".svn"
                | ".idea"
                | ".vscode"
                | "target"
                | "node_modules"
                | "dist"
                | "build"
                | ".next"
                | "coverage"
                | "__pycache__"
        )
    )
}

fn is_source_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|extension| extension.to_str()),
        Some(
            "rs" | "toml"
                | "js"
                | "jsx"
                | "ts"
                | "tsx"
                | "mjs"
                | "cjs"
                | "py"
                | "go"
                | "java"
                | "kt"
                | "kts"
                | "c"
                | "h"
                | "cc"
                | "cpp"
                | "hpp"
                | "cs"
                | "swift"
                | "rb"
                | "php"
                | "sh"
                | "bash"
                | "zsh"
                | "fish"
                | "html"
                | "css"
                | "scss"
                | "json"
                | "yaml"
                | "yml"
                | "sql"
                | "md"
        )
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_required_repo_arg() {
        let config = parse_args(["--repo", "/tmp/example"]).unwrap();

        assert_eq!(
            Config {
                repo: PathBuf::from("/tmp/example"),
                output: None,
            },
            config
        );
    }

    #[test]
    fn output_path_stays_in_data_dir() {
        let path = output_path(
            Path::new("/tmp/example-repo"),
            Some(Path::new("/tmp/ignored/custom.txt")),
        )
        .unwrap();

        assert_eq!(PathBuf::from("data/custom.txt"), path);
    }

    #[test]
    fn detects_common_source_extensions() {
        assert!(is_source_file(Path::new("src/main.rs")));
        assert!(is_source_file(Path::new("package.json")));
        assert!(!is_source_file(Path::new("image.png")));
    }

    #[test]
    fn skips_generated_directories() {
        assert!(should_skip_dir(Path::new("/repo/target")));
        assert!(should_skip_dir(Path::new("/repo/node_modules")));
        assert!(!should_skip_dir(Path::new("/repo/src")));
    }
}
