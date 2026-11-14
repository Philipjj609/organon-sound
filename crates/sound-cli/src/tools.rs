//! The external tools this organ wraps (never hides): ffmpeg (decoding and its `whisper`
//! filter), ffprobe, yt-dlp, curl. The only code here that spawns processes.

use std::path::Path;
use std::process::{Command, Output};

use anyhow::{bail, Context as _};

/// Sample rate everything is decoded to: whisper's own rate, and enough for tempo and key
/// (chroma stops at about 2 kHz).
pub const RATE: u32 = 16_000;

/// Run a tool, failing with the end of its stderr if it exits non-zero.
pub fn run(tool: &str, args: &[&str], cwd: Option<&Path>) -> anyhow::Result<Output> {
    let mut cmd = Command::new(tool);
    cmd.args(args);
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    let out = cmd
        .output()
        .with_context(|| format!("running {tool} (is it installed? see `organon-sound doctor`)"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let tail: Vec<&str> = err.lines().rev().take(5).collect();
        bail!(
            "{tool} failed: {}",
            tail.into_iter().rev().collect::<Vec<_>>().join(" | ")
        );
    }
    Ok(out)
}

/// First line of `tool --version` (or `-version`), if the tool is present.
pub fn version(tool: &str) -> Option<String> {
    ["--version", "-version"].iter().find_map(|flag| {
        let out = Command::new(tool).arg(flag).output().ok()?;
        out.status.success().then(|| {
            String::from_utf8_lossy(&out.stdout)
                .lines()
                .next()
                .unwrap_or_default()
                .to_owned()
        })
    })
}

/// ffmpeg's version number (`ffmpeg version 9.0.1-full_build-…` → `9.0.1`).
pub fn ffmpeg_version() -> Option<String> {
    let line = version("ffmpeg")?;
    let v = line.split_whitespace().nth(2)?;
    Some(v.split('-').next().unwrap_or(v).to_owned())
}

/// Whether this ffmpeg was built with the `whisper` filter (ffmpeg 8+, `--enable-whisper`).
pub fn has_whisper_filter() -> bool {
    run("ffmpeg", &["-hide_banner", "-filters"], None).is_ok_and(|o| {
        String::from_utf8_lossy(&o.stdout)
            .lines()
            .any(|l| l.split_whitespace().nth(1) == Some("whisper"))
    })
}

/// Whether a media file has an audio stream.
pub fn has_audio(media: &Path) -> anyhow::Result<bool> {
    let path = media.to_string_lossy();
    let out = run(
        "ffprobe",
        &[
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            &path,
        ],
        None,
    )?;
    Ok(!String::from_utf8_lossy(&out.stdout).trim().is_empty())
}

/// Whether a media file has a video stream (cover art counts as none).
pub fn has_video(media: &Path) -> anyhow::Result<bool> {
    let path = media.to_string_lossy();
    let out = run(
        "ffprobe",
        &[
            "-v",
            "error",
            "-select_streams",
            "v",
            "-show_entries",
            "stream=codec_name:stream_disposition=attached_pic",
            "-of",
            "csv=p=0",
            &path,
        ],
        None,
    )?;
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .any(|l| !l.trim().is_empty() && !l.trim_end().ends_with(",1")))
}

/// The first audio stream as mono `f32` samples at [`RATE`].
pub fn decode(media: &Path) -> anyhow::Result<Vec<u8>> {
    let path = media.to_string_lossy();
    let rate = RATE.to_string();
    let out = run(
        "ffmpeg",
        &[
            "-v", "error", "-nostdin", "-i", &path, "-map", "0:a:0", "-ac", "1", "-ar", &rate,
            "-f", "f32le", "-",
        ],
        None,
    )?;
    Ok(out.stdout)
}

/// A path as an ffmpeg filter option value. A filter graph is unescaped twice (graph, then
/// option), so the value is single-quoted for the graph with its `:` escaped for the option:
/// `'C\:/Users/…'`. `None` for a path containing `'`, which cannot be quoted this way.
pub fn filter_path(p: &Path) -> Option<String> {
    let s = p.to_string_lossy().replace('\\', "/");
    (!s.contains('\'')).then(|| format!("'{}'", s.replace(':', "\\:")))
}

/// Run ffmpeg's `whisper` filter over the first audio stream; returns its JSON lines.
/// `dir` is where the output file goes (ffmpeg runs there, so the destination needs no
/// escaping).
pub fn whisper(media: &Path, filter: &str, dir: &Path) -> anyhow::Result<String> {
    let out_file = dir.join("whisper.json");
    if out_file.exists() {
        std::fs::remove_file(&out_file)?;
    }
    let path = media.to_string_lossy();
    run(
        "ffmpeg",
        &[
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            &path,
            "-map",
            "0:a:0",
            "-af",
            filter,
            "-f",
            "null",
            "-",
        ],
        Some(dir),
    )?;
    std::fs::read_to_string(&out_file)
        .with_context(|| format!("ffmpeg's whisper filter wrote no {}", out_file.display()))
}

/// Best audio of a URL via yt-dlp into `dir/audio.<ext>`; returns the file.
pub fn fetch_audio(url: &str, dir: &Path) -> anyhow::Result<std::path::PathBuf> {
    std::fs::create_dir_all(dir)?;
    if let Some(found) = find_audio(dir)? {
        return Ok(found);
    }
    run(
        "yt-dlp",
        &[
            "-q",
            "--no-warnings",
            "--no-playlist",
            "-f",
            "ba[abr<=160]/ba/b",
            "-o",
            "audio.%(ext)s",
            url,
        ],
        Some(dir),
    )?;
    find_audio(dir)?.context("yt-dlp wrote no audio file")
}

fn find_audio(dir: &Path) -> anyhow::Result<Option<std::path::PathBuf>> {
    Ok(std::fs::read_dir(dir)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| {
            p.file_stem().is_some_and(|s| s == "audio")
                && p.extension().is_some_and(|e| e != "part" && e != "ytdl")
        }))
}

/// Download `url` to `file` with curl (present on Windows 10+, macOS and Linux).
pub fn download(url: &str, file: &Path) -> anyhow::Result<()> {
    let partial = file.with_extension("part");
    let out = partial.to_string_lossy();
    run("curl", &["-fsSL", "--retry", "2", "-o", &out, url], None)?;
    std::fs::rename(&partial, file)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_paths_are_quoted_for_filters() {
        // Regression: unquoted `C\:/…` failed to parse in ffmpeg 9.0.1 on Windows.
        assert_eq!(
            filter_path(Path::new(r"C:\Users\me\.organon\ggml-base.bin")).as_deref(),
            Some(r"'C\:/Users/me/.organon/ggml-base.bin'")
        );
        assert_eq!(filter_path(Path::new("it's.bin")), None);
    }
}
