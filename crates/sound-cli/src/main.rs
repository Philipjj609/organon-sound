//! `organon-sound`: give it a sound, get back what was said, where the music is (tempo, key),
//! and which sound events happen when.
//!
//! Work folder per input (default `$ORGANON_HOME/sound/<key>`, `ORGANON_HOME` = `~/.organon`):
//! `audio.*` (when fetched from a URL) · `whisper.json` · `index.md` (read first) ·
//! `transcript.md` · `windows.tsv` · `messages.jsonl` (organon-protocol, for `mneme ingest`).

#![forbid(unsafe_code)]

mod engines;
mod render;
mod tools;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context as _};
use clap::{Parser, Subcommand, ValueEnum};
use organon_protocol::{sha256_hex, Anchor, Message, ProducedBy, Scope, Source, SourceId};
use organon_sound_core::emit::{self, ORGAN};
use organon_sound_core::hear::{hear, Options, Tag, Transcribe};
use organon_sound_core::key::Profile;
use organon_sound_core::Pcm;
use serde_json::Value as J;

use crate::engines::FfmpegWhisper;

const VERSION: &str = env!("CARGO_PKG_VERSION");
/// Where whisper.cpp publishes its ggml models.
const MODELS_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";
const VAD_URL: &str = "https://huggingface.co/ggml-org/whisper-vad/resolve/main";
const VAD_MODEL: &str = "ggml-silero-v5.1.2.bin";
/// YAMNet as ONNX: a tf2onnx export of Google's SavedModel (weights unchanged, says its card).
const YAMNET_URL: &str = "https://huggingface.co/audiomagic/yamnet-onnx/resolve/main";
const YAMNET_MODEL: &str = "yamnet.onnx";
const YAMNET_CLASSES: &str = "yamnet_class_map.csv";

#[derive(Parser)]
#[command(
    name = "organon-sound",
    version,
    about = "Hear a sound: speech, music (tempo, key), sound events"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Which external tools and models are present
    Doctor,
    /// Sense + perceive: prints where the index is
    Hear {
        /// An audio or video file, an organon-video work folder, or a URL (yt-dlp)
        input: String,
        /// Work folder (default: $ORGANON_HOME/sound/<key>)
        #[arg(long)]
        dir: Option<PathBuf>,
        #[command(flatten)]
        speech: SpeechArgs,
        /// Key profile family
        #[arg(long, value_enum, default_value_t = KeyProfile::Krumhansl)]
        profile: KeyProfile,
        /// Skip sound-event tagging (YAMNet), even when its model is present
        #[arg(long)]
        no_tags: bool,
    },
    /// Download a model: whisper.cpp (`base`, `small`, `medium`, `large-v3-turbo`, …), `vad`
    /// (Silero) or `yamnet` (sound events)
    Model { name: String },
}

/// Speech-to-text choices.
#[derive(clap::Args)]
struct SpeechArgs {
    /// Whisper model name in the models folder (`base`, `small`, …) or a path
    #[arg(long, default_value = "base")]
    model: String,
    /// Spoken language (`auto` detects it)
    #[arg(long, default_value = "auto")]
    language: String,
    /// Seconds per whisper pass
    #[arg(long, default_value_t = 20)]
    queue: u32,
    /// Cut at pauses with the Silero VAD model (`organon-sound model vad`). Measured 9x
    /// slower with ffmpeg 9.0.1 on CPU, so off by default
    #[arg(long)]
    vad: bool,
    /// Let whisper use the GPU (measured slower than CPU on a machine without one)
    #[arg(long)]
    gpu: bool,
    /// Skip speech-to-text (music and events only)
    #[arg(long)]
    no_speech: bool,
}

#[derive(Clone, Copy, ValueEnum)]
enum KeyProfile {
    Krumhansl,
    Temperley,
}

impl From<KeyProfile> for Profile {
    fn from(p: KeyProfile) -> Self {
        match p {
            KeyProfile::Krumhansl => Self::Krumhansl,
            KeyProfile::Temperley => Self::Temperley,
        }
    }
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().cmd {
        Cmd::Doctor => doctor(),
        Cmd::Model { name } => model(&name),
        Cmd::Hear {
            input,
            dir,
            speech,
            profile,
            no_tags,
        } => {
            let sensed = sense(&input, dir)?;
            let engine = if speech.no_speech {
                None
            } else {
                Some(whisper_engine(&sensed.dir, speech)?)
            };
            let tagger = if no_tags { None } else { tagger()? };
            let index = perceive(
                &sensed,
                engine.as_ref().map(|s| s as &dyn Transcribe),
                tagger.as_deref(),
                profile.into(),
            )?;
            println!("{}", index.display());
            Ok(())
        }
    }
}

fn organon_home() -> PathBuf {
    std::env::var_os("ORGANON_HOME").map_or_else(
        || {
            let home = std::env::var_os("USERPROFILE")
                .or_else(|| std::env::var_os("HOME"))
                .unwrap_or_default();
            PathBuf::from(home).join(".organon")
        },
        PathBuf::from,
    )
}

fn models_dir() -> PathBuf {
    organon_home().join("sound").join("models")
}

fn doctor() -> anyhow::Result<()> {
    for tool in ["ffmpeg", "ffprobe", "yt-dlp", "curl"] {
        println!(
            "{tool:8} {}",
            tools::version(tool).unwrap_or_else(|| "MISSING".into())
        );
    }
    println!(
        "whisper  {}",
        if tools::has_whisper_filter() {
            "ffmpeg has the whisper filter"
        } else {
            "MISSING: needs ffmpeg 8+ built with --enable-whisper (or use --no-speech)"
        }
    );
    let dir = models_dir();
    let models: Vec<String> = fs::read_dir(&dir)
        .map(|r| {
            r.filter_map(Result::ok)
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n.ends_with(".bin") || n.ends_with(".onnx"))
                .collect()
        })
        .unwrap_or_default();
    println!(
        "models   {} ({})",
        if models.is_empty() {
            "none: run `organon-sound model base`".into()
        } else {
            models.join(", ")
        },
        dir.display()
    );
    Ok(())
}

fn model(name: &str) -> anyhow::Result<()> {
    let files: Vec<(String, String)> = match name {
        "vad" => vec![(VAD_MODEL.to_owned(), format!("{VAD_URL}/{VAD_MODEL}"))],
        "yamnet" => [YAMNET_MODEL, YAMNET_CLASSES]
            .iter()
            .map(|f| ((*f).to_owned(), format!("{YAMNET_URL}/{f}")))
            .collect(),
        _ => {
            let file = format!("ggml-{name}.bin");
            let url = format!("{MODELS_URL}/{file}");
            vec![(file, url)]
        }
    };
    let dir = models_dir();
    fs::create_dir_all(&dir)?;
    for (file, url) in files {
        let path = dir.join(&file);
        if path.exists() {
            println!("{} (already here)", path.display());
            continue;
        }
        tools::download(&url, &path).with_context(|| format!("downloading {url}"))?;
        println!("{}", path.display());
    }
    Ok(())
}

/// YAMNet when its model is present (and this build has the `yamnet` feature).
fn tagger() -> anyhow::Result<Option<Box<dyn Tag>>> {
    let (model, classes) = (
        models_dir().join(YAMNET_MODEL),
        models_dir().join(YAMNET_CLASSES),
    );
    if !(model.is_file() && classes.is_file()) {
        return Ok(None);
    }
    #[cfg(feature = "yamnet")]
    {
        let y = engines::Yamnet::load(&model, &classes).context("loading YAMNet")?;
        Ok(Some(Box::new(y)))
    }
    #[cfg(not(feature = "yamnet"))]
    Ok(None)
}

fn whisper_engine(dir: &Path, args: SpeechArgs) -> anyhow::Result<FfmpegWhisper> {
    let model = args.model.as_str();
    let path = if Path::new(model).is_file() {
        PathBuf::from(model)
    } else {
        models_dir().join(format!("ggml-{model}.bin"))
    };
    if !path.is_file() {
        bail!(
            "no whisper model at {}: run `organon-sound model {model}` (or use --no-speech)",
            path.display()
        );
    }
    if !tools::has_whisper_filter() {
        bail!(
            "this ffmpeg has no whisper filter (ffmpeg 8+ with --enable-whisper); use --no-speech"
        );
    }
    let vad = models_dir().join(VAD_MODEL);
    if args.vad && !vad.is_file() {
        bail!(
            "no VAD model at {}: run `organon-sound model vad`",
            vad.display()
        );
    }
    Ok(FfmpegWhisper {
        model: path,
        vad: args.vad.then_some(vad),
        language: args.language,
        queue: args.queue,
        gpu: args.gpu,
        dir: dir.to_owned(),
        ffmpeg: tools::ffmpeg_version().unwrap_or_else(|| "?".into()),
    })
}

/// What sensing gives perception: the media file, its source record, and where to write.
struct Sensed {
    dir: PathBuf,
    media: PathBuf,
    source: Source,
    anchor: Option<Anchor>,
}

fn sense(input: &str, dir: Option<PathBuf>) -> anyhow::Result<Sensed> {
    let sensed = if input.starts_with("http://") || input.starts_with("https://") {
        sense_url(input, dir)?
    } else {
        let p = Path::new(input);
        if p.join("source.json").is_file() {
            sense_video_dir(p, dir)?
        } else if p.is_file() {
            sense_file(p, dir)?
        } else {
            bail!("not a file, an organon-video folder or a URL: {input}");
        }
    };
    if !tools::has_audio(&sensed.media)? {
        bail!("{} has no audio stream", sensed.media.display());
    }
    Ok(sensed)
}

/// The 11-character YouTube id of a watch, short or shorts URL.
fn youtube_id(url: &str) -> Option<&str> {
    let rest = url.split_once("://")?.1;
    let rest = rest.strip_prefix("www.").or(Some(rest))?;
    let rest = rest.strip_prefix("m.").unwrap_or(rest);
    let id = if let Some(r) = rest.strip_prefix("youtu.be/") {
        r
    } else if let Some(r) = rest.strip_prefix("youtube.com/shorts/") {
        r
    } else {
        let q = rest.strip_prefix("youtube.com/watch?")?;
        q.split('&').find_map(|kv| kv.strip_prefix("v="))?
    };
    let id = id.split(['?', '&', '/', '#']).next()?;
    (id.len() == 11
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'))
    .then_some(id)
}

fn sense_url(url: &str, dir: Option<PathBuf>) -> anyhow::Result<Sensed> {
    let yt = youtube_id(url);
    let key = yt.map_or_else(
        || format!("url-{}", &sha256_hex(url.as_bytes())[..12]),
        |id| format!("yt-{id}"),
    );
    let dir = dir.unwrap_or_else(|| organon_home().join("sound").join(&key));
    let media = tools::fetch_audio(url, &dir)?;
    let title_file = dir.join("title.txt");
    if !title_file.exists() {
        let out = tools::run(
            "yt-dlp",
            &[
                "--no-warnings",
                "--no-playlist",
                "--skip-download",
                "--print",
                "title",
                url,
            ],
            None,
        )?;
        fs::write(&title_file, String::from_utf8_lossy(&out.stdout).trim())?;
    }
    let title = fs::read_to_string(&title_file)?;
    // Same identity as organon-video, so speech and pictures of one video share a source.
    let (identity, kind, anchor, uri) = match yt {
        Some(id) => (
            format!("youtube:{id}"),
            "video",
            Some(format!("video:yt/{id}")),
            format!("https://youtu.be/{id}"),
        ),
        None => (format!("url:{url}"), "audio", None, url.to_owned()),
    };
    let files = BTreeMap::from([("audio".to_owned(), file_name(&media))]);
    let source = source_record(
        &identity,
        kind,
        &uri,
        &title,
        &media,
        files,
        anchor.as_deref(),
        "yt-dlp",
    )?;
    let anchor = anchor.map(|a| new_anchor(&a, &title)).transpose()?;
    Ok(Sensed {
        dir,
        media,
        source,
        anchor,
    })
}

/// An organon-video work folder: its source (same id, so observations attach to the video),
/// its video file, or the audio fetched again when the video was downloaded without sound.
fn sense_video_dir(vdir: &Path, dir: Option<PathBuf>) -> anyhow::Result<Sensed> {
    let source: Source = serde_json::from_str(&fs::read_to_string(vdir.join("source.json"))?)
        .context("reading organon-video's source.json")?;
    let key = vdir
        .file_name()
        .map_or_else(|| "video".into(), |n| n.to_string_lossy().into_owned());
    let dir = dir.unwrap_or_else(|| organon_home().join("sound").join(key));
    fs::create_dir_all(&dir)?;
    let video = source.files.get("video").map(|f| vdir.join(f));
    let media = match video {
        Some(v) if v.is_file() && tools::has_audio(&v)? => v,
        _ if source.uri.starts_with("http") => tools::fetch_audio(&source.uri, &dir)?,
        _ => bail!(
            "the video in {} has no audio and no URL to fetch it from",
            vdir.display()
        ),
    };
    Ok(Sensed {
        dir,
        media,
        source,
        anchor: None,
    })
}

fn sense_file(file: &Path, dir: Option<PathBuf>) -> anyhow::Result<Sensed> {
    let bytes = fs::read(file)?;
    // Same identity as organon-video gives a file, so one recording is one source.
    let key = sha256_hex(&bytes)[..12].to_owned();
    let dir = dir.unwrap_or_else(|| organon_home().join("sound").join(format!("file-{key}")));
    fs::create_dir_all(&dir)?;
    let kind = if tools::has_video(file)? {
        "video"
    } else {
        "audio"
    };
    let anchor = format!("{kind}:file/{key}");
    let title = file
        .file_stem()
        .map_or_else(|| "sound".into(), |s| s.to_string_lossy().into_owned());
    let uri = fs::canonicalize(file)
        .unwrap_or_else(|_| file.to_owned())
        .to_string_lossy()
        .trim_start_matches(r"\\?\")
        .to_owned();
    let source = source_record(
        &format!("file:{key}"),
        kind,
        &uri,
        &title,
        file,
        BTreeMap::new(),
        Some(&anchor),
        "ffprobe",
    )?;
    Ok(Sensed {
        dir,
        media: file.to_owned(),
        source,
        anchor: Some(new_anchor(&anchor, &title)?),
    })
}

#[allow(clippy::too_many_arguments)]
fn source_record(
    identity: &str,
    kind: &str,
    uri: &str,
    title: &str,
    media: &Path,
    files: BTreeMap<String, String>,
    anchor: Option<&str>,
    tool: &str,
) -> anyhow::Result<Source> {
    let mut meta = BTreeMap::new();
    if let Some(a) = anchor {
        meta.insert("anchor".to_owned(), J::from(a));
    }
    Ok(Source {
        id: SourceId::of(identity.as_bytes()),
        kind: kind.to_owned(),
        uri: uri.to_owned(),
        title: title.to_owned(),
        captured_at: now(),
        content_hash: sha256_hex(&fs::read(media)?),
        files,
        meta,
        produced_by: Some(ProducedBy {
            organ: ORGAN.to_owned(),
            version: VERSION.to_owned(),
            tool: Some(tool.to_owned()),
            model: None,
        }),
    })
}

fn new_anchor(id: &str, name: &str) -> anyhow::Result<Anchor> {
    Ok(Anchor {
        id: id.parse()?,
        name: name.to_owned(),
        aliases: Vec::new(),
        scope: Scope::world(),
        summary: None,
    })
}

fn perceive(
    sensed: &Sensed,
    speech: Option<&dyn Transcribe>,
    tagger: Option<&dyn Tag>,
    profile: Profile,
) -> anyhow::Result<PathBuf> {
    let pcm = Pcm::from_f32le(tools::RATE, &tools::decode(&sensed.media)?)?;
    let opts = Options {
        profile,
        ..Options::default()
    };
    let heard = hear(&pcm, &sensed.media, speech, tagger, &opts)?;
    let msgs = emit::messages(&heard, &sensed.source, sensed.anchor.as_ref(), VERSION);
    let jsonl: String = msgs
        .iter()
        .map(|m: &Message| serde_json::to_string(m).map(|l| l + "\n"))
        .collect::<Result<_, _>>()?;
    let dir = &sensed.dir;
    fs::write(dir.join("messages.jsonl"), jsonl)?;
    fs::write(dir.join("transcript.md"), render::transcript(&heard))?;
    fs::write(dir.join("windows.tsv"), render::windows_tsv(&heard))?;
    let index = dir.join("index.md");
    fs::write(
        &index,
        render::index(&heard, &sensed.source.title, &sensed.source.uri),
    )?;
    Ok(index)
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// RFC 3339 UTC now, second precision, without a date library.
fn now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let (days, rem) = (secs / 86_400, secs % 86_400);
    // Civil-from-days (Howard Hinnant), valid for the Unix era.
    let z = i64::try_from(days).unwrap_or(0) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem / 60 % 60,
        rem % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn youtube_ids() {
        for url in [
            "https://youtu.be/1XRTvj__ZPY",
            "https://www.youtube.com/watch?v=1XRTvj__ZPY&t=10",
            "https://youtube.com/watch?feature=x&v=1XRTvj__ZPY",
            "https://m.youtube.com/shorts/1XRTvj__ZPY?si=a",
        ] {
            assert_eq!(youtube_id(url), Some("1XRTvj__ZPY"), "{url}");
        }
        assert_eq!(youtube_id("https://example.com/a.mp3"), None);
        assert_eq!(youtube_id("https://youtu.be/short"), None);
    }

    #[test]
    fn source_ids_match_organon_video() {
        // organon-video: SourceId::of("youtube:<id>"); the Uncle Bob talk is s_bee7260cd6ed3e7b.
        assert_eq!(
            SourceId::of(b"youtube:1XRTvj__ZPY").to_string(),
            "s_bee7260cd6ed3e7b"
        );
    }

    #[test]
    fn now_is_rfc3339() {
        let t = now();
        assert_eq!(t.len(), 20);
        assert!(t.ends_with('Z') && t.as_bytes()[10] == b'T');
    }
}
