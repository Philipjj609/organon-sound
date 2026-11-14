//! Engines behind the core's traits.

use std::path::{Path, PathBuf};

use organon_sound_core::hear::{Engine, Transcribe};
use organon_sound_core::speech::parse_whisper_json;
use organon_sound_core::Segment;

use crate::tools;

/// Speech-to-text with ffmpeg's `whisper` filter (whisper.cpp inside ffmpeg 8+): no other
/// binary, any container ffmpeg reads, a ggml model file.
pub struct FfmpegWhisper {
    pub model: PathBuf,
    /// Silero VAD model (ggml). With it the filter cuts audio at pauses instead of every
    /// `queue` seconds.
    pub vad: Option<PathBuf>,
    /// `auto` or a language code.
    pub language: String,
    /// Seconds of audio per whisper pass. ffmpeg's docs: 10–20 s is more accurate and cheaper
    /// than the default 3 s, which is for live streams.
    pub queue: u32,
    /// Let whisper.cpp use a GPU. ffmpeg's default is on; on a machine without a usable GPU
    /// it measured slower (21 s vs 13 s for a minute of speech), so ours is off.
    pub gpu: bool,
    /// Work folder, where the filter writes `whisper.json`.
    pub dir: PathBuf,
    pub ffmpeg: String,
}

impl FfmpegWhisper {
    /// The filter graph string; `None` when a model path cannot be written into one.
    pub fn filter(&self) -> Option<String> {
        let mut f = format!(
            "whisper=model={}:language={}:queue={}:use_gpu={}:destination=whisper.json:format=json",
            tools::filter_path(&self.model)?,
            self.language,
            self.queue,
            self.gpu
        );
        if let Some(vad) = &self.vad {
            f.push_str(&format!(":vad_model={}", tools::filter_path(vad)?));
        }
        Some(f)
    }
}

impl Transcribe for FfmpegWhisper {
    fn engine(&self) -> Engine {
        Engine {
            tool: Some(format!("ffmpeg {} whisper", self.ffmpeg)),
            model: Some(model_name(&self.model, self.vad.as_deref())),
        }
    }

    fn transcribe(&self, media: &Path) -> Result<Vec<Segment>, String> {
        let filter = self
            .filter()
            .ok_or("a model path containing ' cannot go into an ffmpeg filter; move the model")?;
        let json = tools::whisper(media, &filter, &self.dir).map_err(|e| format!("{e:#}"))?;
        parse_whisper_json(&json).map_err(|e| e.to_string())
    }
}

/// `ggml-base.bin` → `ggml-base`, plus `+silero-v5.1.2` when a VAD model is used.
fn model_name(model: &Path, vad: Option<&Path>) -> String {
    let stem = |p: &Path| {
        p.file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    match vad {
        Some(v) => format!("{}+{}", stem(model), stem(v).trim_start_matches("ggml-")),
        None => stem(model),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_escapes_paths_and_adds_vad() {
        let w = FfmpegWhisper {
            model: PathBuf::from(r"C:\m\ggml-base.bin"),
            vad: Some(PathBuf::from(r"C:\m\ggml-silero-v5.1.2.bin")),
            language: "auto".into(),
            queue: 20,
            gpu: false,
            dir: PathBuf::from("."),
            ffmpeg: "9.0.1".into(),
        };
        assert_eq!(
            w.filter().as_deref(),
            Some(
                r"whisper=model='C\:/m/ggml-base.bin':language=auto:queue=20:use_gpu=false:destination=whisper.json:format=json:vad_model='C\:/m/ggml-silero-v5.1.2.bin'"
            )
        );
        assert_eq!(w.engine().model.as_deref(), Some("ggml-base+silero-v5.1.2"));
    }
}

/// Sound-event tagging with YAMNet (521 AudioSet classes, mAP 0.306) run by tract. The model
/// takes raw 16 kHz samples; it is compiled once for a fixed chunk of 21 frames, and chunks
/// step by exactly 21 hops so frames tile the recording without seams.
#[cfg(feature = "yamnet")]
pub struct Yamnet {
    plan: tract_onnx::prelude::TypedRunnableModel<tract_onnx::prelude::TypedModel>,
    labels: Vec<String>,
}

#[cfg(feature = "yamnet")]
mod yamnet_consts {
    /// Samples per frame window (0.96 s) and per hop (0.48 s) at 16 kHz.
    pub const WINDOW: usize = 15_600;
    pub const HOP: usize = 7_680;
    pub const FRAMES: usize = 21;
    pub const CHUNK: usize = WINDOW + HOP * (FRAMES - 1);
}

#[cfg(feature = "yamnet")]
impl Yamnet {
    /// Load `yamnet.onnx` and its class map.
    pub fn load(model: &Path, class_map: &Path) -> anyhow::Result<Self> {
        use tract_onnx::prelude::*;
        let labels =
            organon_sound_core::tagging::parse_class_map(&std::fs::read_to_string(class_map)?);
        let plan = tract_onnx::onnx()
            .model_for_path(model)?
            .with_input_fact(0, f32::fact([yamnet_consts::CHUNK]).into())?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { plan, labels })
    }

    fn scores(&self, samples: &[f32]) -> anyhow::Result<Vec<Vec<f32>>> {
        use tract_onnx::prelude::*;
        use yamnet_consts::{CHUNK, FRAMES, HOP, WINDOW};
        let total = if samples.len() <= WINDOW {
            1
        } else {
            (samples.len() - WINDOW).div_ceil(HOP) + 1
        };
        let mut out = Vec::with_capacity(total);
        for start in (0..samples.len().max(1)).step_by(FRAMES * HOP) {
            let mut chunk = vec![0.0f32; CHUNK];
            let end = (start + CHUNK).min(samples.len());
            chunk[..end - start].copy_from_slice(&samples[start..end]);
            let input: Tensor = tract_ndarray::Array1::from(chunk).into();
            let result = self.plan.run(tvec!(input.into()))?;
            let scores = result[0].to_array_view::<f32>()?;
            for row in scores.rows() {
                if out.len() < total {
                    out.push(row.to_vec());
                }
            }
        }
        Ok(out)
    }
}

#[cfg(feature = "yamnet")]
impl organon_sound_core::Tag for Yamnet {
    fn engine(&self) -> Engine {
        Engine {
            tool: Some("tract".into()),
            model: Some("yamnet".into()),
        }
    }

    fn tag(&self, pcm: &organon_sound_core::Pcm) -> Result<Vec<organon_sound_core::Event>, String> {
        use organon_sound_core::tagging::{events, TagRules};
        if pcm.rate() != 16_000 {
            return Err(format!("YAMNet needs 16 kHz, got {} Hz", pcm.rate()));
        }
        let scores = self.scores(pcm.samples()).map_err(|e| format!("{e:#}"))?;
        // Silence is measured by level, which places its edges to 50 ms.
        Ok(events(
            &scores,
            &self.labels,
            &["Silence"],
            &TagRules::YAMNET,
            &self.engine(),
        ))
    }
}
