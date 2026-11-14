//! The hearing flow. Engines that need a model or a process (speech-to-text, event tagging)
//! sit behind traits the binary implements; level, tempo and key are measured here.

use std::path::Path;

use crate::error::{Error, Result};
use crate::key::{self, Profile};
use crate::music::{self, Features, Span, Thresholds, Window};
use crate::pcm::Pcm;
use crate::speech::{self, Segment};
use crate::{level, tempo};

/// Who produced a result: an external tool and model, or neither (measured by this organ).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Engine {
    pub tool: Option<String>,
    pub model: Option<String>,
}

/// A labelled stretch of sound: `silence`, `music` (as whisper annotates it), `applause`, …
#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    pub t0: f64,
    pub t1: f64,
    pub label: String,
    /// The engine's score in `[0, 1]`, when it gives one.
    pub score: Option<f64>,
    pub engine: Engine,
}

/// Speech-to-text. Takes the media file itself, because engines resample and chunk audio
/// their own way (ffmpeg's `whisper` filter reads any container).
pub trait Transcribe {
    fn engine(&self) -> Engine;
    /// Raw segments; the flow tidies them.
    ///
    /// # Errors
    /// The engine's own message.
    fn transcribe(&self, media: &Path) -> std::result::Result<Vec<Segment>, String>;
}

/// Sound-event tagging on decoded audio.
pub trait Tag {
    fn engine(&self) -> Engine;
    /// # Errors
    /// The engine's own message.
    fn tag(&self, pcm: &Pcm) -> std::result::Result<Vec<Event>, String>;
}

/// Choices and thresholds of a hearing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Options {
    pub profile: Profile,
    pub music: Thresholds,
    /// Frames below this level (dBFS) are silent.
    pub silence_db: f32,
    /// Silent stretches shorter than this are pauses, not events.
    pub min_silence: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            profile: Profile::default(),
            music: Thresholds::default(),
            silence_db: -50.0,
            min_silence: 2.0,
        }
    }
}

/// Everything heard in one recording.
#[derive(Debug, Clone)]
pub struct Heard {
    pub duration: f64,
    pub segments: Vec<Segment>,
    pub speech_engine: Option<Engine>,
    pub events: Vec<Event>,
    pub music: Vec<Span>,
    /// The measures behind `music`, kept so thresholds can be checked against real audio.
    pub windows: Vec<Window>,
    pub profile: Profile,
}

/// Hear a recording: decoded `pcm` for the measures, the `media` file for the transcriber.
///
/// # Errors
/// An engine failed.
pub fn hear(
    pcm: &Pcm,
    media: &Path,
    speech: Option<&dyn Transcribe>,
    tagger: Option<&dyn Tag>,
    opts: &Options,
) -> Result<Heard> {
    let db = level::frame_db(pcm);
    let mut events: Vec<Event> = level::silent_spans(&db, opts.silence_db, opts.min_silence)
        .into_iter()
        .map(|(t0, t1)| Event {
            t0,
            t1: t1.min(pcm.duration()),
            label: "silence".into(),
            score: None,
            engine: Engine::default(),
        })
        .collect();
    let features = Features {
        onsets: tempo::onset_envelope(pcm),
        chroma: key::chroma_frames(pcm),
        chroma_seconds: key::frame_seconds(pcm),
        db,
        duration: pcm.duration(),
    };
    let windows = music::windows(&features, opts.profile, &opts.music);
    let spans = music::spans(&features, &windows, opts.profile, &opts.music);

    let (segments, speech_engine) = match speech {
        Some(s) => {
            let engine = s.engine();
            let raw = s.transcribe(media).map_err(|message| Error::Engine {
                engine: engine.tool.clone().unwrap_or_default(),
                message,
            })?;
            let (said, notes) = speech::tidy(raw);
            events.extend(notes.into_iter().map(|n| Event {
                t0: n.t0,
                t1: n.t1,
                label: n.label,
                score: None,
                engine: engine.clone(),
            }));
            (said, Some(engine))
        }
        None => (Vec::new(), None),
    };
    if let Some(t) = tagger {
        let engine = t.engine();
        events.extend(t.tag(pcm).map_err(|message| Error::Engine {
            engine: engine.tool.clone().unwrap_or_default(),
            message,
        })?);
    }
    events.sort_by(|a, b| a.t0.total_cmp(&b.t0).then_with(|| a.label.cmp(&b.label)));
    Ok(Heard {
        duration: pcm.duration(),
        segments,
        speech_engine,
        events,
        music: spans,
        windows,
        profile: opts.profile,
    })
}

#[cfg(test)]
pub(crate) mod fakes {
    //! Engines for tests: fixed answers, no models, no processes.

    use super::*;

    pub struct FakeSpeech(pub Vec<Segment>);

    impl Transcribe for FakeSpeech {
        fn engine(&self) -> Engine {
            Engine {
                tool: Some("fake-whisper".into()),
                model: Some("fake".into()),
            }
        }
        fn transcribe(&self, _: &Path) -> std::result::Result<Vec<Segment>, String> {
            Ok(self.0.clone())
        }
    }

    pub struct Broken;

    impl Transcribe for Broken {
        fn engine(&self) -> Engine {
            Engine {
                tool: Some("broken".into()),
                model: None,
            }
        }
        fn transcribe(&self, _: &Path) -> std::result::Result<Vec<Segment>, String> {
            Err("model not found".into())
        }
    }

    pub struct FakeTagger;

    impl Tag for FakeTagger {
        fn engine(&self) -> Engine {
            Engine {
                tool: Some("fake-tagger".into()),
                model: None,
            }
        }
        fn tag(&self, pcm: &Pcm) -> std::result::Result<Vec<Event>, String> {
            Ok(vec![Event {
                t0: 0.0,
                t1: pcm.duration(),
                label: "tone".into(),
                score: Some(0.9),
                engine: self.engine(),
            }])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fakes::*;
    use super::*;
    use crate::signal::synth;

    fn tone_gap_tone() -> Pcm {
        let rate = 16_000;
        let mut x = synth::sine(440.0, 3.0, rate);
        x.extend(vec![0.0; 3 * rate as usize]);
        x.extend(synth::sine(440.0, 3.0, rate));
        Pcm::new(rate, x).expect("rate")
    }

    #[test]
    fn hears_silence_speech_annotations_and_tags() {
        let speech = FakeSpeech(vec![
            Segment {
                t0: 0.0,
                t1: 3.0,
                text: " Hello there.".into(),
            },
            Segment {
                t0: 6.0,
                t1: 9.0,
                text: "[Music]".into(),
            },
        ]);
        let heard = hear(
            &tone_gap_tone(),
            Path::new("x.wav"),
            Some(&speech),
            Some(&FakeTagger),
            &Options::default(),
        )
        .expect("fakes do not fail");
        assert_eq!(heard.segments.len(), 1);
        assert_eq!(heard.segments[0].text, "Hello there.");
        let labels: Vec<&str> = heard.events.iter().map(|e| e.label.as_str()).collect();
        assert_eq!(labels, ["tone", "silence", "music"]);
        assert!(heard.music.is_empty(), "a steady tone has no pulse");
    }

    #[test]
    fn an_engine_failure_names_the_engine() {
        let err = hear(
            &tone_gap_tone(),
            Path::new("x"),
            Some(&Broken),
            None,
            &Options::default(),
        )
        .expect_err("broken engine");
        assert_eq!(err.to_string(), "broken failed: model not found");
    }
}
