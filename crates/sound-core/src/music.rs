//! Where there is music. Each window of the recording gets two measures: **pulse** (how
//! clearly its onsets repeat at one period) and **tonality** (how well its chroma fits a key).
//! Windows clear on both are music; neighbouring music windows merge into a span, and the
//! span's tempo and key are measured again over all of it.

use crate::key::{self, Key, Profile};
use crate::level;
use crate::tempo::{self, Tempo};

/// Window length and step, in seconds.
pub const WINDOW: f64 = 10.0;
pub const STEP: f64 = 5.0;

/// When a window counts as music. Defaults are from runs on real recordings (DESIGN.md).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Thresholds {
    /// Minimum pulse (autocorrelation at the tempo).
    pub pulse: f64,
    /// Minimum key correlation, together with the pulse.
    pub tonality: f64,
    /// Key correlation that makes a window music without any pulse (ambient intros, rubato).
    /// Speech reached 0.71 at most in a 13-minute talk, so this sits above it.
    pub tonal_alone: f64,
    /// Windows quieter than this (dBFS) are never music.
    pub min_db: f32,
    /// Spans shorter than this are dropped.
    pub min_span: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            pulse: 0.3,
            tonality: 0.6,
            tonal_alone: 0.8,
            min_db: -45.0,
            min_span: 10.0,
        }
    }
}

/// The measures of one window.
#[derive(Debug, Clone, PartialEq)]
pub struct Window {
    pub t0: f64,
    pub t1: f64,
    pub tempo: Option<Tempo>,
    pub key: Option<Key>,
    pub db: f32,
    pub music: bool,
}

/// A stretch of music with its tempo and key.
#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    pub t0: f64,
    pub t1: f64,
    pub tempo: Option<Tempo>,
    pub key: Option<Key>,
}

/// What the windows are measured from: features of the whole recording, computed once.
#[derive(Debug, Clone)]
pub struct Features {
    /// Onset envelope at [`tempo::FPS`].
    pub onsets: Vec<f32>,
    /// Chroma frames, `chroma_seconds` apart.
    pub chroma: Vec<[f32; 12]>,
    pub chroma_seconds: f64,
    /// Level per [`level::HOP_SECONDS`].
    pub db: Vec<f32>,
    pub duration: f64,
}

impl Features {
    fn tempo(&self, t0: f64, t1: f64) -> Option<Tempo> {
        let (a, b) = range(t0, t1, 1.0 / tempo::FPS, self.onsets.len());
        tempo::estimate(&self.onsets[a..b])
    }

    fn key(&self, t0: f64, t1: f64, profile: Profile) -> Option<Key> {
        let (a, b) = range(t0, t1, self.chroma_seconds, self.chroma.len());
        key::estimate(&key::sum(&self.chroma[a..b]), profile)
    }
}

fn range(t0: f64, t1: f64, step: f64, len: usize) -> (usize, usize) {
    let a = ((t0 / step) as usize).min(len);
    let b = ((t1 / step).ceil() as usize).clamp(a, len);
    (a, b)
}

/// Measure every window. A recording shorter than one window is one window.
#[must_use]
pub fn windows(f: &Features, profile: Profile, th: &Thresholds) -> Vec<Window> {
    let mut out = Vec::new();
    let mut t0 = 0.0;
    loop {
        let t1 = (t0 + WINDOW).min(f.duration);
        let (tempo, key, db) = (
            f.tempo(t0, t1),
            f.key(t0, t1, profile),
            level::mean_db(&f.db, t0, t1),
        );
        let corr = key.map_or(f64::NEG_INFINITY, |k| k.corr);
        let pulsed = tempo.is_some_and(|t| t.pulse >= th.pulse);
        let music = db >= th.min_db && (corr >= th.tonal_alone || (pulsed && corr >= th.tonality));
        out.push(Window {
            t0,
            t1,
            tempo,
            key,
            db,
            music,
        });
        if t1 >= f.duration {
            break;
        }
        t0 += STEP;
    }
    out
}

/// Merge overlapping music windows into spans, measure each span whole, drop short ones.
#[must_use]
pub fn spans(f: &Features, windows: &[Window], profile: Profile, th: &Thresholds) -> Vec<Span> {
    let mut merged: Vec<(f64, f64)> = Vec::new();
    for w in windows.iter().filter(|w| w.music) {
        match merged.last_mut() {
            Some(last) if w.t0 <= last.1 => last.1 = last.1.max(w.t1),
            _ => merged.push((w.t0, w.t1)),
        }
    }
    merged
        .into_iter()
        .filter(|(a, b)| b - a >= th.min_span)
        .map(|(t0, t1)| Span {
            t0,
            t1,
            tempo: f.tempo(t0, t1),
            key: f.key(t0, t1, profile),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcm::Pcm;
    use crate::signal::synth;

    fn features(pcm: &Pcm) -> Features {
        Features {
            onsets: tempo::onset_envelope(pcm),
            chroma: key::chroma_frames(pcm),
            chroma_seconds: key::frame_seconds(pcm),
            db: level::frame_db(pcm),
            duration: pcm.duration(),
        }
    }

    #[test]
    fn music_between_silences_is_one_span_with_tempo_and_key() {
        let rate = 16_000;
        let mut x = vec![0.0; 15 * rate as usize];
        let notes = [57, 60, 64].map(synth::midi); // A minor triad
        let mut song = synth::chord(&notes, 30.0, rate);
        synth::clicks(&mut song, 110.0, rate);
        x.extend(song);
        x.extend(vec![0.0; 15 * rate as usize]);
        let pcm = Pcm::new(rate, x).expect("rate");
        let f = features(&pcm);
        let th = Thresholds::default();
        let w = windows(&f, Profile::Krumhansl, &th);
        let s = spans(&f, &w, Profile::Krumhansl, &th);
        assert_eq!(s.len(), 1, "{w:#?}");
        let span = &s[0];
        assert!(span.t0 >= 5.0 && span.t0 <= 15.0 && span.t1 >= 40.0 && span.t1 <= 50.0);
        let tempo = span.tempo.expect("tempo");
        assert!((tempo.bpm - 110.0).abs() < 2.0, "{tempo:?}");
        assert_eq!(span.key.expect("key").to_string(), "A minor");
    }

    #[test]
    fn a_beatless_tonal_pad_is_music() {
        // Regression: a song's 70 s intro without drums (r 0.8–0.96, pulse 0.1–0.2) was
        // missed when every music window needed a pulse.
        let rate = 16_000;
        let mut x = Vec::new();
        for _ in 0..4 {
            for root in [0, 5, 7, 0] {
                let t = 60 + root;
                // I, IV and V of C major are all major triads.
                x.extend(synth::chord(
                    &[t - 12, t, t + 4, t + 7].map(synth::midi),
                    2.0,
                    rate,
                ));
            }
        }
        let pcm = Pcm::new(rate, x).expect("rate");
        let f = features(&pcm);
        let th = Thresholds::default();
        let w = windows(&f, Profile::Krumhansl, &th);
        assert!(
            w.iter().all(|w| w.tempo.is_none_or(|t| t.pulse < th.pulse)),
            "{w:#?}"
        );
        let s = spans(&f, &w, Profile::Krumhansl, &th);
        assert_eq!(s.len(), 1, "{w:#?}");
        assert_eq!(s[0].key.expect("key").to_string(), "C major");
    }

    #[test]
    fn silence_is_not_music() {
        let pcm = Pcm::new(16_000, vec![0.0; 16_000 * 30]).expect("rate");
        let f = features(&pcm);
        let w = windows(&f, Profile::Krumhansl, &Thresholds::default());
        assert!(w.iter().all(|w| !w.music));
    }

    #[test]
    fn short_recordings_are_one_window() {
        let pcm = Pcm::new(16_000, vec![0.0; 16_000 * 3]).expect("rate");
        let w = windows(&features(&pcm), Profile::Krumhansl, &Thresholds::default());
        assert_eq!(w.len(), 1);
        assert_eq!((w[0].t0, w[0].t1), (0.0, 3.0));
    }
}
