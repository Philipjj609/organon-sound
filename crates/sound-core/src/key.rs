//! Musical key: a chroma vector (energy per pitch class) correlated with the 24 rotations of
//! a major and a minor key profile. Which profile fits depends on the music (research:
//! Krumhansl for pop, Temperley for classical), so the profile is a choice, and the result
//! carries its correlation and its margin over the runner-up.

use std::fmt;

use crate::pcm::Pcm;
use crate::signal::{bin_hz, for_each_spectrum};

/// Pitch-class names, C = 0.
pub const PITCH_CLASSES: [&str; 12] = [
    "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B",
];
/// Lowest and highest frequency counted (C2 to about C7).
const MIN_HZ: f64 = 65.0;
const MAX_HZ: f64 = 2100.0;
/// Chroma frames per second at the default hop (0.256 s at 16 kHz).
const FRAME_SECONDS: f64 = 0.256;

/// Major or minor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Major,
    Minor,
}

impl Mode {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Major => "major",
            Self::Minor => "minor",
        }
    }
}

/// Key profile family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Profile {
    /// Krumhansl and Kessler (1982), from listening experiments; general and pop music.
    #[default]
    Krumhansl,
    /// Temperley (1999), from a corpus of classical music; stronger on minor keys there.
    Temperley,
}

impl Profile {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Krumhansl => "krumhansl",
            Self::Temperley => "temperley",
        }
    }

    /// Major and minor weights, tonic first.
    const fn weights(self) -> ([f64; 12], [f64; 12]) {
        match self {
            Self::Krumhansl => (
                [
                    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
                ],
                [
                    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
                ],
            ),
            Self::Temperley => (
                [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0],
                [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0],
            ),
        }
    }
}

/// The best-fitting key.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Key {
    /// Pitch class of the tonic, C = 0.
    pub tonic: usize,
    pub mode: Mode,
    /// Pearson correlation of the chroma with this key's profile.
    pub corr: f64,
    /// How far ahead of the second-best key it is (relative and parallel keys are close by
    /// nature, so a small margin is common and worth showing).
    pub margin: f64,
}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", PITCH_CLASSES[self.tonic], self.mode.as_str())
    }
}

/// Chroma frames of a recording: summed spectral magnitude per pitch class, one frame every
/// [`FRAME_SECONDS`] (by sample count, so at any rate).
#[must_use]
pub fn chroma_frames(pcm: &Pcm) -> Vec<[f32; 12]> {
    let rate = pcm.rate();
    // About 2 Hz per bin at 16 kHz: a semitone near C2 is 3.9 Hz wide.
    let n_fft = ((f64::from(rate) * 0.5) as usize).next_power_of_two();
    let hop = ((FRAME_SECONDS * f64::from(rate)) as usize).max(1);
    let classes: Vec<Option<usize>> = (0..=n_fft / 2)
        .map(|k| {
            let hz = bin_hz(k, n_fft, rate);
            (MIN_HZ..=MAX_HZ).contains(&hz).then(|| {
                let midi = 69.0 + 12.0 * (hz / 440.0).log2();
                (midi.round() as i64).rem_euclid(12) as usize
            })
        })
        .collect();
    let mut out = Vec::new();
    for_each_spectrum(pcm.samples(), n_fft, hop, |_, mags| {
        let mut c = [0.0f32; 12];
        for (m, class) in mags.iter().zip(&classes) {
            if let Some(pc) = class {
                c[*pc] += m;
            }
        }
        out.push(c);
    });
    out
}

/// Seconds per chroma frame.
#[must_use]
pub fn frame_seconds(pcm: &Pcm) -> f64 {
    ((FRAME_SECONDS * f64::from(pcm.rate())) as usize).max(1) as f64 / f64::from(pcm.rate())
}

/// Sum of chroma frames.
#[must_use]
pub fn sum(frames: &[[f32; 12]]) -> [f32; 12] {
    frames.iter().fold([0.0; 12], |mut acc, f| {
        acc.iter_mut().zip(f).for_each(|(a, b)| *a += b);
        acc
    })
}

/// Best key for a chroma vector. `None` when the chroma is flat (silence, noise).
#[must_use]
pub fn estimate(chroma: &[f32; 12], profile: Profile) -> Option<Key> {
    let (major, minor) = profile.weights();
    let mut scores: Vec<(f64, usize, Mode)> = Vec::with_capacity(24);
    for tonic in 0..12 {
        for (weights, mode) in [(&major, Mode::Major), (&minor, Mode::Minor)] {
            let rotated: [f64; 12] = std::array::from_fn(|pc| weights[(pc + 12 - tonic) % 12]);
            scores.push((pearson(chroma, &rotated)?, tonic, mode));
        }
    }
    scores.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (corr, tonic, mode) = scores[0];
    Some(Key {
        tonic,
        mode,
        corr,
        margin: corr - scores[1].0,
    })
}

fn pearson(x: &[f32; 12], y: &[f64; 12]) -> Option<f64> {
    let mx = x.iter().map(|v| f64::from(*v)).sum::<f64>() / 12.0;
    let my = y.iter().sum::<f64>() / 12.0;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        let (dx, dy) = (f64::from(*a) - mx, b - my);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    (sxx > 1e-12 && syy > 1e-12).then(|| sxy / (sxx * syy).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::synth;

    /// A I–IV–V–I progression in a key, as triads with a bass note.
    fn cadence(tonic: i32, minor: bool) -> Pcm {
        let third = if minor { 3 } else { 4 };
        let rate = 16_000;
        let mut x = Vec::new();
        for root in [0, 5, 7, 0] {
            let t = tonic + root;
            // The V chord is major in both modes (harmonic minor).
            let th = if root == 7 { 4 } else { third };
            let iv_third = if root == 5 && minor { 3 } else { th };
            let notes = [t - 12, t, t + iv_third, t + 7].map(synth::midi);
            x.extend(synth::chord(&notes, 1.5, rate));
        }
        Pcm::new(rate, x).expect("rate")
    }

    #[test]
    fn finds_major_and_minor_keys_of_cadences() {
        for (tonic, minor, want) in [
            (60, false, "C major"),
            (67, false, "G major"),
            (62, false, "D major"),
            (57, true, "A minor"),
            (64, true, "E minor"),
        ] {
            let pcm = cadence(tonic, minor);
            let key = estimate(&sum(&chroma_frames(&pcm)), Profile::Krumhansl).expect("tonal");
            assert_eq!(key.to_string(), want, "corr {}", key.corr);
            assert!(key.corr > 0.6, "{want}: corr {}", key.corr);
        }
    }

    #[test]
    fn silence_has_no_key() {
        let pcm = Pcm::new(16_000, vec![0.0; 32_000]).expect("rate");
        assert!(estimate(&sum(&chroma_frames(&pcm)), Profile::Krumhansl).is_none());
    }

    #[test]
    fn a_single_note_class_maps_to_its_pitch_class() {
        let pcm = Pcm::new(16_000, synth::sine(synth::midi(57), 1.0, 16_000)).expect("rate");
        let c = sum(&chroma_frames(&pcm));
        let top = (0..12)
            .max_by(|&a, &b| c[a].total_cmp(&c[b]))
            .expect("12 classes");
        assert_eq!(PITCH_CLASSES[top], "A");
    }
}
