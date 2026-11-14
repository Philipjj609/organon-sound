//! Tempo: an onset envelope (spectral flux), then the lag where it best repeats itself
//! (autocorrelation), weighted towards 120 BPM. This is the classic method (Percival and
//! Tzanetakis 2014 is a refined form of it) that research found adequate for a first organ;
//! its known failure is octave errors (half the tempo) above about 150 BPM.

use crate::pcm::Pcm;
use crate::signal::for_each_spectrum;

/// Onset envelope frames per second (10 ms hops).
pub const FPS: f64 = 100.0;
/// Range searched.
pub const MIN_BPM: f64 = 40.0;
pub const MAX_BPM: f64 = 240.0;

/// How strongly something changes at each 10 ms step: summed positive change of the
/// log spectrum, with its local mean (±0.25 s) removed and negatives cut off.
#[must_use]
pub fn onset_envelope(pcm: &Pcm) -> Vec<f32> {
    let rate = pcm.rate();
    let hop = ((f64::from(rate) / FPS) as usize).max(1);
    let n_fft = ((f64::from(rate) * 0.064) as usize).next_power_of_two();
    let mut prev: Vec<f32> = Vec::new();
    let mut flux = Vec::new();
    for_each_spectrum(pcm.samples(), n_fft, hop, |_, mags| {
        let logs: Vec<f32> = mags.iter().map(|m| (1.0 + 100.0 * m).ln()).collect();
        let f = if prev.is_empty() {
            0.0
        } else {
            logs.iter().zip(&prev).map(|(a, b)| (a - b).max(0.0)).sum()
        };
        flux.push(f);
        prev = logs;
    });
    detrend(&flux, (0.25 * FPS) as usize)
}

fn detrend(x: &[f32], half: usize) -> Vec<f32> {
    let mut prefix = vec![0.0f64; x.len() + 1];
    for (i, v) in x.iter().enumerate() {
        prefix[i + 1] = prefix[i] + f64::from(*v);
    }
    (0..x.len())
        .map(|i| {
            let (a, b) = (i.saturating_sub(half), (i + half + 1).min(x.len()));
            let mean = (prefix[b] - prefix[a]) / (b - a) as f64;
            (x[i] - mean as f32).max(0.0)
        })
        .collect()
}

/// A tempo estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tempo {
    /// Beats per minute.
    pub bpm: f64,
    /// How clearly the envelope repeats at that period: its autocorrelation there, in
    /// `[-1, 1]`. A click track is near 1; speech is low.
    pub pulse: f64,
}

/// Tempo of an onset envelope sampled at [`FPS`]. `None` when it is shorter than two beats
/// at the slowest tempo, or flat.
#[must_use]
pub fn estimate(env: &[f32]) -> Option<Tempo> {
    let min_lag = (60.0 * FPS / MAX_BPM).floor() as usize;
    let max_lag = (60.0 * FPS / MIN_BPM).ceil() as usize;
    if env.len() < 2 * max_lag {
        return None;
    }
    let mean = env.iter().map(|v| f64::from(*v)).sum::<f64>() / env.len() as f64;
    let x: Vec<f64> = env.iter().map(|v| f64::from(*v) - mean).collect();
    let energy = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    if energy <= f64::EPSILON {
        return None;
    }
    let r = |lag: usize| {
        let n = x.len() - lag;
        x[..n]
            .iter()
            .zip(&x[lag..])
            .map(|(a, b)| a * b)
            .sum::<f64>()
            / n as f64
            / energy
    };
    let acf: Vec<f64> = (0..=max_lag + 1)
        .map(|l| if l < min_lag - 1 { 0.0 } else { r(l) })
        .collect();
    // Log-normal prior centred on 120 BPM, one octave wide (as librosa).
    let weighted = |lag: f64| (-0.5 * (60.0 * FPS / lag / 120.0).log2().powi(2)).exp();
    let best = (min_lag..=max_lag)
        .max_by(|&a, &b| (acf[a] * weighted(a as f64)).total_cmp(&(acf[b] * weighted(b as f64))))?;
    // Parabolic refinement of the peak of the raw autocorrelation.
    let (y0, y1, y2) = (acf[best - 1], acf[best], acf[best + 1]);
    let denom = y0 - 2.0 * y1 + y2;
    let shift = if denom.abs() > f64::EPSILON {
        (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    Some(Tempo {
        bpm: 60.0 * FPS / (best as f64 + shift),
        pulse: y1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::synth;

    fn click_track(bpm: f64, seconds: f64) -> Pcm {
        let rate = 16_000;
        let mut x = synth::sine(220.0, seconds, rate);
        x.iter_mut().for_each(|s| *s *= 0.1);
        synth::clicks(&mut x, bpm, rate);
        Pcm::new(rate, x).expect("rate")
    }

    #[test]
    fn finds_the_tempo_of_click_tracks() {
        for bpm in [72.0, 100.0, 120.0, 128.0, 140.0] {
            let t = estimate(&onset_envelope(&click_track(bpm, 20.0))).expect("long enough");
            assert!((t.bpm - bpm).abs() / bpm < 0.02, "{bpm} → {}", t.bpm);
            assert!(t.pulse > 0.5, "pulse {}", t.pulse);
        }
    }

    #[test]
    fn a_steady_tone_has_no_tempo() {
        let pcm = Pcm::new(16_000, synth::sine(440.0, 10.0, 16_000)).expect("rate");
        let t = estimate(&onset_envelope(&pcm));
        assert!(t.is_none_or(|t| t.pulse < 0.3), "{t:?}");
    }

    #[test]
    fn too_short_is_none() {
        assert!(estimate(&[1.0; 100]).is_none());
    }
}
