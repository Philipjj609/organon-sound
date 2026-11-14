//! Short-time spectra, streamed: each frame's magnitudes are handed to a closure and dropped,
//! so a long recording never holds a whole spectrogram (13 minutes at 10 ms hops would be
//! about 100 MB).

use std::f32::consts::PI;

use rustfft::{num_complex::Complex, FftPlanner};

/// Hann-windowed magnitude spectra of `samples`, `n_fft` long every `hop` samples. Calls
/// `each(frame_index, magnitudes)` with `n_fft / 2 + 1` magnitudes per frame. The last frame
/// is zero-padded.
pub fn for_each_spectrum(
    samples: &[f32],
    n_fft: usize,
    hop: usize,
    mut each: impl FnMut(usize, &[f32]),
) {
    if samples.is_empty() || n_fft == 0 || hop == 0 {
        return;
    }
    let fft = FftPlanner::<f32>::new().plan_fft_forward(n_fft);
    let window: Vec<f32> = (0..n_fft)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / n_fft as f32).cos())
        .collect();
    let mut buf = vec![Complex::new(0.0, 0.0); n_fft];
    let mut mags = vec![0.0; n_fft / 2 + 1];
    for (index, start) in (0..samples.len()).step_by(hop).enumerate() {
        for (i, b) in buf.iter_mut().enumerate() {
            let x = samples.get(start + i).copied().unwrap_or(0.0);
            *b = Complex::new(x * window[i], 0.0);
        }
        fft.process(&mut buf);
        for (m, b) in mags.iter_mut().zip(&buf) {
            *m = b.norm();
        }
        each(index, &mags);
    }
}

/// Frequency of FFT bin `k`.
#[must_use]
pub fn bin_hz(k: usize, n_fft: usize, rate: u32) -> f64 {
    k as f64 * f64::from(rate) / n_fft as f64
}

#[cfg(test)]
pub(crate) mod synth {
    //! Synthetic signals for tests: written here, never recorded.

    use std::f64::consts::TAU;

    /// A sine of `hz` for `seconds`.
    pub fn sine(hz: f64, seconds: f64, rate: u32) -> Vec<f32> {
        let n = (seconds * f64::from(rate)) as usize;
        (0..n)
            .map(|i| (0.3 * (TAU * hz * i as f64 / f64::from(rate)).sin()) as f32)
            .collect()
    }

    /// Mix of sines (a chord), each at equal level.
    pub fn chord(hz: &[f64], seconds: f64, rate: u32) -> Vec<f32> {
        let parts: Vec<Vec<f32>> = hz.iter().map(|&f| sine(f, seconds, rate)).collect();
        (0..parts[0].len())
            .map(|i| parts.iter().map(|p| p[i]).sum::<f32>() / hz.len() as f32)
            .collect()
    }

    /// Short decaying noise bursts (clicks) at `bpm`, added onto `base`.
    pub fn clicks(base: &mut [f32], bpm: f64, rate: u32) {
        let period = (60.0 / bpm * f64::from(rate)) as usize;
        let len = (rate / 50) as usize; // 20 ms
        let mut seed = 0x2545_f491_u32;
        for start in (0..base.len()).step_by(period) {
            for i in 0..len.min(base.len() - start) {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                let noise = (seed as f32 / u32::MAX as f32) * 2.0 - 1.0;
                base[start + i] += 0.8 * noise * (1.0 - i as f32 / len as f32);
            }
        }
    }

    /// Frequency of a MIDI note.
    pub fn midi(note: i32) -> f64 {
        440.0 * 2f64.powf(f64::from(note - 69) / 12.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sine_peaks_in_its_bin() {
        let rate = 16_000;
        let x = synth::sine(1000.0, 0.5, rate);
        let mut peak = 0;
        for_each_spectrum(&x, 1024, 512, |i, m| {
            if i == 2 {
                peak = m
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map_or(0, |(k, _)| k);
            }
        });
        assert!((bin_hz(peak, 1024, rate) - 1000.0).abs() < 16.0);
    }

    #[test]
    fn frames_cover_the_signal() {
        let mut n = 0;
        for_each_spectrum(&[0.0; 1000], 256, 100, |_, _| n += 1);
        assert_eq!(n, 10);
    }
}
