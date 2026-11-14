//! Mono audio samples.

use crate::error::{Error, Result};

/// Mono samples, nominally in `[-1, 1]`, at a fixed rate.
#[derive(Debug, Clone, PartialEq)]
pub struct Pcm {
    rate: u32,
    samples: Vec<f32>,
}

impl Pcm {
    /// # Errors
    /// The rate is zero.
    pub fn new(rate: u32, samples: Vec<f32>) -> Result<Self> {
        if rate == 0 {
            return Err(Error::Audio("sample rate 0".into()));
        }
        Ok(Self { rate, samples })
    }

    /// Raw little-endian `f32` samples, as `ffmpeg -f f32le` writes them. A trailing partial
    /// sample is dropped.
    ///
    /// # Errors
    /// The rate is zero.
    pub fn from_f32le(rate: u32, bytes: &[u8]) -> Result<Self> {
        let samples = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Self::new(rate, samples)
    }

    #[must_use]
    pub const fn rate(&self) -> u32 {
        self.rate
    }

    #[must_use]
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// Length in seconds.
    #[must_use]
    pub fn duration(&self) -> f64 {
        self.samples.len() as f64 / f64::from(self.rate)
    }
}

/// Seconds rounded to the millisecond. Observation ids hash their numbers, so every time
/// goes through this before it becomes an observation.
#[must_use]
pub fn ms(t: f64) -> f64 {
    (t * 1000.0).round() / 1000.0
}

/// A measure rounded to three decimals, for the same reason as [`ms`].
#[must_use]
pub fn r3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32le_round_trips_and_drops_partial_samples() {
        let mut bytes: Vec<u8> = [0.5f32, -0.25]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        bytes.push(7);
        let pcm = Pcm::from_f32le(16_000, &bytes).expect("rate is not zero");
        assert_eq!(pcm.samples(), &[0.5, -0.25]);
    }

    #[test]
    fn zero_rate_is_an_error() {
        assert!(Pcm::new(0, vec![0.0]).is_err());
    }

    #[test]
    fn rounding_is_stable() {
        assert_eq!(ms(1.234_56), 1.235);
        assert_eq!(r3(0.123_44), 0.123);
    }
}
