//! Loudness per frame and silent spans.

use crate::pcm::Pcm;

/// Frame length of the level measure: 50 ms.
pub const HOP_SECONDS: f64 = 0.05;
/// Level of digital silence, so `log10(0)` never appears.
pub const FLOOR_DB: f32 = -100.0;

/// RMS level of each 50 ms frame in dBFS (a full-scale sine is about -3 dB).
#[must_use]
pub fn frame_db(pcm: &Pcm) -> Vec<f32> {
    let hop = ((HOP_SECONDS * f64::from(pcm.rate())) as usize).max(1);
    pcm.samples()
        .chunks(hop)
        .map(|c| {
            let mean_sq = c.iter().map(|x| x * x).sum::<f32>() / c.len() as f32;
            if mean_sq > 0.0 {
                (10.0 * mean_sq.log10()).max(FLOOR_DB)
            } else {
                FLOOR_DB
            }
        })
        .collect()
}

/// Spans `(t0, t1)` in seconds where every frame is below `threshold_db` for at least
/// `min_seconds`.
#[must_use]
pub fn silent_spans(db: &[f32], threshold_db: f32, min_seconds: f64) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, &d) in db.iter().chain(std::iter::once(&f32::INFINITY)).enumerate() {
        match (d < threshold_db, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                let (t0, t1) = (s as f64 * HOP_SECONDS, i as f64 * HOP_SECONDS);
                if t1 - t0 >= min_seconds {
                    out.push((t0, t1));
                }
                start = None;
            }
            _ => {}
        }
    }
    out
}

/// Mean level of the frames in `[t0, t1)`, in dB of the mean power (not the mean of dB, so
/// a quiet gap inside loud music barely moves it).
#[must_use]
pub fn mean_db(db: &[f32], t0: f64, t1: f64) -> f32 {
    let (a, b) = (
        (t0 / HOP_SECONDS) as usize,
        ((t1 / HOP_SECONDS) as usize).min(db.len()),
    );
    if a >= b {
        return FLOOR_DB;
    }
    let power = db[a..b].iter().map(|d| 10f32.powf(d / 10.0)).sum::<f32>() / (b - a) as f32;
    (10.0 * power.log10()).max(FLOOR_DB)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::synth;

    #[test]
    fn finds_a_gap_between_two_tones() {
        let rate = 8000;
        let mut x = synth::sine(440.0, 2.0, rate);
        x.extend(vec![0.0; 2 * rate as usize]);
        x.extend(synth::sine(440.0, 2.0, rate));
        let db = frame_db(&Pcm::new(rate, x).expect("rate"));
        let spans = silent_spans(&db, -50.0, 1.0);
        assert_eq!(spans.len(), 1);
        let (t0, t1) = spans[0];
        assert!(
            (t0 - 2.0).abs() < 0.06 && (t1 - 4.0).abs() < 0.06,
            "{t0} {t1}"
        );
    }

    #[test]
    fn short_gaps_and_trailing_silence() {
        let db = [-10.0, -80.0, -10.0, -80.0, -80.0, -80.0];
        let spans = silent_spans(&db, -50.0, 0.1);
        assert_eq!(spans.len(), 1, "the 50 ms gap is a pause: {spans:?}");
        assert!((spans[0].0 - 0.15).abs() < 1e-9 && (spans[0].1 - 0.3).abs() < 1e-9);
        assert!(silent_spans(&db, -50.0, 0.2).is_empty());
    }

    #[test]
    fn a_sine_measures_minus_13_db_at_amplitude_0_3() {
        let db = frame_db(&Pcm::new(8000, synth::sine(440.0, 1.0, 8000)).expect("rate"));
        // 20·log10(0.3/√2) = -13.5
        assert!((mean_db(&db, 0.0, 1.0) + 13.5).abs() < 0.2);
    }
}
