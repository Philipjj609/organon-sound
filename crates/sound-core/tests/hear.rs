//! End to end through the public API: synthesized audio and a fake transcriber in, protocol
//! messages out.

use std::f64::consts::TAU;
use std::path::Path;

use organon_protocol::{Message, Source, SourceId};
use organon_sound_core::emit::{messages, MUSIC, SOUND_EVENT};
use organon_sound_core::{hear, Engine, Options, Pcm, Segment, Transcribe};

const RATE: u32 = 16_000;

/// 12 s of silence, then 40 s of an E minor triad with a click every 0.5 s (120 BPM).
fn recording() -> Pcm {
    let mut x = vec![0.0f32; 12 * RATE as usize];
    let n = 40 * RATE as usize;
    let notes = [64.0, 67.0, 71.0, 52.0].map(|m: f64| 440.0 * 2f64.powf((m - 69.0) / 12.0));
    let period = RATE as usize / 2;
    x.extend((0..n).map(|i| {
        let t = i as f64 / f64::from(RATE);
        let tone: f64 = notes.iter().map(|f| (TAU * f * t).sin()).sum::<f64>() * 0.08;
        let k = i % period;
        let click = if k < 160 {
            0.6 * (1.0 - k as f64 / 160.0) * ((i * 7919) % 13) as f64 / 13.0
        } else {
            0.0
        };
        (tone + click) as f32
    }));
    Pcm::new(RATE, x).expect("rate is not zero")
}

struct Says;

impl Transcribe for Says {
    fn engine(&self) -> Engine {
        Engine {
            tool: Some("fake".into()),
            model: None,
        }
    }
    fn transcribe(&self, _: &Path) -> Result<Vec<Segment>, String> {
        Ok(vec![
            Segment {
                t0: 0.0,
                t1: 4.0,
                text: "[BLANK_AUDIO]".into(),
            },
            Segment {
                t0: 12.0,
                t1: 20.0,
                text: "la la".into(),
            },
        ])
    }
}

#[test]
fn a_recording_becomes_located_observations() {
    let heard = hear(
        &recording(),
        Path::new("rec.wav"),
        Some(&Says),
        None,
        &Options::default(),
    )
    .expect("fake engines do not fail");
    let source = Source {
        id: SourceId::of(b"file:test"),
        kind: "audio".into(),
        uri: "rec.wav".into(),
        title: "rec".into(),
        captured_at: "2026-11-14T00:00:00Z".into(),
        content_hash: "00".into(),
        files: Default::default(),
        meta: Default::default(),
        produced_by: None,
    };
    let msgs = messages(&heard, &source, None, "0.1.0");
    let obs: Vec<_> = msgs
        .iter()
        .filter_map(|m| match m {
            Message::Observation(o) => Some(o),
            _ => None,
        })
        .collect();
    assert!(obs
        .iter()
        .all(|o| o.id_is_consistent() && o.source == source.id));

    let music: Vec<_> = obs.iter().filter(|o| o.kind.as_str() == MUSIC).collect();
    assert_eq!(music.len(), 1, "{:#?}", heard.windows);
    assert_eq!(music[0].value["key"], "E minor");
    let bpm = music[0].value["bpm"].as_f64().expect("bpm");
    assert!((bpm - 120.0).abs() < 2.5, "bpm {bpm}");

    let labels: Vec<&str> = obs
        .iter()
        .filter(|o| o.kind.as_str() == SOUND_EVENT)
        .filter_map(|o| o.value["label"].as_str())
        .collect();
    assert_eq!(labels, ["blank audio", "silence"]);
    let said: Vec<_> = obs
        .iter()
        .filter(|o| o.kind.as_str() == "transcript_segment")
        .collect();
    assert_eq!(said.len(), 1);
}
