//! What was heard as organon-protocol messages: the source (and its anchor, when this organ
//! sensed it), then located observations.
//!
//! | kind | at | value |
//! |---|---|---|
//! | `transcript_segment` | time span | `{text}` |
//! | `music` | time span | `{bpm, pulse, key, key_corr, key_margin, profile}` (measured ones) |
//! | `sound_event` | time span | `{label, score?}` |
//!
//! Times are rounded to the millisecond and measures to three decimals, because ids hash them.
//! Who measured (ffmpeg's whisper, a tagger, this organ) is in `produced_by`, not the value,
//! so two engines that agree make the same observation id.

use organon_protocol::{
    Anchor, Location, Message, Observation, ObservationKind, ProducedBy, Source,
};
use serde_json::{json, Map, Value};

use crate::hear::{Engine, Heard};
use crate::pcm::{ms, r3};

/// Organ name in `produced_by`.
pub const ORGAN: &str = "organon-sound";
/// A stretch of music with its tempo and key.
pub const MUSIC: &str = "music";
/// A labelled sound: silence, an annotation of the speech engine, a tag.
pub const SOUND_EVENT: &str = "sound_event";

/// Messages for one hearing: `source`, `anchor` if given, then observations in time order
/// within each kind (speech, music, events).
#[must_use]
pub fn messages(
    heard: &Heard,
    source: &Source,
    anchor: Option<&Anchor>,
    version: &str,
) -> Vec<Message> {
    let by = |e: &Engine| ProducedBy {
        organ: ORGAN.to_owned(),
        version: version.to_owned(),
        tool: e.tool.clone(),
        model: e.model.clone(),
    };
    let at = |t0: f64, t1: f64| Location::Time {
        t0: ms(t0),
        t1: ms(t1),
    };
    let observe = |kind: &str, loc: Location, value: Value, e: &Engine| {
        Message::Observation(Observation::new(
            source.id.clone(),
            ObservationKind::new(kind),
            loc,
            value,
            by(e),
        ))
    };
    let mut out = vec![Message::Source(source.clone())];
    out.extend(anchor.cloned().map(Message::Anchor));
    if let Some(engine) = &heard.speech_engine {
        out.extend(heard.segments.iter().map(|s| {
            observe(
                ObservationKind::TRANSCRIPT_SEGMENT,
                at(s.t0, s.t1),
                json!({ "text": s.text }),
                engine,
            )
        }));
    }
    out.extend(heard.music.iter().map(|span| {
        let mut v = Map::new();
        if let Some(t) = span.tempo {
            v.insert("bpm".into(), json!(r3(t.bpm)));
            v.insert("pulse".into(), json!(r3(t.pulse)));
        }
        if let Some(k) = span.key {
            v.insert("key".into(), json!(k.to_string()));
            v.insert("key_corr".into(), json!(r3(k.corr)));
            v.insert("key_margin".into(), json!(r3(k.margin)));
            v.insert("profile".into(), json!(heard.profile.as_str()));
        }
        observe(
            MUSIC,
            at(span.t0, span.t1),
            Value::Object(v),
            &Engine::default(),
        )
    }));
    out.extend(heard.events.iter().map(|e| {
        let mut v = json!({ "label": e.label });
        if let Some(s) = e.score {
            v["score"] = json!(r3(s));
        }
        observe(SOUND_EVENT, at(e.t0, e.t1), v, &e.engine)
    }));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hear::Event;
    use crate::key::{Key, Mode, Profile};
    use crate::music::Span;
    use crate::speech::Segment;
    use crate::tempo::Tempo;
    use organon_protocol::SourceId;
    use std::collections::BTreeMap;

    fn source() -> Source {
        Source {
            id: SourceId::of(b"file:abc"),
            kind: "audio".into(),
            uri: "song.mp3".into(),
            title: "song".into(),
            captured_at: "2026-11-14T00:00:00Z".into(),
            content_hash: "00".into(),
            files: BTreeMap::new(),
            meta: BTreeMap::new(),
            produced_by: None,
        }
    }

    fn heard() -> Heard {
        let whisper = Engine {
            tool: Some("ffmpeg whisper".into()),
            model: Some("ggml-base".into()),
        };
        Heard {
            duration: 60.0,
            segments: vec![Segment {
                t0: 1.000_4,
                t1: 2.5,
                text: "hi".into(),
            }],
            speech_engine: Some(whisper.clone()),
            events: vec![Event {
                t0: 50.0,
                t1: 60.0,
                label: "silence".into(),
                score: None,
                engine: Engine::default(),
            }],
            music: vec![Span {
                t0: 5.0,
                t1: 45.0,
                tempo: Some(Tempo {
                    bpm: 120.123_4,
                    pulse: 0.5,
                }),
                key: Some(Key {
                    tonic: 9,
                    mode: Mode::Minor,
                    corr: 0.81,
                    margin: 0.05,
                }),
            }],
            windows: Vec::new(),
            profile: Profile::Krumhansl,
        }
    }

    #[test]
    fn emits_source_then_located_observations() {
        let msgs = messages(&heard(), &source(), None, "0.1.0");
        let lines: Vec<Value> = msgs
            .iter()
            .map(|m| serde_json::to_value(m).expect("serializes"))
            .collect();
        assert_eq!(lines[0]["type"], "source");
        assert_eq!(lines[1]["kind"], "transcript_segment");
        assert_eq!(lines[1]["at"], json!({"t0": 1.0, "t1": 2.5}));
        assert_eq!(lines[1]["produced_by"]["model"], "ggml-base");
        assert_eq!(lines[2]["kind"], "music");
        assert_eq!(lines[2]["value"]["key"], "A minor");
        assert_eq!(lines[2]["value"]["bpm"], 120.123);
        assert_eq!(lines[3]["value"], json!({"label": "silence"}));
        assert!(lines[3]["produced_by"].get("tool").is_none());
        for m in &msgs {
            if let Message::Observation(o) = m {
                assert!(o.id_is_consistent());
            }
        }
    }

    #[test]
    fn ids_do_not_depend_on_the_engine() {
        let a = messages(&heard(), &source(), None, "0.1.0");
        let mut other = heard();
        other.speech_engine = Some(Engine {
            tool: Some("whisper-rs".into()),
            model: None,
        });
        let b = messages(&other, &source(), None, "0.2.0");
        let id = |m: &Message| match m {
            Message::Observation(o) => Some(o.id.clone()),
            _ => None,
        };
        assert_eq!(
            a.iter().map(id).collect::<Vec<_>>(),
            b.iter().map(id).collect::<Vec<_>>()
        );
    }
}
