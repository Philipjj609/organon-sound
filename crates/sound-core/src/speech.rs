//! Speech: whisper's JSON output → transcript segments.
//!
//! ffmpeg's `whisper` filter (`format=json`) writes one object per line:
//! `{"start":11720,"end":14920,"text":"…"}`, times in milliseconds counted from the first
//! sample. Whisper also writes what it heard but did not understand as a bracketed line
//! (`[Music]`, `(applause)`, `[BLANK_AUDIO]`, `♪`). Those are not speech, so they become
//! annotations (sound events from the speech engine) instead of transcript.

use serde::Deserialize;

use crate::error::{Error, Result};

/// What was said between two times.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub t0: f64,
    pub t1: f64,
    pub text: String,
}

/// A bracketed annotation from the speech engine (`[Music]` → `music`).
#[derive(Debug, Clone, PartialEq)]
pub struct Annotation {
    pub t0: f64,
    pub t1: f64,
    pub label: String,
}

#[derive(Deserialize)]
struct Line {
    start: f64,
    end: f64,
    text: String,
}

/// Parse whisper JSON lines into raw segments (times in seconds). Blank lines are skipped.
///
/// # Errors
/// A non-blank line is not a whisper segment object.
pub fn parse_whisper_json(text: &str) -> Result<Vec<Segment>> {
    text.trim_start_matches('\u{feff}')
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| {
            let l = l.trim();
            let line = match serde_json::from_str::<Line>(l) {
                Ok(line) => line,
                Err(e) => unescaped_line(l).ok_or_else(|| Error::Whisper {
                    line: i + 1,
                    reason: e.to_string(),
                })?,
            };
            Ok(Segment {
                t0: line.start / 1000.0,
                t1: line.end / 1000.0,
                text: line.text,
            })
        })
        .collect()
}

/// ffmpeg 9.0.1 writes the text without escaping it, so a quote in speech breaks the JSON
/// (`"text":"he says, "So you know""`). The layout is fixed, so the text is everything
/// between `,"text":"` and the closing `"}`.
fn unescaped_line(l: &str) -> Option<Line> {
    let rest = l.strip_prefix("{\"start\":")?;
    let (start, rest) = rest.split_once(",\"end\":")?;
    let (end, rest) = rest.split_once(",\"text\":\"")?;
    let text = rest.strip_suffix("\"}")?;
    Some(Line {
        start: start.trim().parse().ok()?,
        end: end.trim().parse().ok()?,
        text: text.to_owned(),
    })
}

/// Clean raw segments: trim text, drop empty ones, make times monotonic (whisper's chunks
/// can overlap by a few ms), and split off bracketed annotations.
#[must_use]
pub fn tidy(raw: Vec<Segment>) -> (Vec<Segment>, Vec<Annotation>) {
    let (mut said, mut notes) = (Vec::new(), Vec::new());
    let mut last_end = 0.0f64;
    for s in raw {
        let text = s.text.trim();
        if text.is_empty() {
            continue;
        }
        let t0 = s.t0.max(last_end);
        let t1 = s.t1.max(t0);
        last_end = t1;
        match annotation_label(text) {
            Some(label) => notes.push(Annotation { t0, t1, label }),
            None => said.push(Segment {
                t0,
                t1,
                text: text.to_owned(),
            }),
        }
    }
    (said, notes)
}

/// `[Music]` → `music`, `(audience laughing)` → `audience laughing`, `[BLANK_AUDIO]` →
/// `blank audio`, `♪` alone → `music`. Anything else is speech.
fn annotation_label(text: &str) -> Option<String> {
    let inner = [('[', ']'), ('(', ')'), ('*', '*')]
        .iter()
        .find_map(|(open, close)| {
            let inner = text.strip_prefix(*open)?.strip_suffix(*close)?;
            (!inner.contains(*open) && !inner.contains(*close)).then_some(inner)
        });
    let label = match inner {
        Some(inner) => inner.trim().to_lowercase().replace('_', " "),
        None if text
            .chars()
            .all(|c| c == '♪' || c == '♫' || c.is_whitespace()) =>
        {
            "music".to_owned()
        }
        None => return None,
    };
    (!label.is_empty()).then_some(label)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ffmpeg_whisper_json_lines() {
        let out = "{\"start\":0,\"end\":11720,\"text\":\"as draw diagrams.\"}\n\
                   \n{\"start\":11720,\"end\":14920,\"text\":\" The copy module\"}\n";
        let s = parse_whisper_json(out).expect("valid");
        assert_eq!(s.len(), 2);
        assert_eq!((s[1].t0, s[1].t1), (11.72, 14.92));
    }

    #[test]
    fn unescaped_quotes_from_ffmpeg_are_kept() {
        // Regression: line 44 of a real run (ffmpeg 9.0.1) failed to parse.
        let out = r#"{"start":199993,"end":209033,"text":"A few months later your boss comes to you and says, "So you know that program you wrote"}"#;
        let s = parse_whisper_json(out).expect("lenient");
        assert_eq!(
            s[0].text,
            r#"A few months later your boss comes to you and says, "So you know that program you wrote"#
        );
        assert_eq!((s[0].t0, s[0].t1), (199.993, 209.033));
    }

    #[test]
    fn a_bad_line_names_its_number() {
        let err = parse_whisper_json("{\"start\":0,\"end\":1,\"text\":\"a\"}\nnot json")
            .expect_err("line 2 is bad");
        assert!(err.to_string().contains("line 2"), "{err}");
    }

    #[test]
    fn overlaps_are_clamped_and_annotations_split_off() {
        let raw = vec![
            Segment {
                t0: 0.0,
                t1: 20.0,
                text: "printer writer.".into(),
            },
            Segment {
                t0: 19.994,
                t1: 25.9,
                text: " It also exits.".into(),
            },
            Segment {
                t0: 25.9,
                t1: 30.0,
                text: "[Music]".into(),
            },
            Segment {
                t0: 30.0,
                t1: 31.0,
                text: "  ".into(),
            },
            Segment {
                t0: 31.0,
                t1: 32.0,
                text: "[BLANK_AUDIO]".into(),
            },
            Segment {
                t0: 32.0,
                t1: 33.0,
                text: "♪ ♪".into(),
            },
            Segment {
                t0: 33.0,
                t1: 34.0,
                text: "(laughs) yes (laughs)".into(),
            },
        ];
        let (said, notes) = tidy(raw);
        assert_eq!(said[1].t0, 20.0);
        assert_eq!(said[1].text, "It also exits.");
        assert_eq!(
            said.last().map(|s| s.text.as_str()),
            Some("(laughs) yes (laughs)")
        );
        let labels: Vec<&str> = notes.iter().map(|n| n.label.as_str()).collect();
        assert_eq!(labels, ["music", "blank audio", "music"]);
    }
}
