//! Frame scores of an audio tagger (YAMNet: 521 AudioSet classes, one frame every 0.48 s,
//! each looking at 0.96 s) → labelled events: frames where a class scores at least the
//! threshold, merged while the class stays on.

use crate::hear::{Engine, Event};

/// How frames become events.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TagRules {
    /// Seconds between frames.
    pub hop: f64,
    /// Seconds each frame covers.
    pub window: f64,
    /// Minimum score for a class to be on in a frame.
    pub threshold: f32,
    /// Off-frames tolerated inside one event.
    pub max_gap: usize,
}

impl TagRules {
    /// YAMNet's framing, with a threshold of 0.3 on its sigmoid scores.
    pub const YAMNET: Self = Self {
        hop: 0.48,
        window: 0.96,
        threshold: 0.3,
        max_gap: 1,
    };
}

/// Events from per-frame class scores. `labels[c]` names class `c`; classes in `skip` (e.g.
/// `Silence`, measured better by level) are ignored. Each event's score is its class's best
/// frame. Events are ordered by start, then label.
#[must_use]
pub fn events(
    scores: &[Vec<f32>],
    labels: &[String],
    skip: &[&str],
    rules: &TagRules,
    engine: &Engine,
) -> Vec<Event> {
    let mut out = Vec::new();
    for (c, label) in labels.iter().enumerate() {
        if skip.contains(&label.as_str()) {
            continue;
        }
        // (first frame, last on-frame, best score)
        let mut open: Option<(usize, usize, f32)> = None;
        for (f, frame) in scores.iter().enumerate() {
            let s = frame.get(c).copied().unwrap_or(0.0);
            open = match (s >= rules.threshold, open) {
                (true, None) => Some((f, f, s)),
                (true, Some((a, _, best))) => Some((a, f, best.max(s))),
                (false, Some((a, last, best))) if f - last > rules.max_gap => {
                    out.push(event(a, last, best, label, rules, engine));
                    None
                }
                (_, o) => o,
            };
        }
        if let Some((a, last, best)) = open {
            out.push(event(a, last, best, label, rules, engine));
        }
    }
    out.sort_by(|a, b| a.t0.total_cmp(&b.t0).then_with(|| a.label.cmp(&b.label)));
    out
}

fn event(first: usize, last: usize, score: f32, label: &str, r: &TagRules, e: &Engine) -> Event {
    Event {
        t0: first as f64 * r.hop,
        t1: last as f64 * r.hop + r.window,
        label: label.to_lowercase(),
        score: Some(f64::from(score)),
        engine: e.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels() -> Vec<String> {
        ["Speech", "Music", "Silence"].map(String::from).to_vec()
    }

    #[test]
    fn frames_merge_into_events_across_one_gap() {
        let s = |speech: f32, music: f32| vec![speech, music, 0.9];
        let scores = vec![
            s(0.9, 0.0),
            s(0.8, 0.0),
            s(0.1, 0.5), // one off-frame for speech: still one event
            s(0.7, 0.6),
            s(0.0, 0.0),
            s(0.0, 0.0),
            s(0.4, 0.0),
        ];
        let ev = events(
            &scores,
            &labels(),
            &["Silence"],
            &TagRules::YAMNET,
            &Engine::default(),
        );
        let got: Vec<(String, f64, f64)> =
            ev.iter().map(|e| (e.label.clone(), e.t0, e.t1)).collect();
        assert_eq!(
            got,
            [
                ("speech".into(), 0.0, 3.0 * 0.48 + 0.96),
                ("music".into(), 2.0 * 0.48, 3.0 * 0.48 + 0.96),
                ("speech".into(), 6.0 * 0.48, 6.0 * 0.48 + 0.96),
            ]
        );
        assert_eq!(ev[0].score, Some(f64::from(0.9f32)));
    }

    #[test]
    fn nothing_above_threshold_is_no_event() {
        let scores = vec![vec![0.1, 0.2, 0.0]; 5];
        assert!(events(
            &scores,
            &labels(),
            &[],
            &TagRules::YAMNET,
            &Engine::default()
        )
        .is_empty());
    }
}

/// Class names from a YAMNet class map (`index,mid,display_name`, names may be quoted and
/// contain commas). The header line is skipped.
#[must_use]
pub fn parse_class_map(csv: &str) -> Vec<String> {
    csv.trim_start_matches('\u{feff}')
        .lines()
        .skip(1)
        .filter_map(|l| {
            let mut parts = l.splitn(3, ',');
            parts.next()?.trim().parse::<usize>().ok()?;
            parts.next()?;
            let name = parts.next()?.trim();
            Some(name.trim_matches('"').replace("\"\"", "\""))
        })
        .collect()
}

#[cfg(test)]
mod class_map_tests {
    use super::*;

    #[test]
    fn quoted_names_keep_their_commas() {
        let csv = "index,mid,display_name\r\n0,/m/09x0r,Speech\r\n1,/m/0ytgt,\"Child speech, kid speaking\"\r\n";
        assert_eq!(
            parse_class_map(csv),
            ["Speech", "Child speech, kid speaking"]
        );
    }
}
