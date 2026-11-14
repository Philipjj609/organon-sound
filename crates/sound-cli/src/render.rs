//! Text for the reader: `index.md` (read first), `transcript.md`, `windows.tsv`.

use std::fmt::Write as _;

use organon_sound_core::{Event, Heard};

/// Overlapping events shown as one moment, with its top labels.
struct Group {
    t0: f64,
    t1: f64,
    labels: String,
}

/// Tagger labels that only repeat what the transcript already says.
const REDUNDANT: [&str; 2] = ["speech", "narration, monologue"];
/// Labels shown per moment.
const TOP: usize = 3;

/// Events whose spans overlap form one moment; its labels are listed best score first
/// (unscored events, like silence, first), each once.
fn event_groups(events: &[Event]) -> Vec<Group> {
    let mut sorted: Vec<&Event> = events
        .iter()
        .filter(|e| !REDUNDANT.contains(&e.label.as_str()))
        .collect();
    sorted.sort_by(|a, b| a.t0.total_cmp(&b.t0));
    let mut clusters: Vec<(f64, Vec<&Event>)> = Vec::new();
    for e in sorted {
        match clusters.last_mut() {
            Some((end, c)) if e.t0 < *end => {
                *end = end.max(e.t1);
                c.push(e);
            }
            _ => clusters.push((e.t1, vec![e])),
        }
    }
    clusters
        .into_iter()
        .map(|(t1, mut c)| {
            let t0 = c[0].t0;
            c.sort_by(|a, b| b.score.unwrap_or(1.0).total_cmp(&a.score.unwrap_or(1.0)));
            let mut tags: Vec<(&str, String)> = Vec::new();
            for e in c {
                if tags.iter().all(|(l, _)| *l != e.label) {
                    let tag = match e.score {
                        Some(x) => format!("{} {x:.2}", e.label),
                        None => format!(
                            "{} ({})",
                            e.label,
                            e.engine.tool.as_deref().unwrap_or("level")
                        ),
                    };
                    tags.push((e.label.as_str(), tag));
                }
            }
            let more = tags.len().saturating_sub(TOP);
            let mut labels: Vec<String> = tags.into_iter().take(TOP).map(|(_, t)| t).collect();
            if more > 0 {
                labels.push(format!("+{more}"));
            }
            Group {
                t0,
                t1,
                labels: labels.join(", "),
            }
        })
        .collect()
}

/// `75.5` → `1:15`, `3725` → `1:02:05`.
pub fn clock(t: f64) -> String {
    let s = t.max(0.0).round() as u64;
    if s >= 3600 {
        format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
    } else {
        format!("{}:{:02}", s / 60, s % 60)
    }
}

/// The index: what is in this recording and where, in a few hundred tokens.
pub fn index(heard: &Heard, title: &str, uri: &str) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "# {title}\n\n{uri} · {} long\n", clock(heard.duration));
    let words: usize = heard
        .segments
        .iter()
        .map(|x| x.text.split_whitespace().count())
        .sum();
    match &heard.speech_engine {
        Some(e) => {
            let _ = writeln!(
                s,
                "**Speech:** {} segments, {words} words ({} · {}). Full text: `transcript.md`.",
                heard.segments.len(),
                e.tool.as_deref().unwrap_or("?"),
                e.model.as_deref().unwrap_or("?")
            );
        }
        None => s.push_str("**Speech:** not transcribed (`--no-speech`).\n"),
    }
    let music_s: f64 = heard.music.iter().map(|m| m.t1 - m.t0).sum();
    let _ = writeln!(
        s,
        "**Music:** {} span(s), {} of {} ({} profile).\n",
        heard.music.len(),
        clock(music_s),
        clock(heard.duration),
        heard.profile.as_str()
    );
    if !heard.music.is_empty() {
        s.push_str("| from | to | tempo | pulse | key | fit (r) | margin |\n|---|---|---|---|---|---|---|\n");
        for m in &heard.music {
            let (bpm, pulse) = m.tempo.map_or(("–".into(), "–".into()), |t| {
                (format!("{:.1}", t.bpm), format!("{:.2}", t.pulse))
            });
            let (key, corr, margin) =
                m.key.map_or(("–".into(), "–".into(), "–".into()), |k| {
                    (
                        k.to_string(),
                        format!("{:.2}", k.corr),
                        format!("{:.2}", k.margin),
                    )
                });
            let _ = writeln!(
                s,
                "| {} | {} | {bpm} | {pulse} | {key} | {corr} | {margin} |",
                clock(m.t0),
                clock(m.t1)
            );
        }
        s.push('\n');
    }
    let groups = event_groups(&heard.events);
    let _ = writeln!(
        s,
        "**Sound events:** {} in {} moments (speech tags left out here: the transcript has them)\n",
        heard.events.len(),
        groups.len()
    );
    if !groups.is_empty() {
        s.push_str("| from | to | heard (best score) |\n|---|---|---|\n");
        for g in &groups {
            let _ = writeln!(s, "| {} | {} | {} |", clock(g.t0), clock(g.t1), g.labels);
        }
        s.push('\n');
    }
    s.push_str("Per-window measures behind the music spans: `windows.tsv`. Observations for memory: `messages.jsonl` (`mneme ingest`).\n");
    s
}

/// One line per segment: `[m:ss] text`.
pub fn transcript(heard: &Heard) -> String {
    heard
        .segments
        .iter()
        .map(|x| format!("[{}] {}\n", clock(x.t0), x.text))
        .collect()
}

/// Tab-separated window measures, for checking thresholds against real audio.
pub fn windows_tsv(heard: &Heard) -> String {
    let mut s = String::from("t0\tt1\tdb\tbpm\tpulse\tkey\tkey_corr\tmusic\n");
    for w in &heard.windows {
        let _ = writeln!(
            s,
            "{:.0}\t{:.0}\t{:.1}\t{}\t{}\t{}\t{}\t{}",
            w.t0,
            w.t1,
            w.db,
            w.tempo.map_or("".into(), |t| format!("{:.1}", t.bpm)),
            w.tempo.map_or("".into(), |t| format!("{:.3}", t.pulse)),
            w.key.map_or(String::new(), |k| k.to_string()),
            w.key.map_or("".into(), |k| format!("{:.3}", k.corr)),
            u8::from(w.music)
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlapping_events_group_with_top_labels() {
        let ev = |t0: f64, t1: f64, label: &str, score: Option<f64>| Event {
            t0,
            t1,
            label: label.into(),
            score,
            engine: organon_sound_core::Engine::default(),
        };
        let events = [
            ev(0.0, 2.0, "typing", Some(0.95)),
            ev(0.5, 3.0, "typewriter", Some(0.85)),
            ev(1.0, 2.0, "coin (dropping)", Some(0.7)),
            ev(1.5, 2.5, "scissors", Some(0.4)),
            ev(0.0, 60.0, "speech", Some(1.0)),
            ev(10.0, 13.0, "silence", None),
        ];
        let g = event_groups(&events);
        assert_eq!(
            g.len(),
            2,
            "speech is left out, so it does not swallow the rest"
        );
        assert_eq!(
            g[0].labels,
            "typing 0.95, typewriter 0.85, coin (dropping) 0.70, +1"
        );
        assert_eq!((g[0].t0, g[0].t1), (0.0, 3.0));
        assert_eq!(g[1].labels, "silence (level)");
    }

    #[test]
    fn clocks() {
        assert_eq!(clock(75.5), "1:16");
        assert_eq!(clock(3725.0), "1:02:05");
        assert_eq!(clock(-1.0), "0:00");
    }
}
