//! Sound perception for organon: what was said, where there is music (tempo, key), and which
//! sound events happen when, as located organon-protocol observations.
//!
//! - [`pcm`]: mono samples and time rounding.
//! - [`signal`]: short-time spectra, streamed frame by frame (no whole spectrogram in memory).
//! - [`level`]: loudness per frame and silent spans.
//! - [`tempo`]: onset envelope and tempo by autocorrelation (Percival-style).
//! - [`key`]: chroma and key by template correlation (Krumhansl or Temperley profiles).
//! - [`music`]: which windows are music, merged into spans with their tempo and key.
//! - [`speech`]: whisper JSON → transcript segments; bracketed annotations become events.
//! - [`hear`]: the flow, behind two traits the binary implements ([`hear::Transcribe`],
//!   [`hear::Tag`]), so everything here is testable on synthetic audio.
//! - [`tagging`]: a tagger's frame scores → labelled events.
//! - [`emit`]: what was heard as protocol messages.
//!
//! Perception does not interpret: a music span says "pulse 0.62 at 121 BPM, A minor fits
//! best (r = 0.81)", never "this is a happy song".

#![forbid(unsafe_code)]

pub mod emit;
pub mod error;
pub mod hear;
pub mod key;
pub mod level;
pub mod music;
pub mod pcm;
pub mod signal;
pub mod speech;
pub mod tagging;
pub mod tempo;

pub use error::Error;
pub use hear::{hear, Engine, Event, Heard, Options, Tag, Transcribe};
pub use pcm::Pcm;
pub use speech::Segment;
