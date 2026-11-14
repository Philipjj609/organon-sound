//! Typed errors of the core.

/// What can go wrong while hearing.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A sample rate of zero, or audio too short to measure.
    #[error("bad audio: {0}")]
    Audio(String),
    /// A line of whisper's JSON output that is not `{"start","end","text"}`.
    #[error("whisper output line {line}: {reason}")]
    Whisper { line: usize, reason: String },
    /// An engine (transcriber, tagger) failed; the message is the engine's.
    #[error("{engine} failed: {message}")]
    Engine { engine: String, message: String },
    /// An anchor id built from the input is invalid.
    #[error(transparent)]
    Id(#[from] organon_protocol::IdError),
}

/// Result with the core's error.
pub type Result<T> = std::result::Result<T, Error>;
