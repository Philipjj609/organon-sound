# Working in organon-sound (for AI agents)

**How we write code:** [organon/docs/ENGINEERING.md](https://github.com/Philipjj609/organon/blob/main/docs/ENGINEERING.md). Read it before changing anything. Design, measurements and thresholds: [docs/DESIGN.md](docs/DESIGN.md).

**Layout:**
- `crates/sound-core` (`organon-sound-core`): pure logic, with no process spawning, no network and no printing.
  - `pcm`, `signal`: samples, streamed spectra.
  - `level`, `tempo`, `key`, `music`: measures and music spans.
  - `speech`: whisper JSON → segments and annotations.
  - `hear`: the flow, behind the `Transcribe` and `Tag` traits (fakes in `hear::fakes`).
  - `emit`: protocol messages.
- `crates/sound-cli` (`organon-sound`): the binary.
  - `tools`: the only code that runs ffmpeg, ffprobe, yt-dlp and curl.
  - `engines`: ffmpeg's `whisper` filter behind `Transcribe`.
  - `render`: `index.md`, `transcript.md`, `windows.tsv`.

**Quality gate** (CI on Linux, macOS, Windows): `cargo fmt --all --check && cargo clippy --all-targets -- -D warnings && cargo test`. Every bug fix gets a regression test.

**Rules:**
1. Perception does not interpret. Measures and located observations only ("pulse 0.6 at 121 BPM, A minor fits with r = 0.81"); meaning enters memory through `mneme digest`.
2. Observation ids hash their numbers: times go through `pcm::ms`, measures through `pcm::r3`. Engines go in `produced_by`, never in the value.
3. Never commit audio, transcripts or models. Work folders live in `$ORGANON_HOME/sound`; test signals are synthesized in code (`signal::synth`).
4. A threshold change needs evidence: run on real recordings, check `windows.tsv`, and record the result in DESIGN.md.
5. Source ids match organon-video's (`youtube:<id>`, `file:<sha256 prefix>`), so the speech and the pictures of one video are one source.

**Try it:** `organon-sound model base`, then `organon-sound hear https://youtu.be/1XRTvj__ZPY` and read the `index.md` it prints.
