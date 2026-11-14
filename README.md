# organon-sound

**Give it a sound; get back what was said, where the music is (tempo, key), and what else happened when.** The hearing organ of [organon](https://github.com/Philipjj609/organon): it perceives and does not interpret. Comprehension is the reader's job, and the result goes to a memory such as [mneme](https://github.com/Philipjj609/mneme) as evidence-linked claims.

```mermaid
flowchart LR
  IN[audio or video file<br/>organon-video folder<br/>URL] -->|sense: yt-dlp / ffprobe| SRC[Source<br/>same id as organon-video]
  SRC -->|ffmpeg -f f32le<br/>16 kHz mono| PCM[samples]
  SRC -->|ffmpeg whisper filter<br/>whisper.cpp model| SEG[transcript segments]
  SEG -->|'[Music]', '(applause)'| EV
  PCM --> LV[level per 50 ms] --> SIL[silence ≥ 2 s] --> EV[sound events]
  PCM --> ON[onset envelope] --> TP[tempo + pulse]
  PCM --> CH[chroma] --> KY[key + fit r]
  TP & KY --> WIN[10 s windows<br/>music if pulse and tonal] --> MUS[music spans<br/>tempo · key]
  SEG & MUS & EV --> IDX[index.md<br/>read first]
  SEG & MUS & EV --> MSG[messages.jsonl<br/>organon-protocol]
  MSG -->|mneme ingest| MEM[(memory)]
  PCM -->|tract| TAG[YAMNet<br/>521 AudioSet classes] --> EV
```

## Why

A transcript is what most tools take from audio, and it drops the rest: that a talk opens with music, where the silences are, that a song sits at 142 BPM in G minor. This organ keeps speech, music and events side by side, each located in time, so a reader can cite "the moment it is said" or "the stretch of music" and drill into the audio only where needed.

## Use

Needs `ffmpeg` 8+ built with the `whisper` filter (Gyan and BtbN full builds have it; `organon-sound doctor` checks), plus `yt-dlp` for URLs. Without the filter, `--no-speech` still gives music and events.

```bash
cargo install --git https://github.com/Philipjj609/organon-sound organon-sound
organon-sound doctor
organon-sound model base                              # whisper.cpp ggml model, 148 MB, into ~/.organon/sound/models
organon-sound model yamnet                            # sound-event tagger, 16 MB (optional)
organon-sound hear https://youtu.be/1XRTvj__ZPY       # → ~/.organon/sound/yt-1XRTvj__ZPY/index.md
organon-sound hear ~/.organon/video/yt-1XRTvj__ZPY    # a video organon-video already saw: same source id
organon-sound hear song.mp3 --no-speech               # music and events only
mneme ingest ~/.organon/sound/<key>/messages.jsonl
```

Options: `--model small` (or a path), `--language en`, `--queue 20` (seconds per whisper pass), `--profile temperley` (key profiles for classical music), `--vad` (Silero VAD, `organon-sound model vad`), `--gpu`, `--no-tags`.

The work folder (`$ORGANON_HOME/sound/<key>`, `ORGANON_HOME` = `~/.organon`) holds `index.md`, `transcript.md`, `windows.tsv` (the measures behind every music decision), `whisper.json` and `messages.jsonl`. It may hold third-party audio, so it never belongs in a repo.

## What comes out

`messages.jsonl` holds one [organon-protocol](https://github.com/Philipjj609/organon-protocol) message per line: the `Source` (its id matches organon-video's for the same video or file), an anchor when this organ sensed the input, and observations located by time:

| kind | value | by |
|---|---|---|
| `transcript_segment` | `{text}` | ffmpeg's whisper + model |
| `music` | `{bpm, pulse, key, key_corr, key_margin, profile}` | measured here |
| `sound_event` | `{label, score?}`: `silence`, whisper's own `[Music]`-style annotations, YAMNet's AudioSet classes | level / whisper / tract + yamnet |

## Measured, on real recordings

| | |
|---|---|
| Speech vs a talk's manual captions | 1,642 of 1,710 words in order (96.0%), `ggml-base`, 12:51 heard in about 3 min on CPU |
| Music vs speech | no music found in 13:39 of speech; 93% of 26:19 of music found (8 recordings) |
| Tempo vs librosa | within 2% on 3 of 5; the other 2 differ by metre (3:2, 2:1) |
| Key vs librosa CQT chroma | same on 3 of 5; the other 2 a fifth or the relative key away |
| Sound events | YAMNet: 164 events in 45 moments on the talk in about 10 s; spot checks match the narration, not checked by ear |

Method, thresholds and every number: [docs/DESIGN.md](docs/DESIGN.md).

## Engines, swappable

Research (organon-research run `tools-should-rust-sound-perception-organ`, digested into memory) picked by evidence; each engine sits behind a trait so a better one can replace it:

| job | v0.1 | why | next |
|---|---|---|---|
| decode | ffmpeg → f32le | already required; reads video containers and Opus, which Symphonia does not by default | Symphonia for ffmpeg-free audio |
| speech | ffmpeg `whisper` filter (whisper.cpp) | no Python, no second binary, any container | whisper-rs for word timestamps |
| tempo | onset autocorrelation (Percival-style) | adequate classic; known octave errors above ~150 BPM | a beat tracker for beat times |
| key | chroma × Krumhansl / Temperley profiles | template choice matters by genre | EDMA profiles, tuning estimation |
| events | silence + whisper's annotations + YAMNet (ONNX) in tract | pure Rust inference; YAMNet takes raw 16 kHz | EfficientAT (mAP 0.483 vs 0.306) |

## License

MIT OR Apache-2.0.
