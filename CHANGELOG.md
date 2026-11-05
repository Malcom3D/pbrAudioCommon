
# pbrAudioCommon
## [v0.3.0] - 2026-11-04

### Added
- Added bz2 compression and decompression support for classes saved by the `KleptoEntityStore` class in `pbrAudioCommon`.
- Introduced `pbrAudioDAW` featuring a new Terminal User Interface (TUI).
- Integrated the Viewport Panel and new 3D viewport widgets GUI into `pbrAudioBlender`.
- Added new mono modal displacement to acoustic pressure processors, modal displacement processors, and audio forces processors in `pbrAudioSignals`.
- Added new methods in the `StorageEngine` class to edit metadata on the backend.
- Added `SynthPostProcess` and rigidBody postProcess classes to describe audio data processing chains and rigidBodySynth post-processing in `pbrAudioStorage` metadata.
- Added Blosc2 support to handle large NumPy arrays in `ScoreEvent` within `pbrAudioShaders`.

### Changed
- Switched `blosc2Backend` to parallel append and mmapped mode in `pbrAudioStorage`.
- Handled `track_group` in `pbrAudioSignals` to subdivide processors based on DAG node/edge goals.
- Performed a complete code rewrite of `pbrAudioStorage` and aligned postProcess classes with new storage and signal features.

### Fixed
- Fixed typos, syntax errors, and missing definitions across `KleptoEntityStore`, `StorageEngine`, `Blosc2Backend`, `pbrAudioShaders`, `pbrAudioSignals`, and `pbrAudioBlender`.
- Fixed misalignment issues between signals and `ScoreEvent` originating from `physicsSolver` and `rigidBody`.
- Prevented Out Of Memory (OOM) errors in `ModalComposer` within `rigidBody`.
- Fixed storage `root_path` and `enable_proxy_synth` indentation in `pbrAudioBlender`.

-
