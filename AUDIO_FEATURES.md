# Audio Features and ALSA Requirements

The `gemini-ox` crate provides optional audio input and output capabilities for use with the Gemini Live API. These features are disabled by default to ensure the crate can be built and used on systems without audio dependencies.

## Features

### `audio` - Audio Input
Enables microphone input functionality using the `cpal` crate.

**Requirements:**
- Linux: ALSA development libraries (`libasound2-dev` on Ubuntu/Debian, `alsa-lib-devel` on RHEL/CentOS)
- macOS: Core Audio (included with system)
- Windows: WASAPI (included with system)

### `audio-output` - Audio Output
Enables audio playback functionality for receiving speech responses from the Gemini API.

**Requirements:**
- Same as `audio` feature (ALSA on Linux, Core Audio on macOS, WASAPI on Windows)

### `video` - Video Input
Enables camera input functionality using the `nokhwa` crate.

**Requirements:**
- Platform-specific camera access (handled by `nokhwa`)

## Usage Examples

### Text-only (no ALSA required)
```bash
cargo run --example live_multimodal_chat
```

### With audio input
```bash
cargo run --example live_multimodal_chat --features audio
```

### With audio output
```bash
cargo run --example live_multimodal_chat --features audio-output
```

### Full multimedia support
```bash
cargo run --example live_multimodal_chat --features audio,audio-output,video
```

## Installing ALSA Dependencies

### Ubuntu/Debian
```bash
sudo apt-get install libasound2-dev pkg-config
```

### RHEL/CentOS/Fedora
```bash
sudo dnf install alsa-lib-devel pkgconf-pkg-config
# or on older systems:
sudo yum install alsa-lib-devel pkgconfig
```

### Using Nix
The included `flake.nix` provides a development environment with ALSA libraries:
```bash
nix develop
```

## Troubleshooting

### Build Error: "The system library `alsa` required by crate `alsa-sys` was not found"
This error occurs when trying to build with audio features on a system without ALSA development libraries.

**Solutions:**
1. Install ALSA development packages (see above)
2. Build without audio features: `cargo build --no-default-features`
3. Use only specific features: `cargo build --features video` (excludes audio)

### CI/Headless Builds
For continuous integration or headless environments, build without audio features:
```bash
cargo test --workspace --lib  # Tests without audio/video
cargo build --workspace       # Build without audio/video
```

## Architecture

The audio functionality is designed with conditional compilation to ensure:
- Zero overhead when audio features are disabled
- Clean builds on systems without audio libraries
- Separation between input and output functionality
- Platform-specific optimizations through `cpal`

Audio input and output can be enabled independently, allowing for use cases like:
- Audio input only (speech-to-text)
- Audio output only (text-to-speech playback)
- Full duplex audio communication