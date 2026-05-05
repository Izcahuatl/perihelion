# Perihelion
**TTS app made for VRChat!!**

Perihelion runs a tiny (not) Qwen3 ASR model directly on your machine and connects it to VRChat.

---
## Features?
- Offline...Yeah. It's an offline model.
- It connects to VRChat via OSC.
- It can be set to listen via a toggle, an avatar parameter, or always on.
- You can set up Hotwords in case the model keeps ignoring your phrases.
- The model can run on either CPU or GPU via DirectML or CUDA.
  - Prooooobably keep it on CPU unless you've got the VRAM to spare.
---
## Getting it
Grab the latest build from [Releases](../../releases). Duh.
> **Windows only for now.** I don't have a Linux machine to test, sorry!!
---
## First launch
The app will download the model (~1 GB) on first run. This takes a minute depending on your internet. That's it.
After that, subsequent launches will load the model immediately.

---
## How to use this thing
Honestly a lot of it is self-explanatory, but here's some more important stuff so you're not confused.
### AI Preferences
- **High Accuracy Mode** uses beam search instead of greedy search. Better  but eats more CPU
  - **Search Depth**
    - 1 to 10. Higher = smarter = slower
- **Hardware Processor**
  - `CPU`
    - Works everywhere. Use this!
  - `GPU`
    - DirectML, in case you want speed and can afford it
  - `GPU-CUDA`
    - If you have an NVIDIA card and CUDA set up
- **CPU Threads**
  - 1 to 16. More threads = faster, but please don't increase it from the default `4` unless you know what you're doing
### Custom Dictionary
One word or phrase per line. The model will try harder to recognize these. Hotword Focus controls how hard it tries.
### Danger Zone
- **Annihilate Model** deletes the model files. Use this if the download got corrupted or the model is acting weird. You can redownload it immediately
- **Reset Config** nukes all your settings back to factory defaults. Useful if you broke something
---
## Will it run on my grandma's life support?
If your PC can handle 5 eboy avatars it can handle this.

The model download is ~1 GB and it uses about 1.1 GB of RAM while running

---
## Building it yourself
If you're into that:
- [Rust](https://rustup.rs/) (latest stable)
- A C++ compiler (MSVC on Windows)
```bash
git clone https://github.com/yourusername/perihelion.git
cd perihelion
cargo build --release
```
You'll find the binary at `target/release/perihelion.exe`

---
## Waaaahh it broke
**"Model not ready" / won't start**
- Make sure your PC isn't a coffee maker
- Set **Hardware Processor** to `CPU`
- Use **Annihilate Model** and let it re-download

**It's hearing things wrong**
- Turn on **High Accuracy Mode**
- Add the problematic words to the **Custom Dictionary**
- Check your mic. If it sounds like you're underwater, the AI will think you're underwater

**OSC mode does nothing**
- Is OSC enabled in VRChat? (`Options > OSC > Enabled`)
- Does your avatar actually have a parameter named `perihelion`?
- Can it even access ports 9000/9001?

**App crashes on launch**
- Nuke your config folder and start fresh:
  - `%APPDATA%\perihelion\` on Windows
---
## Serious License Stuff
Copyright (C) 2026 Izcahuatl

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. It can also be found at <https://www.gnu.org/licenses/>.