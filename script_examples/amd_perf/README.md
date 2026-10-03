# AMD performance workflows

API-format prompts from the `dev/amd-perf-1` comparison, not frontend workflow
exports. Use the matching `rattus128/comfy-kitchen` branch with its native HIP
backend built. Checkpoint files are not included; their exact filenames are in
the loader nodes. Prompt-enhancement LLMs are disabled; text encoders remain.

| Files | Output | Sampling steps |
| --- | --- | --- |
| `zimage-{512,1024}.json` | Z-Image Turbo, square | 8 |
| `krea-{512,1024}.json` | Krea 2 Turbo, square | 8 |
| `qwen-{512,1024}.json` | Qwen Image 2.1, square | 25 |
| `h3-640x384-39frames.json` | 640x384, 39 frames, 1.625 s | 4 |
| `h3-640x384-124frames.json` | 640x384, 124 frames, 5.166667 s | 4 |
| `h3-1152x640-192frames.json` | 1152x640, 192 frames, 8 s | 4 |
| `h3-576x320-192frames-warmup.json` | 576x320 warmup for the 640p case | 4 |

H3 uses 24 fps. Its node rounds the requested lengths 24 and 120 up to the
model's frame grid, producing 39 and 124 frames. Length 192 is exact.

The comparison used an RX 9070 XT with these ComfyUI flags:

```sh
python main.py --enable-dynamic-vram --use-ck-attention --bf16-unet --preview-method none --reserve-vram 2
```

CUDA/HIP graphs were enabled. Submit a prompt to the local server, for example:

```sh
jq '{prompt: .}' script_examples/amd_perf/zimage-512.json |
  curl -H 'Content-Type: application/json' --data-binary @- http://127.0.0.1:8188/prompt
```

For the eight original cases, warm up once at the target resolution, then use
different seeds for at most two measured runs. These files retain the first
measured seed and output prefix; `master` in the prefix is just an output label.
Increment the sampler's `seed` or `noise_seed` between runs to avoid cached
execution, and use identical seeds across implementations.

For 640p H3, run the supplied 576x320 warmup first, then the 1152x640 prompt
exactly once in the same process. This includes any new-resolution graph setup.
Report sampler and whole-workflow timing separately; workflow time includes
decoding and saving. The benchmark's separate timing probe is not required by
these prompts.
