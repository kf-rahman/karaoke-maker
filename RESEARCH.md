# Research: Replacing Spleeter with Quantized HTDemucs

## Why replace Spleeter?

Spleeter was a practical first choice — easy to install, simple CLI, gets the job done for Western pop/rock. But it has structural limitations that hurt us for our primary use case (non-English music):

- Trained on **musdb18**, a dataset of ~150 Western pop/rock songs with separately recorded stems
- Uses a **spectrogram-based U-Net** with purely local convolutional filters — no global context, each audio chunk is processed independently
- No attention mechanism, so it can't recognize that the same voice in verse 1 is the same voice in the chorus
- Separation quality degrades significantly on music where vocals and instruments share frequency ranges — common in South Asian, Middle Eastern, and East Asian music (e.g. harmonium alongside a singer, sarangi, oud)

## Why HTDemucs?

HTDemucs (Hybrid Transformer Demucs, v4) is the current state of the art for open-source source separation:

**Architecture differences vs Spleeter:**

| | Spleeter | HTDemucs |
|---|---|---|
| Domain | Spectrogram only | Waveform + spectrogram (hybrid) |
| Architecture | U-Net (CNN) | U-Net + Transformer bottleneck |
| Context | Local (conv window) | Global (attention across full track) |
| Training data | ~150 songs (musdb18) | ~800+ songs (musdb18 + internal Deezer) |
| Separation quality (SDR) | ~6 dB | ~9 dB |

The transformer bottleneck gives the model global context — it can recognize the same instrument or voice across the whole track and make consistent separation decisions rather than treating each chunk independently.

**Practical effect for our app:**
- Less "ghost vocal" bleed-through in the karaoke track
- More consistent separation quality throughout the song
- Better handling of instruments in vocal frequency ranges (a core problem for South Asian music)

## The quantization plan

HTDemucs weights in float32 are ~80MB. That fits in memory comfortably, but combined with Whisper small (~500MB) and inference overhead we need to be careful on Railway's 8GB.

**Approach: PyTorch dynamic int8 quantization**

```python
import torch
from demucs.pretrained import get_model

model = get_model("htdemucs")
model.eval()

# Dynamic quantization targets Linear and Conv layers
# No calibration data needed — weights quantized offline,
# activations quantized dynamically at inference time
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv1d, torch.nn.ConvTranspose1d},
    dtype=torch.qint8,
)

torch.save(quantized_model.state_dict(), "htdemucs_int8.pt")
```

**Expected memory footprint:**

| Component | float32 | int8 |
|---|---|---|
| HTDemucs weights | ~80MB | ~20MB |
| Inference peak (4min song) | ~2-3GB | ~1-1.5GB |
| Whisper small (always loaded) | ~500MB | — |
| OS + Python overhead | ~500MB | — |
| **Total estimate** | **~4GB** | **~2.5GB** |

Dropping from ~4GB to ~2.5GB peak gives spleeter's slot back comfortably on Railway's 8GB.

**Quality tradeoff:**
- Convolutional layers quantize cleanly — negligible perceptual difference
- Transformer attention layers are more sensitive to quantization
- In practice for source separation, int8 dynamic quantization loses <0.5 dB SDR — imperceptible to most listeners

## Implementation plan

### 1. Replace the subprocess call with a Python API call

Current (Spleeter via subprocess):
```python
sep = subprocess.run(["spleeter", "separate", "-p", "spleeter:2stems", ...])
```

Target (HTDemucs via Python API):
```python
from demucs.apply import apply_model
from demucs.audio import AudioFile, save_audio

wav = AudioFile(audio_path).read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)
wav = wav.unsqueeze(0)  # add batch dim
sources = apply_model(model, wav)
# sources shape: [batch, stems, channels, time]
# stems: [drums, bass, other, vocals] for htdemucs
accompaniment = sources[0, :-1].sum(dim=0)  # everything except vocals
```

### 2. Pre-load model at startup (same pattern as Whisper)

```python
_demucs_model = None

def _load_demucs_model():
    global _demucs_model
    from demucs.pretrained import get_model
    import torch
    model = get_model("htdemucs")
    model.eval()
    _demucs_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.Conv1d, torch.nn.ConvTranspose1d}, dtype=torch.qint8
    )
    logging.info("HTDemucs model ready.")
```

### 3. Remove Spleeter entirely

- Remove `spleeter` from `requirements.txt`
- Remove TensorFlow dependency (Spleeter pulls in TF, which itself is ~500MB — removing it alone helps memory)
- Replace the `run_spleeter()` coroutine with a `run_demucs()` coroutine

### 4. Docker image changes

- Drop: `pip install spleeter` (removes TensorFlow too)
- Add: `pip install demucs`
- Bake in quantized weights at build time

## Open questions

1. **Does PyTorch dynamic quantization work well with HTDemucs's ConvTranspose layers?** Need to benchmark SDR before/after on a few test tracks.

2. **Inference time on CPU?** HTDemucs is slower than Spleeter on CPU (no GPU on Railway). Benchmarking needed — worst case we set a tighter max duration or run with `torch.inference_mode()` and see if chunked processing helps.

3. **4-stem vs 2-stem?** HTDemucs by default separates into 4 stems (drums, bass, other, vocals). We'd sum drums+bass+other to get accompaniment. This is actually better than Spleeter's 2-stem approach since the model has learned each instrument independently.

4. **`htdemucs_ft` variant?** The fine-tuned variant has slightly better SDR on vocals specifically — worth testing whether it quantizes as cleanly.

## References

- [HTDemucs paper](https://arxiv.org/abs/2211.00847)
- [Demucs GitHub](https://github.com/facebookresearch/demucs)
- [PyTorch quantization docs](https://pytorch.org/docs/stable/quantization.html)
- [musdb18 dataset](https://sigsep.github.io/datasets/musdb.html)
