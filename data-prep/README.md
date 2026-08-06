# data-prep

Offline preparation, run once and then forgotten. Nothing here is imported by
the training code — this is how the artifacts in `data/` were produced.

| file | what it does |
| --- | --- |
| `quickdraw_to_png.py` | downloads QuickDraw bitmaps and writes the flat, shuffled PNG pool that becomes `data/formations/` |
| `autoencoder_latent_search.ipynb` | trains the target-image autoencoder and sweeps its latent width; produced `data/image_encoder.pt` |
| `ae_outputs/` | reconstructions and the latent sweep from that notebook, kept as the record of why `Z` is what it is |

For diagnostics that run *against* the live pipeline, see `python/tools/`.
