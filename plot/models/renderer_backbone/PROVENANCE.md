Copied from `3d2d-world-model/2daction/persist` on 2026-09-08.
Source repository HEAD: `1ab2005687af7ab7ce423b5876837485dd953e5a`.
Files were copied from the working tree; this hash identifies its base revision.

Local imports are namespaced. PLOT adapters and training live outside this package.
The original model parameter names are retained for explicit checkpoint transfer.
Rasterization dependencies are loaded lazily; camera-space depth samples can also
be supplied explicitly for CPU contract tests and offline projection.

PLOT modifications in `dit_pixel.py`: call-local projection reuse, input-sized
camera aspect ratio, dtype-preserving projection, lazy rasterizer import, and
removal of unused cross-raster rotary embedding initialization. Parameters of
the used DiT backbone keep their original names.
`embeddings.py` skips identity NTK rescaling, which otherwise divides by zero
for dimension-two rotary embeddings in small VAE smoke configurations.

`vae_voxel.py` was copied unchanged from the local `PERSIST/models/vae_voxel.py`
on 2026-09-10 for the static multi-view M1 experiment. Its upstream MIT license
is included as `PERSIST_LICENSE`. The static adaptation in
`plot/models/multiview_voxel_dit.py` retains PERSIST-S spatial/cross-attention parameter
names, omits temporal/action modules, and conditions on unordered image/ray tokens.
