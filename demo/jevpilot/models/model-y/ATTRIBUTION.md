# Tesla Model Y 2021

- Original artist: [763468712](https://sketchfab.com/763468712).
- Original model: [Tesla Model Y 2021](https://sketchfab.com/3d-models/tesla-model-y-2021-c0a86cac582d4b33aba0fb1b1912d970).
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Original credit notice retained in LICENSE.txt.
- Downloaded from [Tina 3D Tesla](https://github.com/Tina2088/tina-3d-tesla/tree/main/public/models), which adapted normalization, mesh-island separation, polygon reduction, and studio materials.
- Jevpilot changes: additional polygon reduction, Draco compression, -Z forward axis, 4.75m length, pearl-white paint, glass and interior materials, per-material batching, and rolling/steering wheel pivots. Calipers steer without spinning.

Rebuild the optimized asset from the credited source with glTF Transform CLI:

```sh
gltf-transform simplify source.glb simplified.glb --ratio 0.3 --error 0.0005
gltf-transform draco simplified.glb model-y.glb
```

Tesla names and emblems identify the depicted vehicle. This project is not affiliated with Tesla.
