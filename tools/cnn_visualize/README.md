# NOTE

This is an excerpt copy from [another repo](https://github.com/utkuozbulak/pytorch-cnn-visualizations/). It contains the following changes:

- Limit visualization methods to gradcam;
- Fix feature extractor to work with `OrderedDicts` (was using `.items()` on a `nn.Module` before, now uses `enumerate()`)
- `gradcam.py`: Change L90:L101 to use scipy instead of custom resizing implementation
- Fix imports to respect new structure
