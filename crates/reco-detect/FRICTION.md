# Detection consumer API friction

## Source colour range is absent from detector inputs

`reco_core::detect::detector` carries image layout, dimensions and residency, but `YuvView` and wgpu inputs do not carry the source's full/limited colour range. Rendering receives that information separately; Metal can derive it from the retained pixel buffer's native format. Other detector paths cannot consistently interpret limited-range YUV without assuming a range or maintaining a second configuration owner. The input contract needs to retain the source colour interpretation through the existing zero-copy paths. Matrix and resize differences between preprocessing implementations are separate backend correctness concerns, not a reason to duplicate source metadata in consumers.
