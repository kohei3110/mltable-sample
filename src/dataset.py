from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ImageSample:
    """A normalized image sample from the manifest.

    The production column is ``image_path``. The repository also includes
    ``local_image_path`` so the sample can be smoke-tested without Azure
    Storage access.
    """

    image_path: Any
    label: str
    split: str
    season: str = ""
    source: str = ""
    local_image_path: str = ""


class ManifestImageDataset:
    """Small framework-agnostic dataset backed by a manifest DataFrame/list.

    This class deliberately avoids a hard dependency on PyTorch. If you use
    PyTorch, wrap this object or copy the ``__len__`` / ``__getitem__`` pattern
    into a ``torch.utils.data.Dataset``.
    """

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        root_dir: str | Path | None = None,
        split: str | None = None,
        prefer_local_path: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir).resolve() if root_dir else None
        self.prefer_local_path = prefer_local_path
        self.samples = [self._to_sample(row) for row in rows if split is None or row.get("split") == split]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, str]:
        sample = self.samples[index]
        return self.resolve_image_source(sample), sample.label

    def resolve_image_source(self, sample: ImageSample) -> Any:
        """Return a local path, MLTable stream object, or URI for the image.

        In Azure, MLTable can return stream objects that expose ``open()``.
        Locally, this sample can resolve ``local_image_path`` relative to the
        curated release folder. If neither is available, the Azure URI is
        returned and the caller should open it with the appropriate runtime.
        """

        if self.prefer_local_path and sample.local_image_path:
            local_path = Path(sample.local_image_path)
            if not local_path.is_absolute() and self.root_dir:
                local_path = self.root_dir / local_path
            return local_path

        return sample.image_path

    def open_image(self, index: int) -> Any:
        """Open an image with Pillow when available.

        This method supports local files and MLTable stream objects. Azure URI
        strings are intentionally not downloaded here; prefer Azure ML data
        runtime mount/download or MLTable streams instead of custom download
        code in the dataloader.
        """

        source = self[index][0]

        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install Pillow to open sample images locally.") from exc

        if hasattr(source, "open"):
            with source.open() as stream:
                return Image.open(stream).copy()

        source_path = Path(source)
        if source_path.exists():
            return Image.open(source_path)

        raise FileNotFoundError(
            f"Image source is not a local file or MLTable stream: {source}. "
            "Use Azure ML data runtime instead of downloading in user code."
        )

    @staticmethod
    def _to_sample(row: Mapping[str, Any]) -> ImageSample:
        return ImageSample(
            image_path=row.get("image_path", ""),
            local_image_path=str(row.get("local_image_path", "") or ""),
            label=str(row.get("label", "") or ""),
            split=str(row.get("split", "") or ""),
            season=str(row.get("season", "") or ""),
            source=str(row.get("source", "") or ""),
        )