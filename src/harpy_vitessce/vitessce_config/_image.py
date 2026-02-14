from dataclasses import dataclass
from pathlib import Path

import zarr
from loguru import logger
from vitessce import (
    CoordinationLevel as CL,
)


@dataclass
class _ImageLayerConfigBuilder:
    img_source: str
    is_img_remote: bool
    base_dir: str | Path | None
    visualize_as_rgb: bool = True

    def _resolve_local_img_path(self) -> Path | None:
        if self.is_img_remote:
            return None
        path = Path(self.img_source)
        if path.is_absolute():
            return path
        if self.base_dir is not None:
            return Path(self.base_dir) / path
        return path

    def _infer_channel_count(self) -> int | None:
        img_path = self._resolve_local_img_path()
        if img_path is None:
            logger.warning(
                "Could not infer channel count for remote image source '{}'.",
                self.img_source,
            )
            return None
        if not img_path.exists():
            logger.warning(
                "Image source '{}' does not exist locally; could not infer channel count.",
                str(img_path),
            )
            return None

        try:
            z_root = zarr.open_group(str(img_path), mode="r")
        except Exception as e:
            logger.warning(
                "Could not open OME-Zarr at '{}' to infer channel count: {}",
                str(img_path),
                e,
            )
            return None

        # Preferred source: OMERO channels metadata.
        omero = z_root.attrs.get("omero")
        if isinstance(omero, dict):
            channels = omero.get("channels")
            if isinstance(channels, list) and len(channels) > 0:
                return len(channels)

        # Fallback: use multiscales axis metadata + first dataset shape.
        multiscales = z_root.attrs.get("multiscales")
        if isinstance(multiscales, list) and len(multiscales) > 0:
            first = multiscales[0]
            axes = first.get("axes")
            if isinstance(axes, list):
                axis_names: list[str | None] = []
                for axis in axes:
                    if isinstance(axis, dict):
                        axis_names.append(axis.get("name"))
                    elif isinstance(axis, str):
                        axis_names.append(axis)
                    else:
                        axis_names.append(None)
                if "c" in axis_names:
                    c_axis = axis_names.index("c")
                    datasets = first.get("datasets")
                    if isinstance(datasets, list) and len(datasets) > 0:
                        first_dataset = datasets[0]
                        if isinstance(first_dataset, dict):
                            dataset_path = first_dataset.get("path")
                            if isinstance(dataset_path, str):
                                try:
                                    arr = z_root[dataset_path]
                                    return int(arr.shape[c_axis])
                                except Exception as e:
                                    logger.warning(
                                        "Could not read dataset '{}' in '{}' to infer "
                                        "channel count: {}",
                                        dataset_path,
                                        str(img_path),
                                        e,
                                    )

        logger.warning(
            "Could not infer channel count from '{}'.",
            str(img_path),
        )
        return None

    @staticmethod
    def _channel_color(index: int, n_channels: int) -> list[int]:
        if n_channels == 1:
            return [255, 255, 255]
        palette = [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 255, 255],
        ]
        return palette[index % len(palette)]

    def _image_channels_coordination(
        self,
        n_channels: int,
        visible_channels: set[int] | None = None,
    ) -> CL:
        if visible_channels is None:
            visible_channels = set(range(n_channels))
        return CL(
            [
                {
                    "spatialTargetC": i,
                    "spatialChannelColor": self._channel_color(i, n_channels),
                    "spatialChannelVisible": i in visible_channels,
                    "spatialChannelOpacity": 1.0,
                }
                for i in range(n_channels)
            ]
        )

    def build(self, file_uid: str) -> tuple[dict[str, object], bool]:
        """
        Build the Vitessce `imageLayer` coordination entry for the image.

        Parameters
        ----------
        file_uid
            File identifier used to link this layer to the image file.

        Returns
        -------
        tuple[dict[str, object], bool]
            A pair of `(image_layer, can_render_as_rgb)` where:
            - `image_layer` is the layer config for Vitessce `imageLayer`.
            - `can_render_as_rgb` is `True` only when RGB rendering is selected
              and at least 3 channels are detected in the OME-Zarr.
        """
        n_channels = self._infer_channel_count()
        image_layer: dict[str, object] = {
            "fileUid": file_uid,
            "spatialLayerVisible": True,
            "spatialLayerOpacity": 1.0,
        }

        if self.visualize_as_rgb:
            if n_channels is not None and n_channels >= 3:
                image_layer["photometricInterpretation"] = "RGB"
                if n_channels > 3:
                    logger.warning(
                        "visualize_as_rgb=True and found {} channel(s); rendering the "
                        "first three channels (0, 1, 2) as RGB.",
                        n_channels,
                    )
                    # if channels > 3, we need a way to tell vitessce to only render the
                    # first three images, therefore we need to add "imageChannel" key in this case.
                    # layer we then set disableChannelsIfRgbDetected to True
                    image_layer["imageChannel"] = self._image_channels_coordination(
                        n_channels=n_channels,
                        visible_channels={0, 1, 2},
                    )
                return image_layer, True

            image_layer["photometricInterpretation"] = "BlackIsZero"
            if n_channels is not None and n_channels > 0:
                logger.warning(
                    "visualize_as_rgb=True but found fewer than 3 channels ({}); "
                    "falling back to multiplex rendering.",
                    n_channels,
                )
                image_layer["imageChannel"] = self._image_channels_coordination(
                    n_channels=n_channels
                )
            else:
                logger.warning(
                    "visualize_as_rgb=True but channel count is unknown; leaving "
                    "imageChannel unspecified."
                )
            return image_layer, False

        image_layer["photometricInterpretation"] = "BlackIsZero"
        if n_channels is not None and n_channels > 0:
            image_layer["imageChannel"] = self._image_channels_coordination(
                n_channels=n_channels
            )
        else:
            logger.warning(
                "Could not infer channel count; leaving imageChannel unspecified."
            )
        return image_layer, False
