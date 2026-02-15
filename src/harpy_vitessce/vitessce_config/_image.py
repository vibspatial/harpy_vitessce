from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
    layer_opacity: float = 1.0
    channel_opacity: float = 1.0
    default_visible_channels: set[int] | None = None

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

    def _infer_channel_windows(
        self, n_channels: int
    ) -> list[list[float] | None] | None:
        # TODO: probably remove this block
        """
        Infer per-channel intensity windows for initial rendering.

        Returns
        -------
        list[list[float] | None] | None
            Per-channel [start, end] windows (or None entries when unknown).
            Returns None when image is remote or cannot be inspected locally.
        """
        img_path = self._resolve_local_img_path()
        if img_path is None or not img_path.exists():
            return None

        try:
            z_root = zarr.open_group(str(img_path), mode="r")
        except Exception as e:
            logger.warning(
                "Could not open OME-Zarr at '{}' to infer channel windows: {}",
                str(img_path),
                e,
            )
            return None

        # Get level-0 dataset + channel axis for any data-driven fallback.
        dataset_path: str | None = None
        c_axis: int | None = None
        level0_arr = None
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
                    maybe_path = first_dataset.get("path")
                    if isinstance(maybe_path, str):
                        dataset_path = maybe_path
                        try:
                            level0_arr = z_root[dataset_path]
                        except Exception as e:
                            logger.warning(
                                "Could not open level-0 dataset '{}' in '{}': {}",
                                dataset_path,
                                str(img_path),
                                e,
                            )

        # Build a robust quantile-based fallback from image data.
        def _quantile_windows() -> list[list[float] | None] | None:
            if level0_arr is None or c_axis is None:
                return None
            try:
                # Subsample spatial axes for speed.
                axis_names = []
                for axis in multiscales[0].get("axes", []):  # type: ignore[index]
                    if isinstance(axis, dict):
                        axis_names.append(axis.get("name"))
                    elif isinstance(axis, str):
                        axis_names.append(axis)
                    else:
                        axis_names.append(None)

                slicer = []
                for i, name in enumerate(axis_names):
                    if name in {"t", "z"}:
                        slicer.append(0)
                    elif name == "c":
                        slicer.append(slice(None))
                    elif name in {"y", "x"}:
                        step = max(1, level0_arr.shape[i] // 512)
                        slicer.append(slice(None, None, step))
                    else:
                        slicer.append(slice(None))

                sample = np.asarray(level0_arr[tuple(slicer)])
                sample = np.moveaxis(sample, c_axis, 0)
                windows: list[list[float] | None] = []
                for i in range(n_channels):
                    if i >= sample.shape[0]:
                        windows.append(None)
                        continue
                    vals = np.asarray(sample[i], dtype=np.float64)
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        windows.append(None)
                        continue
                    lo, hi = np.quantile(vals, [0.01, 0.999])
                    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                        vmin = float(np.min(vals))
                        vmax = float(np.max(vals))
                        if (
                            not np.isfinite(vmin)
                            or not np.isfinite(vmax)
                            or vmax <= vmin
                        ):
                            windows.append(None)
                        else:
                            windows.append([vmin, vmax])
                    else:
                        windows.append([float(lo), float(hi)])
                return windows
            except Exception as e:
                logger.warning(
                    "Could not infer quantile channel windows from '{}' in '{}': {}",
                    dataset_path,
                    str(img_path),
                    e,
                )
                return None

        # Preferred source: per-channel OMERO windows, unless they are just
        # the full dtype range defaults (which tend to render too dark).
        omero = z_root.attrs.get("omero")
        if isinstance(omero, dict):
            channels = omero.get("channels")
            if isinstance(channels, list) and len(channels) >= n_channels:
                windows: list[list[float] | None] = []
                parsed_any = False
                all_look_default = True
                dt_lo = dt_hi = None
                if level0_arr is not None:
                    dt = np.dtype(level0_arr.dtype)
                    if dt.kind in ("u", "i"):
                        info = np.iinfo(dt)
                        dt_lo = float(info.min if dt.kind == "i" else 0)
                        dt_hi = float(info.max)
                    else:
                        info = np.finfo(dt)
                        dt_lo = float(info.min)
                        dt_hi = float(info.max)
                for i in range(n_channels):
                    ch = channels[i] if i < len(channels) else None
                    start = end = None
                    if isinstance(ch, dict):
                        win = ch.get("window")
                        if isinstance(win, dict):
                            s = win.get("start")
                            e = win.get("end")
                            if isinstance(s, (int, float)) and isinstance(
                                e, (int, float)
                            ):
                                start = float(s)
                                end = float(e)
                    if start is None or end is None:
                        windows.append(None)
                        all_look_default = False
                    else:
                        windows.append([start, end])
                        parsed_any = True
                        if dt_lo is None or dt_hi is None:
                            all_look_default = False
                        else:
                            if not (
                                abs(start - dt_lo) < 1e-9 and abs(end - dt_hi) < 1e-9
                            ):
                                all_look_default = False
                if parsed_any:
                    if all_look_default:
                        qwins = _quantile_windows()
                        if qwins is not None:
                            return qwins
                    return windows

        # Fallback: quantile windows from data.
        qwins = _quantile_windows()
        if qwins is not None:
            return qwins

        # Final fallback: full dtype dynamic range from level 0.
        if level0_arr is not None:
            dt = np.dtype(level0_arr.dtype)
            if dt.kind in ("u", "i"):
                info = np.iinfo(dt)
                lo = float(info.min if dt.kind == "i" else 0)
                hi = float(info.max)
            else:
                info = np.finfo(dt)
                lo = float(info.min)
                hi = float(info.max)
            return [[lo, hi] for _ in range(n_channels)]

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
            [255, 165, 0],
            [165, 255, 0],
        ]
        return palette[index % len(palette)]

    def _image_channels_coordination(
        self,
        n_channels: int,
        visible_channels: set[int] | None = None,
    ) -> CL:
        if visible_channels is None:
            visible_channels = set(range(n_channels))
        # channel_windows = self._infer_channel_windows(n_channels)
        return CL(
            [
                {
                    "spatialTargetC": i,
                    "spatialChannelColor": self._channel_color(
                        i, n_channels
                    ),  # TODO: probably not override colors, or let users specify them
                    "spatialChannelVisible": i in visible_channels,
                    "spatialChannelOpacity": self.channel_opacity,
                    # Keep window scoped per channel so sliders are decoupled.
                    # Use explicit values when available to avoid null-window
                    # rendering edge cases in some viewer states.
                    "spatialChannelWindow": None,
                    # "spatialChannelWindow": (
                    #    channel_windows[i]
                    #    if channel_windows is not None and i < len(channel_windows)
                    #    else None
                    # ),
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
            "spatialLayerOpacity": self.layer_opacity,
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
                n_channels=n_channels,
                visible_channels=self.default_visible_channels,
            )
        else:
            logger.warning(
                "Could not infer channel count; leaving imageChannel unspecified."
            )
        return image_layer, False
