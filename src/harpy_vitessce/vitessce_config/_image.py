from collections.abc import Sequence
from dataclasses import dataclass

from loguru import logger
from vitessce import (
    CoordinationLevel as CL,
)


@dataclass
class _ImageLayerConfigBuilder:
    img_source: str
    visualize_as_rgb: bool = True
    layer_opacity: float = 1.0

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

    def build(
        self,
        channels: Sequence[int | str] | None,
        file_uid: str,
    ) -> tuple[dict[str, object], bool]:
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
            - `render_as_rgb` is `True` only when RGB rendering is selected
              and at least 3 channels are detected in the OME-Zarr.
        """

        def _channel_color(index: int) -> list[int]:
            palette = [
                [255, 0, 0],  # "#FF0000"  Red
                [0, 255, 0],  # "#00FF00"  Green
                [0, 0, 255],  # "#0000FF"  Blue
                [255, 255, 0],  # "#FFFF00"  Yellow
                [128, 0, 128],  # "#800080"  Purple
                [255, 165, 0],  # "#FFA500"  Orange
                [255, 192, 203],  # "#FFC0CB"  Pink
                [139, 69, 19],  # "#8B4513"  Brown
            ]
            return palette[index % len(palette)]

        image_layer: dict[str, object] = {
            "fileUid": file_uid,
            "spatialLayerVisible": True,
            "spatialLayerOpacity": self.layer_opacity,
        }
        # if channels is None -> if visualize_as_rgb -> try rendering as rgb
        if channels is None:
            if self.visualize_as_rgb:
                selected_channels = [
                    0,
                    1,
                    2,
                ]  # log an info, that selected channels not set, and that we will interpret first three channels as rgb
            else:
                logger.info(
                    "No channels were provided, and visualize as rgb set to False; rendering only channel at index 0. "
                    "Additional channels can be enabled in the Vitessce UI."
                )
                selected_channels = [0]
        else:
            selected_channels = channels

        image_layer["imageChannel"] = CL(
            [
                {
                    "spatialTargetC": channel,
                    "spatialChannelColor": _channel_color(pos),
                    # TODO: yes -> override colors?
                    "spatialChannelVisible": True,
                    "spatialChannelWindow": None,
                }
                for pos, channel in enumerate(selected_channels)
            ]
        )

        if self.visualize_as_rgb:
            image_layer["photometricInterpretation"] = "RGB"
            return image_layer, True
        else:
            image_layer["photometricInterpretation"] = "BlackIsZero"
            return image_layer, False
