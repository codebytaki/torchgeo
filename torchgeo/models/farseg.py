# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Foreground-Aware Relation Network (FarSeg) implementations."""

import logging
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules import (
    BatchNorm2d,
    Conv2d,
    Identity,
    Module,
    ModuleList,
    ReLU,
    Sequential,
    Sigmoid,
    UpsamplingBilinear2d,
)
from torchvision.models import resnet
from torchvision.models._api import WeightsEnum
from torchvision.ops import FeaturePyramidNetwork as FPN

logger = logging.getLogger(__name__)


class WeightLoader:
    """Advanced weight loader with multiple strategies for different weight sources.
    
    This class implements sophisticated weight loading techniques to handle:
    - TorchGeo custom weights (URL-based)
    - TorchVision weights (WeightsEnum)
    - Local checkpoint files (path-based)
    - Raw state dictionaries
    - HuggingFace model hub weights
    """
    
    # Backbone layer prefixes for filtering
    BACKBONE_PREFIXES = ('conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4')
    
    @staticmethod
    def detect_weight_type(weights):
        """Detect the type/source of weights provided.
        
        Args:
            weights: Weight object to analyze
            
        Returns:
            str: Weight type identifier ('torchgeo', 'torchvision', 'local_file', 
                 'state_dict', 'huggingface', 'unknown')
        """
        # Check for TorchGeo weights (has url attribute and torchgeo module)
        if hasattr(weights, 'url') and 'torchgeo' in type(weights).__module__:
            return 'torchgeo'
        
        # Check for TorchVision weights (WeightsEnum subclass)
        if isinstance(weights, WeightsEnum):
            return 'torchvision'
        
        # Check for local file path
        if isinstance(weights, (str, Path)):
            path = Path(weights)
            if path.exists() and path.suffix in ['.pth', '.pt', '.ckpt', '.bin']:
                return 'local_file'
        
        # Check for HuggingFace model ID (contains /)
        if isinstance(weights, str) and '/' in weights and not weights.endswith('.pth'):
            return 'huggingface'
        
        # Check for state dictionary
        if isinstance(weights, dict):
            return 'state_dict'
        
        return 'unknown'
    
    @staticmethod
    def load_torchgeo_weights(weights, backbone):
        """Load TorchGeo custom weights from URL.
        
        Strategy: Download from HuggingFace, filter backbone keys, handle formats.
        
        Args:
            weights: TorchGeo WeightsEnum object
            backbone: Backbone module to load weights into
            
        Returns:
            bool: True if loading successful
        """
        try:
            from torch.hub import load_state_dict_from_url
            
            logger.info(f"Loading TorchGeo weights from: {weights.url}")
            
            # Download state dict
            state_dict = load_state_dict_from_url(weights.url, progress=False, map_location='cpu')
            
            # Extract and process state dict
            backbone_state_dict = WeightLoader._extract_backbone_weights(state_dict)
            
            # Load with strict=False to allow fc layer mismatch
            missing, unexpected = backbone.load_state_dict(backbone_state_dict, strict=False)
            
            if missing:
                logger.warning(f"Missing keys: {missing}")
            if unexpected:
                logger.debug(f"Unexpected keys (ignored): {unexpected}")
            
            logger.info("Successfully loaded TorchGeo weights")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load TorchGeo weights: {e}")
            return False
    
    @staticmethod
    def load_torchvision_weights(weights, backbone_name):
        """Load TorchVision weights using native mechanism.
        
        Strategy: Use torchvision's built-in weight loading.
        
        Args:
            weights: TorchVision WeightsEnum object
            backbone_name: Name of backbone (e.g., 'resnet50')
            
        Returns:
            Module: Initialized backbone with weights
        """
        logger.info(f"Loading TorchVision weights for {backbone_name}")
        
        try:
            backbone = getattr(resnet, backbone_name)(weights=weights)
            logger.info("Successfully loaded TorchVision weights")
            return backbone
        except Exception as e:
            logger.error(f"Failed to load TorchVision weights: {e}")
            raise
    
    @staticmethod
    def load_local_checkpoint(filepath, backbone):
        """Load weights from local checkpoint file.
        
        Strategy: Load .pth/.pt/.ckpt file, extract backbone weights, handle formats.
        
        Args:
            filepath: Path to checkpoint file
            backbone: Backbone module to load weights into
            
        Returns:
            bool: True if loading successful
        """
        try:
            filepath = Path(filepath)
            logger.info(f"Loading weights from local file: {filepath}")
            
            # Load checkpoint
            checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
            
            # Handle different checkpoint formats
            state_dict = WeightLoader._extract_state_dict(checkpoint)
            
            # Extract backbone weights
            backbone_state_dict = WeightLoader._extract_backbone_weights(state_dict)
            
            # Load weights
            missing, unexpected = backbone.load_state_dict(backbone_state_dict, strict=False)
            
            if missing:
                logger.warning(f"Missing keys: {missing}")
            if unexpected:
                logger.debug(f"Unexpected keys (ignored): {unexpected}")
            
            logger.info("Successfully loaded local checkpoint")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load local checkpoint: {e}")
            return False
    
    @staticmethod
    def load_huggingface_weights(model_id, backbone):
        """Load weights from HuggingFace model hub.
        
        Strategy: Use transformers library to download and load weights.
        
        Args:
            model_id: HuggingFace model ID (e.g., 'username/model-name')
            backbone: Backbone module to load weights into
            
        Returns:
            bool: True if loading successful
        """
        try:
            from huggingface_hub import hf_hub_download
            
            logger.info(f"Loading weights from HuggingFace: {model_id}")
            
            # Download model file
            model_path = hf_hub_download(repo_id=model_id, filename='pytorch_model.bin')
            
            # Load using local checkpoint method
            return WeightLoader.load_local_checkpoint(model_path, backbone)
            
        except ImportError:
            logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
            return False
        except Exception as e:
            logger.error(f"Failed to load HuggingFace weights: {e}")
            return False
    
    @staticmethod
    def load_state_dict_direct(state_dict, backbone):
        """Load weights directly from state dictionary.
        
        Strategy: Extract backbone keys and load directly.
        
        Args:
            state_dict: Dictionary containing model weights
            backbone: Backbone module to load weights into
            
        Returns:
            bool: True if loading successful
        """
        try:
            logger.info("Loading weights from state dictionary")
            
            # Extract backbone weights
            backbone_state_dict = WeightLoader._extract_backbone_weights(state_dict)
            
            # Load weights
            missing, unexpected = backbone.load_state_dict(backbone_state_dict, strict=False)
            
            if missing:
                logger.warning(f"Missing keys: {missing}")
            if unexpected:
                logger.debug(f"Unexpected keys (ignored): {unexpected}")
            
            logger.info("Successfully loaded state dictionary")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load state dictionary: {e}")
            return False
    
    @staticmethod
    def _extract_state_dict(checkpoint):
        """Extract state dict from various checkpoint formats.
        
        Args:
            checkpoint: Loaded checkpoint object
            
        Returns:
            dict: State dictionary
        """
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Try common keys for state dict
            for key in ['state_dict', 'model_state_dict', 'model', 'net']:
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    return checkpoint[key]
            
            # If checkpoint itself looks like state dict
            if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                return checkpoint
        
        # Return as-is if it's already a state dict
        return checkpoint
    
    @staticmethod
    def _extract_backbone_weights(state_dict):
        """Extract and filter backbone-relevant weights from state dict.
        
        Args:
            state_dict: Full state dictionary
            
        Returns:
            dict: Filtered backbone state dictionary
        """
        backbone_state_dict = {}
        
        for key, value in state_dict.items():
            # Remove common prefixes
            clean_key = key
            for prefix in ['backbone.', 'module.', 'encoder.', 'model.']:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
                    break
            
            # Only include backbone layers
            if clean_key.startswith(WeightLoader.BACKBONE_PREFIXES):
                backbone_state_dict[clean_key] = value
        
        return backbone_state_dict
    
    @staticmethod
    def validate_channel_compatibility(state_dict, backbone):
        """Validate that input channels match between weights and model.
        
        Args:
            state_dict: State dictionary to validate
            backbone: Backbone module
            
        Returns:
            tuple: (is_compatible, message)
        """
        if 'conv1.weight' not in state_dict:
            return True, "No conv1 weights to validate"
        
        weight_shape = state_dict['conv1.weight'].shape
        model_in_channels = backbone.conv1.in_channels
        weight_in_channels = weight_shape[1]
        
        if model_in_channels != weight_in_channels:
            msg = (f"Channel mismatch: model expects {model_in_channels}, "
                   f"weights have {weight_in_channels}")
            return False, msg
        
        return True, "Channels compatible"


class FarSeg(Module):
    """Foreground-Aware Relation Network (FarSeg).

    This model can be used for binary- or multi-class object segmentation, such as
    building, road, ship, and airplane segmentation. It can be also extended as a change
    detection model. It features a foreground-scene relation module to model the
    relation between scene embedding, object context, and object feature, thus improving
    the discrimination of object feature representation.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/pdf/2011.09766
    """

    def __init__(
        self,
        backbone: str = 'resnet50',
        classes: int = 16,
        backbone_weights: WeightsEnum | str | Path | dict | None = None,
    ) -> None:
        """Initialize a new FarSeg model.

        Args:
            backbone: name of ResNet backbone, one of ["resnet18", "resnet34",
                "resnet50", "resnet101"]
            classes: number of output segmentation classes
            backbone_weights: Pre-trained model weights to use. Can be:
                - TorchGeo WeightsEnum (e.g., ResNet50_Weights.LANDSAT_TM_TOA_MOCO)
                - TorchVision WeightsEnum (e.g., ResNet50_Weights.IMAGENET1K_V1)
                - Path to local checkpoint file (.pth, .pt, .ckpt)
                - HuggingFace model ID (e.g., 'username/model-name')
                - State dictionary (dict)
                - None for random initialization

        .. versionchanged:: 0.9
           Enhanced to support multiple weight sources and formats with intelligent
           detection and loading strategies.
        """
        super().__init__()
        if backbone in ['resnet18', 'resnet34']:
            max_channels = 512
        elif backbone in ['resnet50', 'resnet101']:
            max_channels = 2048
        else:
            raise ValueError(f'unknown backbone: {backbone}.')

        # Initialize backbone (weights loaded separately based on type)
        self.backbone = getattr(resnet, backbone)(weights=None)
        
        # Load weights using multi-strategy approach
        if backbone_weights is not None:
            self._load_backbone_weights(backbone_weights, backbone)

        self.fpn = FPN(
            in_channels_list=[max_channels // (2 ** (3 - i)) for i in range(4)],
            out_channels=256,
        )
        self.fsr = _FSRelation(max_channels, [256] * 4, 256)
        self.decoder = _LightWeightDecoder(256, 128, classes)
    
    def _load_backbone_weights(self, weights, backbone_name: str) -> None:
        """Load backbone weights using intelligent multi-strategy approach.
        
        This method detects the weight type and applies the appropriate loading
        strategy. It includes fallback mechanisms and comprehensive error handling.
        
        Args:
            weights: Weight source (WeightsEnum, path, dict, etc.)
            backbone_name: Name of the backbone architecture
        """
        # Detect weight type
        weight_type = WeightLoader.detect_weight_type(weights)
        logger.info(f"Detected weight type: {weight_type}")
        
        # Apply strategy based on detected type
        success = False
        
        if weight_type == 'torchgeo':
            # Strategy 1: TorchGeo custom weights
            success = WeightLoader.load_torchgeo_weights(weights, self.backbone)
            
        elif weight_type == 'torchvision':
            # Strategy 2: TorchVision native weights
            try:
                self.backbone = WeightLoader.load_torchvision_weights(
                    weights, backbone_name
                )
                success = True
            except Exception:
                success = False
            
        elif weight_type == 'local_file':
            # Strategy 3: Local checkpoint file
            success = WeightLoader.load_local_checkpoint(weights, self.backbone)
            
        elif weight_type == 'huggingface':
            # Strategy 4: HuggingFace model hub
            success = WeightLoader.load_huggingface_weights(weights, self.backbone)
            
        elif weight_type == 'state_dict':
            # Strategy 5: Direct state dictionary
            success = WeightLoader.load_state_dict_direct(weights, self.backbone)
            
        else:
            # Unknown type - try generic state dict loading
            logger.warning(f"Unknown weight type, attempting generic loading")
            try:
                if isinstance(weights, dict):
                    success = WeightLoader.load_state_dict_direct(weights, self.backbone)
                else:
                    raise ValueError(f"Unsupported weight type: {type(weights)}")
            except Exception as e:
                logger.error(f"Failed to load weights: {e}")
                success = False
        
        # Handle loading failure
        if not success:
            logger.warning(
                f"Failed to load weights from {weight_type} source. "
                f"Backbone will use random initialization."
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: input image

        Returns:
            output prediction
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        c2 = self.backbone.layer1(x)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        features = [c2, c3, c4, c5]

        coarsest_features = features[-1]
        scene_embedding = F.adaptive_avg_pool2d(coarsest_features, 1)
        fpn_features = self.fpn(
            OrderedDict({f'c{i + 2}': features[i] for i in range(4)})
        )
        features = [v for k, v in fpn_features.items()]
        features = self.fsr(scene_embedding, features)

        logit = self.decoder(features)

        return cast(Tensor, logit)


class _FSRelation(Module):
    """F-S Relation module."""

    def __init__(
        self,
        scene_embedding_channels: int,
        in_channels_list: list[int],
        out_channels: int,
    ) -> None:
        """Initialize the _FSRelation module.

        Args:
            scene_embedding_channels: number of scene embedding channels
            in_channels_list: a list of input channels
            out_channels: number of output channels
        """
        super().__init__()

        self.scene_encoder = ModuleList(
            [
                Sequential(
                    Conv2d(scene_embedding_channels, out_channels, 1),
                    ReLU(True),
                    Conv2d(out_channels, out_channels, 1),
                )
                for _ in range(len(in_channels_list))
            ]
        )

        self.content_encoders = ModuleList()
        self.feature_reencoders = ModuleList()
        for c in in_channels_list:
            self.content_encoders.append(
                Sequential(
                    Conv2d(c, out_channels, 1), BatchNorm2d(out_channels), ReLU(True)
                )
            )
            self.feature_reencoders.append(
                Sequential(
                    Conv2d(c, out_channels, 1), BatchNorm2d(out_channels), ReLU(True)
                )
            )

        self.normalizer = Sigmoid()

    def forward(self, scene_feature: Tensor, features: list[Tensor]) -> list[Tensor]:
        """Forward pass of the model."""
        # [N, C, H, W]
        content_feats = [
            c_en(p_feat) for c_en, p_feat in zip(self.content_encoders, features)
        ]
        scene_feats = [op(scene_feature) for op in self.scene_encoder]
        relations = [
            self.normalizer((sf * cf).sum(dim=1, keepdim=True))
            for sf, cf in zip(scene_feats, content_feats)
        ]

        p_feats = [op(p_feat) for op, p_feat in zip(self.feature_reencoders, features)]

        refined_feats = [r * p for r, p in zip(relations, p_feats)]

        return refined_feats


class _LightWeightDecoder(Module):
    """Light Weight Decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_classes: int,
        in_feature_output_strides: list[int] = [4, 8, 16, 32],
        out_feature_output_stride: int = 4,
    ) -> None:
        """Initialize the _LightWeightDecoder module.

        Args:
            in_channels: number of channels of input feature maps
            out_channels: number of channels of output feature maps
            num_classes: number of output segmentation classes
            in_feature_output_strides: output stride of input feature maps at different
                levels
            out_feature_output_stride: output stride of output feature maps
        """
        super().__init__()

        self.blocks = ModuleList()
        for in_feat_os in in_feature_output_strides:
            num_upsample = int(math.log2(int(in_feat_os))) - int(
                math.log2(int(out_feature_output_stride))
            )
            num_layers = num_upsample if num_upsample != 0 else 1
            self.blocks.append(
                Sequential(
                    *[
                        Sequential(
                            Conv2d(
                                in_channels if idx == 0 else out_channels,
                                out_channels,
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            BatchNorm2d(out_channels),
                            ReLU(inplace=True),
                            (
                                UpsamplingBilinear2d(scale_factor=2)
                                if num_upsample != 0
                                else Identity()
                            ),
                        )
                        for idx in range(num_layers)
                    ]
                )
            )

        self.classifier = Sequential(
            Conv2d(out_channels, num_classes, 3, 1, 1),
            UpsamplingBilinear2d(scale_factor=4),
        )

    def forward(self, features: list[Tensor]) -> Tensor:
        """Forward pass of the model."""
        inner_feat_list = []
        for idx, block in enumerate(self.blocks):
            decoder_feat = block(features[idx])
            inner_feat_list.append(decoder_feat)

        out_feat = sum(inner_feat_list) / len(inner_feat_list)
        out_feat = self.classifier(out_feat)

        return cast(Tensor, out_feat)
