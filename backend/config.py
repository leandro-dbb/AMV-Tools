"""App-wide constants and the default settings used when settings table is empty."""
from __future__ import annotations

VIDEO_EXTENSIONS = (
    ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm",
    ".ts", ".m2ts", ".mts", ".mpg", ".mpeg", ".vob", ".m4v",
    ".f4v", ".3gp", ".ogv", ".mxf",
)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")

DEFAULT_SIGLIP = "google/siglip2-base-patch16-naflex"
# Checkpoints that ship as model_type="siglip" but were uploaded under the
# siglip2-* prefix. Transformers' Siglip2Model rejects them because the patch
# embedding shape differs (conv2D vs. linear projection). We rewrite them on
# load to the NaFlex variant that is genuinely Siglip2-compatible.
_LEGACY_SIGLIP_REMAP = {
    "google/siglip2-base-patch16-256": DEFAULT_SIGLIP,
}
DEFAULT_WD_TAGGER = "SmilingWolf/wd-vit-tagger-v3"

DEFAULT_SETTINGS = {
    "indexing": {
        "device": "auto",
        "batch_size": 8,
        "mode": "fast",
        "sub_segmentation": False,
        "sub_segmentation_threshold": 0.30,
        # Off by default: the MiniEditor streams the source file directly via
        # the /api/scene/<id>/source endpoint (FastAPI handles Range requests
        # so the <video> tag seeks within MKV/MP4 just fine). Turn this on if
        # your library has codecs Chromium can't decode (HEVC, ProRes, etc.)
        # and you want a pre-baked H.264 proxy as a compatibility fallback.
        "generate_proxies": False,
        "proxy_quality": "medium",
        "auto_skip_indexed": True,
        "scene_detect_threshold": 0.40,
    },
    "models": {
        "wd_tagger_variant": DEFAULT_WD_TAGGER,
        "siglip_variant": DEFAULT_SIGLIP,
        "default_tag_threshold": 0.50,
        "vram_idle_offload": 30,
        "use_tensorrt": False,
        # Balanced default: so400m model + 128 patches. The model size is what
        # drives proper-noun precision (anime characters, Pokémon, etc.) — bumping
        # to 256 patches only gains ~5-10% on fine visual detail and costs ~50% more
        # wallclock per episode. Power users can raise to 256 in Settings.
        "siglip_max_num_patches": 128,
        # Mask / alpha-export feature. OFF by default: the roto pipeline is
        # still beta and not production-ready (matte quality, hair, motion blur
        # and low-contrast backgrounds remain unreliable). Power users can opt
        # in from Settings > Models. The model weights are lazy-loaded on first
        # Mask Mode open, so flipping the toggle does not itself download
        # anything.
        "enable_mask": False,
        # Which segmentation engine to use:
        #   - "birefnet" (default): automatic foreground segmentation, trained
        #     on illustrations/portraits/DIS5K. Dramatically better than SAM 2
        #     on cel-shaded anime content; no clicks needed.
        #   - "sam2": click→mask + temporal propagation. Use this when you need
        #     to isolate ONE specific character among several (BiRefNet can't
        #     disambiguate between subjects).
        #   - "matanyone": BiRefNet seeds the first frame, MatAnyone propagates
        #     temporally for frames 2..N. Reduces silhouette flicker on motion-
        #     heavy clips. Slower (~3× track time) and pulls a 500 MB model on
        #     first use.
        "mask_engine": "birefnet",
        "birefnet_variant": "general",
        # SAM 2.1 variant — base_plus is the qual/VRAM sweet spot on 8 GB GPUs
        # once SigLIP/tagger have been offloaded by the model rotation.
        "sam2_variant": "base_plus",
        # Max side used when extracting frames for roto. Hair detail is often
        # only a few pixels wide; 1080 is slower than 720 but much less likely
        # to erase anime hair gaps before the model sees them.
        "mask_max_dim": 1080,
        # Hard-mask cleanup for SAM2/manual masks. Soft alpha mattes from
        # BiRefNet/MatAnyone skip this path so hair tips and antialiased anime
        # line work keep their fractional alpha at export time.
        "mask_shrink_px": 0,
        # Colour-based edge suppression for hard masks only. Disabled by
        # default because it can eat real hair/detail when foreground colours
        # are close to the background.
        "mask_bg_suppress_enabled": False,
        # Euclidean RGB distance threshold. Lower = preserves more subtle
        # FG details, leaves more halo. Higher = aggressive cleanup, may
        # eat fine character details whose colour happens to match the BG.
        "mask_bg_suppress_threshold": 25.0,
        # Soft-alpha cleanup for BiRefNet/MatAnyone. Removes background-colour
        # pixels from the alpha boundary band, which fixes floor/wall patches
        # caught in concave hair gaps without eroding the whole character.
        "mask_soft_bg_suppress_enabled": False,
        "mask_soft_bg_suppress_threshold": 25.0,
        "mask_soft_bg_suppress_edge_px": 16,
        "mask_soft_shrink_px": 0,
        "mask_soft_alpha_black": 0.0,
        "mask_soft_alpha_white": 1.0,
        "mask_rgb_decontaminate_enabled": False,
        "hf_token": "",
    },
    "search": {
        "threshold": 0.10,
        "max_results": 200,
        "sort": "relevance",
        # so400m knows proper nouns (anime characters, Pokémon, etc.) on its
        # own; tag_boost was a crutch when we ran the smaller `base` model.
        # Disabled by default — bump back up to 0.10-0.20 if you want hybrid
        # behaviour.
        "tag_boost": 0.0,
    },
    "export": {
        "codec": "libx264",
        "crf": 18,
        "resolution": "source",
        "audio": "copy",
        "naming_template": "{anime}_{episode}_scene_{scene_id}",
        "output_folder": "",
        "open_folder_after": True,
    },
    "databases": {
        "active": [],
        "primary": "",
    },
    "interface": {
        "theme": "dark",
        "thumbnail_size": "medium",
        "hover_delay_ms": 200,
    },
}
