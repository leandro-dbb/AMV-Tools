export interface SceneResult {
  id: number;
  video_id: number;
  video_display: string;
  scene_index: number;
  start_ms: number;
  end_ms: number;
  score: number;
  confidence?: number;
  tags: string[];
  proxy_path?: boolean | string | null;
}

export interface VideoSummary {
  id: number;
  filepath: string;
  display_name: string;
  duration_ms: number;
  fps: number;
  resolution: string;
  scene_count: number;
  status: 'pending' | 'indexing' | 'completed' | 'failed';
}

export interface TagSummary {
  tag: string;
  category: string;
  count: number;
}

export interface SearchRequest {
  query?: string;
  image_path?: string;
  top_k?: number;
  threshold?: number;
  sort?: 'relevance' | 'duration' | 'video' | 'random';
  tag_boost?: number;
}

export interface IndexQueueItem {
  id: number;
  path: string;
  is_directory: boolean;
  recursive: boolean;
}

export interface ProgressEvent {
  type: 'indexing' | 'proxy' | 'export' | 'idle' | 'install' | 'merge' | 'sam2_track';
  video?: string;
  current?: number;
  total?: number;
  percent?: number;
  message?: string;
  line?: string;
  done?: boolean;
  ok?: boolean;
  session_id?: string;
  frame_idx?: number;
}

export interface SegmentPromptBody {
  positive: [number, number][];
  negative?: [number, number][];
  box?: [number, number, number, number] | null;
  reference_frame_offset?: number | null;
  start_ms?: number;
  end_ms?: number;
  engine?: 'birefnet' | 'sam2';
}

export interface SegmentTrackBody extends SegmentPromptBody {
  session_id: string;
}

export interface SegmentPreviewResponse {
  ok: boolean;
  session_id: string;
  scene_id: number;
  n_frames: number;
  frame_w: number;
  frame_h: number;
  reference_frame_offset: number;
  engine: 'birefnet' | 'sam2';
  has_mask: boolean;
  frame_url: string;
  mask_url: string | null;
}

export interface AppSettings {
  indexing: {
    device: string;
    batch_size: number;
    mode: 'fast' | 'balanced' | 'accurate';
    sub_segmentation: boolean;
    sub_segmentation_threshold: number;
    generate_proxies: boolean;
    proxy_quality: 'low' | 'medium' | 'high';
    auto_skip_indexed: boolean;
    scene_detect_threshold: number;
  };
  models: {
    wd_tagger_variant: string;
    siglip_variant: string;
    default_tag_threshold: number;
    vram_idle_offload: number;
    use_tensorrt: boolean;
    siglip_max_num_patches: number;
    enable_mask: boolean;
    mask_engine: 'birefnet' | 'sam2';
    birefnet_variant: 'general' | 'hr' | 'portrait';
    sam2_variant: 'tiny' | 'small' | 'base_plus' | 'large';
    mask_max_dim: number;
    mask_shrink_px: number;
    mask_bg_suppress_enabled: boolean;
    mask_bg_suppress_threshold: number;
    mask_soft_bg_suppress_enabled: boolean;
    mask_soft_bg_suppress_threshold: number;
    mask_soft_bg_suppress_edge_px: number;
    mask_soft_shrink_px: number;
    mask_soft_alpha_black: number;
    mask_soft_alpha_white: number;
    mask_rgb_decontaminate_enabled: boolean;
    hf_token: string;
  };
  search: {
    threshold: number;
    max_results: number;
    sort: string;
    tag_boost: number;
  };
  export: {
    codec: string;
    crf: number;
    resolution: string;
    audio: string;
    naming_template: string;
    output_folder: string;
    open_folder_after: boolean;
  };
  databases: {
    active: string[];
    primary: string;
  };
  interface: {
    theme: string;
    thumbnail_size: string;
    hover_delay_ms: number;
  };
}

export interface SystemStatus {
  ready: boolean;
  setup_required: boolean;
  device: string;
  gpu_name?: string;
  models_loaded: boolean;
  python_version: string;
  tagger_provider?: string | null;
  warnings?: string[];
}
