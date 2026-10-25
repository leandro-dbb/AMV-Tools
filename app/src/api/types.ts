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
  // search+ (v0.4) — present on search/similar/browse results
  metrics?: SceneMetrics | null;
  characters?: SceneCharacter[];
  label?: 'positive' | 'negative' | null;
}

export type MotionType =
  | 'static' | 'pan_left' | 'pan_right' | 'tilt_up' | 'tilt_down'
  | 'zoom_in' | 'zoom_out' | 'shake' | 'unknown';

export type ColorBucket =
  | 'red' | 'orange' | 'yellow' | 'green' | 'cyan' | 'blue' | 'purple' | 'pink'
  | 'brown' | 'white' | 'gray' | 'black';

export interface SceneMetrics {
  motion_type: MotionType | null;
  motion_magnitude: number | null;
  intensity: number | null;
  activity: number | null;
  brightness: number | null;
  dominant_color: ColorBucket | null;
  accent_color: ColorBucket | null;
  palette: string[];
}

export interface SceneCharacter {
  id: number;
  name: string;
  score: number;
}

export interface SearchFilters {
  characters?: number[];
  character_mode?: 'any' | 'all';
  motion?: MotionType[];
  motion_min?: number | null;
  intensity_min?: number | null;
  intensity_max?: number | null;
  duration_min_ms?: number | null;
  duration_max_ms?: number | null;
  colors?: ColorBucket[];
  groups?: string[];
  video_ids?: number[];
  exclude_video_ids?: number[];
  favorites?: boolean;
}

export interface CharacterExample {
  id: number;
  scene_id: number | null;
  polarity: 1 | -1;
  crop: [number, number, number, number] | null;
  source: 'scene' | 'crop' | 'image' | 'feedback';
  has_thumbnail: boolean;
}

export interface Character {
  id: number;
  name: string;
  group_name: string | null;
  threshold: number;
  threshold_auto: boolean;
  positives: number;
  negatives: number;
  matches: number;
  cover_scene_id: number | null;
  cover_example_id: number | null;
  scored: boolean;
  examples?: CharacterExample[];
}

export interface AnalysisStatus {
  running: boolean;
  done: number;
  total: number;
  percent: number;
  video: string | null;
  skipped_videos: number;
  error: string | null;
  coverage?: { total: number; analyzed: number; pending: number };
}

export interface SearchFacets {
  motion_types: { id: MotionType; count: number }[];
  colors: { id: ColorBucket; count: number }[];
  groups: { name: string; label: string | null; videos: number }[];
  characters: Character[];
  coverage: { total: number; analyzed: number; pending: number };
  analysis: AnalysisStatus;
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
  group_name: string | null;
  derushed: boolean;
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
  filters?: SearchFilters | null;
}

export interface IndexQueueItem {
  id: number;
  path: string;
  is_directory: boolean;
  recursive: boolean;
  group_name: string | null;
}

export interface ProgressEvent {
  type: 'indexing' | 'proxy' | 'export' | 'idle' | 'install' | 'merge' | 'sam2_track' | 'derush_export' | 'derush_prepare' | 'analysis';
  video?: string;
  video_id?: number;
  current?: number;
  total?: number;
  percent?: number;
  message?: string;
  line?: string;
  done?: boolean;
  ok?: boolean;
  output?: string;
  session_id?: string;
  frame_idx?: number;
}

export interface DerushFolder {
  id: number;
  name: string;
  item_count: number;
}

export interface DerushItem {
  id: number;
  scene_id: number;
  folder_ids: number[];
  custom_name: string | null;
  added_at: string;
  video_id: number;
  video_display: string;
  scene_index: number;
  start_ms: number;
  end_ms: number;
  has_proxy: boolean;
  favorite: boolean;
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
    birefnet_variant: 'general' | 'hr' | 'portrait' | 'anime';
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
    mask_edge_refine_enabled: boolean;
    mask_bg_aware_cleanup_enabled: boolean;
    mask_temporal_smooth_enabled: boolean;
    mask_temporal_smooth_strength: number;
    hf_token: string;
    tier?: string;
  };
  search: {
    threshold: number;
    max_results: number;
    sort: string;
    tag_boost: number;
  };
  searchplus: {
    auto_analyze: boolean;
  };
  derush: {
    keys: Record<string, string>;
  };
  export: {
    codec: string;
    crf: number;
    resolution: string;
    audio: string;
    audio_bitrate_kbps: number;
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

export interface TimelineFormat {
  id: 'fcp7xml' | 'fcpxml' | 'edl' | 'otio';
  extension: string;
  label: string;
  label_fr: string;
  apps: string[];
  description: string;
}

export interface TimelineTrimRange {
  start_ms: number;
  end_ms: number;
}

export interface TimelineOptions {
  sequence_fps?: string;
  resolution?: string;
  layout?: 'sequential' | 'gaps';
  gap_frames?: number;
  handles_frames?: number;
  drop_frame?: boolean;
  markers?: boolean;
  include_audio?: boolean;
  collect_media?: boolean;
  path_mode?: 'absolute' | 'relative';
  title?: string;
  otio_use_library?: boolean;
  trims?: Record<string, TimelineTrimRange>;
}

export interface TimelineSelection {
  scene_ids?: number[];
  derush_item_ids?: number[];
  derush_folder_id?: number;
  derush_scope?: 'all' | 'folder' | 'root' | 'favorites';
  options?: TimelineOptions;
}

export interface TimelinePreview {
  clip_count: number;
  duration_frames: number;
  duration_seconds: number;
  sequence_fps: string;
  sequence_fps_decimal: number;
  drop_frame: boolean;
  width: number;
  height: number;
  title: string;
  sources: string[];
  warnings: string[];
}

export interface TimelineExportResult extends TimelinePreview {
  ok: boolean;
  format: string;
  output: string;
  collected_dir: string | null;
}

export interface SystemStatus {
  ready: boolean;
  setup_required: boolean;
  device: string;
  gpu_name?: string;
  vram_gb?: number | null;
  models_loaded: boolean;
  python_version: string;
  tagger_provider?: string | null;
  warnings?: string[];
  requested_device?: string;
  /** Non-null when a GPU backend was requested but the app fell back to CPU. */
  device_fallback?: string | null;
}

export interface DiagnosticModel {
  key: 'siglip' | 'wd_tagger' | 'birefnet' | 'sam2' | string;
  label: string;
  repo_id: string;
  enabled: boolean;
  present: boolean;
  loaded: boolean;
  size_bytes: number | null;
  file_count: number;
  error: string | null;
}

export interface DiagnosticLogEntry {
  time: string;
  level: string;
  logger: string;
  message: string;
  traceback?: string;
}

export interface AutotuneSetting {
  value: number | string;
  default: number | string;
  unit: string;
  measures: Record<string, number | null>;
  note?: string;
}

export interface AutotuneEntry {
  platform: string;
  measured_at: string;
  duration_s: number;
  source: 'library' | 'synthetic';
  settings: Record<string, AutotuneSetting>;
  skipped: Record<string, string>;
  errors: Record<string, string>;
}

export interface AutotuneStatus {
  state: 'idle' | 'waiting' | 'running' | 'done' | 'failed' | 'postponed' | 'disabled';
  step: string | null;
  error: string | null;
  running: boolean;
  key: string;
  fingerprint: Record<string, string | number | null>;
  entry: AutotuneEntry | null;
  cache_path: string;
}

export interface DiagnosticsReport {
  app: { name: string; version: string; frozen: boolean };
  python: string;
  platform: string;
  torch: { version: string | null; cuda_available?: boolean | null; mps_available?: boolean | null; cuda_version?: string | null; error?: string };
  ffmpeg: { path: string | null; version: string | null; error?: string };
  device: {
    requested: string;
    active: string;
    name: string | null;
    vram_gb: number | null;
    gpu_detected: boolean;
    fallback_reason: string | null;
    error: string | null;
  };
  data_dir: { path: string; total_gb?: number; free_gb?: number; used_percent?: number | null; error?: string };
  models: DiagnosticModel[];
  logs: { path: string; dir: string; recent_errors: DiagnosticLogEntry[] };
  settings: { indexing_device: string; mask_enabled: boolean };
  markdown: string;
}

export interface ModelDownloadResult {
  ok: boolean;
  repo_id?: string;
  paths?: string[];
  error?: string;
  hint?: string;
  attempts?: number;
}

// Model tiers (Settings > Models): models picked for the detected GPU memory,
// plus the embeddings-only re-index of the primary database.
export type ModelTierPlatform = 'nvidia' | 'mac' | 'cpu';
export type ModelTierRole = 'embedding' | 'tagger' | 'upscale' | 'interpolate' | 'depth';

export interface ModelTierModel {
  id: string;
  label: string;
  size_mb: number;
  license: string;
  non_commercial?: boolean;
  patches?: number;
}

export interface ModelTier {
  id: string;
  min_gb: number | null;
  models: Partial<Record<ModelTierRole, ModelTierModel>>;
}

export interface ReembedJob {
  running: boolean;
  percent: number;
  message: string;
  error: string | null;
  ok?: boolean;
  done?: boolean;
  videos_missing?: number;
}

export interface ModelTiersInfo {
  platform: ModelTierPlatform;
  detected_gb: number | null;
  auto_tier: string;
  selected: string;
  effective_tier: string;
  tiers: ModelTier[];
  embedding: {
    db_model: string;
    target_model: string;
    reindex_needed: boolean;
    archived_models: string[];
    job: ReembedJob | null;
  };
}

export interface ReembedStartResult {
  ok: boolean;
  error?: string;
  videos_missing?: number;
}

export interface ReembedEvent {
  type: 'reembed';
  percent: number;
  message: string;
  done: boolean;
  ok: boolean;
  videos_missing: number;
}
