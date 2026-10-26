// Hover preview of a scene played straight from its source file, for when no
// proxy was generated (the default). Loops the scene; reports a decode error
// (e.g. HEVC on Linux) so the caller can fall back to the thumbnail.
import { api } from '../api/client';

export default function SourcePreview({ scene, onFail }: { scene: { id: number; start_ms: number; end_ms: number }; onFail: () => void }) {
  return (
    <video
      data-testid="hover-preview"
      src={`${api.sourceUrl(scene.id)}#t=${(scene.start_ms / 1000).toFixed(2)}`}
      autoPlay muted playsInline className="w-full h-full object-cover"
      onTimeUpdate={(e) => {
        const v = e.currentTarget;
        if (v.currentTime * 1000 >= scene.end_ms) v.currentTime = scene.start_ms / 1000;
      }}
      onError={onFail}
    />
  );
}
