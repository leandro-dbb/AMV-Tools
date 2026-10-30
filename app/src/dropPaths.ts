// Absolute paths of files/folders dropped on the window. Electron 32+ no
// longer sets File.path (the drop zone silently did nothing), the preload
// exposes webUtils.getPathForFile instead; File.path stays as a fallback.
export function droppedPaths(files: File[]): string[] {
  const viaBridge = window.amvBridge?.pathForFile;
  return files
    .map((f) => {
      try {
        return viaBridge ? viaBridge(f) : (f as File & { path?: string }).path ?? '';
      } catch {
        return '';
      }
    })
    .filter((p): p is string => !!p);
}
