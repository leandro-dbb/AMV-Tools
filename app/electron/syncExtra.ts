// Optional extra a syncing `uv run` of the sidecar must install.
//
// Once the onboarding has installed a backend ("cu130", "cpu", …, read from
// .setup_complete), every sync keeps its extra: without it uv swaps the CUDA
// torch build for the CPU one.
//
// Before that (very first launch, nothing chosen yet), Linux needs the `cpu`
// extra: with no extra, uv.lock resolves torch 2.4.1 on Linux, which has no
// wheel for Python 3.13 — and uv downloads 3.13 whenever the system has no
// 3.12 (Debian 12, Fedora, Ubuntu 25.04…), so the first launch died with
// "torch==2.4.1 can't be installed". macOS and Windows resolve torch 2.11.0
// from PyPI without an extra.
export function syncExtraArgs(platform: string, backend: string | null): string[] {
  if (backend) return ['--extra', backend, ...(backend === 'dml' ? ['--python', '3.12'] : [])];
  if (platform === 'linux') return ['--extra', 'cpu'];
  return [];
}
