# Releasing AMV Tools

Releases are built by GitHub Actions from a `v*` tag. The workflow produces
Windows, macOS and Linux installers, then attaches them to a **draft** GitHub
Release. A human still reviews the draft and publishes it.

## 1. Prepare the version

1. Update `version` in `pyproject.toml` and `app/package.json` (keep them in
   sync), and add the matching section to `CHANGELOG.md`.
2. Run the same checks as CI locally:

   ```bash
   uv sync --extra cpu
   uv run pytest -m "not slow"
   cd app
   npm ci
   npx tsc --noEmit
   npx vitest run
   npm run build
   ```

3. Commit, then tag and push:

   ```bash
   git tag v0.5.0-beta.1
   git push origin v0.5.0-beta.1
   ```

Pushing the tag runs `.github/workflows/build.yml`. Everyday push/PR validation
runs `.github/workflows/ci.yml` (backend pytest on CPU + frontend type-check,
unit tests and Vite build).

## 2. What the build produces

| Platform | Runner | Artifacts |
| --- | --- | --- |
| Windows | `windows-latest` | NSIS installer `.exe` (x64) |
| macOS | `macos-latest` | `.dmg` + `.zip` (arm64) |
| Linux | `ubuntu-latest` | `.AppImage` + `.deb` (x64) |

Linux packages are built with electron-builder's `AppImage` and `deb` targets
(see `app/package.json`). macOS builds can only run on a macOS runner.

## 3. Signing and notarization (optional)

All secrets live under **Settings → Secrets and variables → Actions**. When a
secret is absent, the corresponding step is skipped and the build stays
unsigned — CI never fails because signing is not configured.

### Windows

Create repository secrets:

- `CSC_LINK` — base64 of the `.pfx` certificate (or a URL).
- `CSC_KEY_PASSWORD` — the certificate password.

electron-builder signs the installer automatically when both are present.

### macOS

Create the same `CSC_LINK` / `CSC_KEY_PASSWORD` (Developer ID Application
certificate), plus:

- `APPLE_ID` — Apple account email.
- `APPLE_APP_SPECIFIC_PASSWORD` — app-specific password from appleid.apple.com.
- `APPLE_TEAM_ID` — 10-character Team ID.

`app/scripts/notarize.cjs` is wired as the electron-builder `afterSign` hook. It
notarizes only when the three Apple variables are present; otherwise it logs a
skip and the app remains unsigned. Without `CSC_LINK`,
`app/scripts/adhoc-sign.cjs` (`afterPack`) signs the bundle **ad hoc** so that
Apple Silicon accepts to run it, and the build workflow checks the signature. The macOS target keeps `hardenedRuntime`
enabled with the entitlements in `app/build/entitlements.mac.plist`.

`CSC_IDENTITY_AUTO_DISCOVERY` is derived from whether `CSC_LINK` is set, so a
fork without secrets still builds.

## 4. Publish the draft

1. Open the **Releases** page and edit the auto-created draft.
2. Verify the asset list (Windows `.exe`, macOS `.dmg`/`.zip`, Linux
   `.AppImage`/`.deb`).
3. Publish when the smoke test on a clean machine passes.

Unsigned Windows installers trigger SmartScreen; macOS unsigned builds require
the usual right-click → Open workaround. Signed/notarized builds are the
recommended path for end users.

## 5. Manual/local packaging

```bash
cd app
npm run build
npx electron-builder --publish never          # current OS
# Linux only:
npx electron-builder --linux AppImage deb --publish never
# Fully unsigned (no certificate lookup):
npx cross-env CSC_IDENTITY_AUTO_DISCOVERY=false electron-builder --publish never
```

Artifacts land in `app/release/` (ignored by Git).
