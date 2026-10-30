// electron-builder afterPack hook.
//
// Apple Silicon refuses to launch code without a valid signature, and
// electron-builder leaves the bundle unsigned when no certificate is set
// (CSC_IDENTITY_AUTO_DISCOVERY=false), after having edited the Electron
// bundle (Info.plist, renamed helpers), which breaks Electron's own seal.
// So without a certificate we sign the whole bundle ad hoc ("-"): no
// identity, no notarization, but macOS arm64 accepts to run it once the user
// has cleared the quarantine (right click → Open, or xattr). No hardened
// runtime: with ad-hoc signatures its library validation rejects Electron's
// frameworks. When CSC_LINK is set, electron-builder signs for real
// afterwards and this hook does nothing.
const { execFileSync } = require('node:child_process');
const path = require('node:path');

module.exports = async function afterPack(context) {
  if (context.electronPlatformName !== 'darwin' || process.platform !== 'darwin') return;
  if (process.env.CSC_LINK) return;

  const appName = context.packager.appInfo.productFilename;
  const appPath = path.join(context.appOutDir, `${appName}.app`);
  console.log(`[adhoc-sign] no certificate — signing ${appPath} ad hoc`);
  execFileSync('codesign', ['--force', '--deep', '--sign', '-', appPath], { stdio: 'inherit' });
  execFileSync('codesign', ['--verify', '--deep', '--strict', '--verbose=2', appPath], { stdio: 'inherit' });
};
