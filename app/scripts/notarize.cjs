// electron-builder afterSign hook.
//
// Notarization is opt-in: if the three Apple credentials are not present in
// the environment we return immediately, so local and unauthenticated CI
// builds stay unsigned instead of failing. Windows signing is handled by
// electron-builder itself when CSC_LINK / CSC_KEY_PASSWORD are set.
const path = require('node:path');

module.exports = async function afterSign(context) {
  if (context.electronPlatformName !== 'darwin') return;

  const { APPLE_ID, APPLE_APP_SPECIFIC_PASSWORD, APPLE_TEAM_ID } = process.env;
  if (!APPLE_ID || !APPLE_APP_SPECIFIC_PASSWORD || !APPLE_TEAM_ID) {
    console.log('[notarize] Apple credentials not set — skipping notarization (unsigned build).');
    return;
  }

  const { notarize } = require('@electron/notarize');
  const appName = context.packager.appInfo.productFilename;
  const appPath = path.join(context.appOutDir, `${appName}.app`);

  console.log(`[notarize] notarizing ${appPath}`);
  await notarize({
    appBundleId: context.packager.appInfo.id || 'com.amvtools.app',
    appPath,
    appleId: APPLE_ID,
    appleIdPassword: APPLE_APP_SPECIFIC_PASSWORD,
    teamId: APPLE_TEAM_ID,
  });
  console.log('[notarize] done');
};
