/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_PORT?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

// app/package.json version, injected by vite.config.ts
declare const __APP_VERSION__: string;
