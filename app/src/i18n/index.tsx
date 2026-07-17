// Lightweight i18n: one flat dictionary of { en, fr } entries, a context that
// holds the active language (persisted to localStorage), and a t() hook.
// Switching language re-renders the whole tree instantly — no reload needed.
import { createContext, useCallback, useContext, useMemo, useState } from 'react';
import { commonStrings } from './strings.common';
import { appStrings } from './strings.app';
import { settingsStrings } from './strings.settings';
import { searchStrings } from './strings.search';
import { tagsStrings } from './strings.tags';
import { derushStrings } from './strings.derush';
import { libraryStrings } from './strings.library';
import { editorStrings } from './strings.editor';
import { maskStrings } from './strings.mask';
import { onboardingStrings } from './strings.onboarding';

export type Lang = 'en' | 'fr';
export type Entry = { en: string; fr: string };
export type Dict = Record<string, Entry>;

const DICT: Dict = {
  ...commonStrings,
  ...appStrings,
  ...settingsStrings,
  ...searchStrings,
  ...tagsStrings,
  ...derushStrings,
  ...libraryStrings,
  ...editorStrings,
  ...maskStrings,
  ...onboardingStrings,
};

const LANG_KEY = 'amv-lang';

export type TFunc = (key: string, vars?: Record<string, string | number>) => string;

function detectLang(): Lang {
  try {
    const stored = localStorage.getItem(LANG_KEY);
    if (stored === 'en' || stored === 'fr') return stored;
  } catch {}
  return navigator.language?.toLowerCase().startsWith('fr') ? 'fr' : 'en';
}

interface I18nContext {
  lang: Lang;
  setLang: (l: Lang) => void;
  t: TFunc;
}

const Ctx = createContext<I18nContext>({ lang: 'en', setLang: () => {}, t: (k) => k });

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [lang, setLangState] = useState<Lang>(detectLang);

  const setLang = useCallback((l: Lang) => {
    setLangState(l);
    try { localStorage.setItem(LANG_KEY, l); } catch {}
  }, []);

  const t = useCallback<TFunc>((key, vars) => {
    const entry = DICT[key];
    // Missing key or missing translation falls back to English, then to the
    // key itself, so an incomplete dictionary never breaks the UI.
    let s = entry ? (entry[lang] || entry.en) : key;
    if (vars) {
      for (const [k, v] of Object.entries(vars)) s = s.split(`{${k}}`).join(String(v));
    }
    return s;
  }, [lang]);

  const value = useMemo(() => ({ lang, setLang, t }), [lang, setLang, t]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useI18n(): I18nContext {
  return useContext(Ctx);
}

export function useT(): TFunc {
  return useContext(Ctx).t;
}
