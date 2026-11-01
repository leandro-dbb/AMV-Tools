import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { AutotuneSetting } from '../api/types';
import { useT } from '../i18n';

// Settings > Indexing "Precise cuts": what the cut network (TransNetV2)
// costs on this machine, from the automatic tuning (backend/autotune,
// "cutnet": "auto" = it runs by itself, "manual" = this setting, off by
// default). Returns the row's hint and whether the network is on anyway.
export function usePreciseCuts(): { hint: string; auto: boolean } {
  const t = useT();
  const [rec, setRec] = useState<AutotuneSetting | null | undefined>(undefined);
  useEffect(() => {
    let alive = true;
    api.autotune().then(
      (s) => alive && setRec(s.entry?.settings?.cutnet ?? null),
      () => alive && setRec(null),
    );
    return () => {
      alive = false;
    };
  }, []);
  const base = t('settings.indexing.preciseCuts.base');
  if (!rec) return { hint: `${base} ${rec === null ? t('settings.indexing.preciseCuts.unmeasured') : ''}`.trim(), auto: false };
  const m = rec.measures as Record<string, number | null>;
  if (rec.value === 'auto') {
    const pct = Math.round(100 * (m.overhead ?? 0));
    return { hint: `${base} ${t('settings.indexing.preciseCuts.auto', { pct: `+${pct}` })}`, auto: true };
  }
  const minutes = Math.max(1, Math.round((m.extra_s_per_episode ?? 0) / 60));
  return { hint: `${base} ${t('settings.indexing.preciseCuts.manual', { minutes })}`, auto: false };
}
