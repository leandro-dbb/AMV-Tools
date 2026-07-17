import type { Dict } from './index';

// Shared vocabulary. Component dictionaries live next to this file; prefer a
// component-scoped key unless the exact same wording is reused across tabs.
export const commonStrings: Dict = {
  'common.cancel': { en: 'Cancel', fr: 'Annuler' },
  'common.close': { en: 'Close', fr: 'Fermer' },
  'common.create': { en: 'Create', fr: 'Créer' },
  'common.delete': { en: 'Delete', fr: 'Supprimer' },
  'common.rename': { en: 'Rename', fr: 'Renommer' },
  'common.export': { en: 'Export', fr: 'Exporter' },
  'common.import': { en: 'Import', fr: 'Importer' },
  'common.loading': { en: 'Loading…', fr: 'Chargement…' },
  'common.saving': { en: 'Saving', fr: 'Enregistrement' },
  'common.error': { en: 'Error: {message}', fr: 'Erreur : {message}' },
  'common.small': { en: 'Small', fr: 'Petit' },
  'common.medium': { en: 'Medium', fr: 'Moyen' },
  'common.large': { en: 'Large', fr: 'Grand' },
  'common.off': { en: 'Off', fr: 'Désactivé' },
  'common.default': { en: 'default', fr: 'défaut' },
  'common.recommended': { en: 'recommended', fr: 'recommandé' },
};
