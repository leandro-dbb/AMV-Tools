import type { Dict } from './index';

// Strings for the model tiers block at the top of Settings > Models
// (ModelTiersSection) and the embeddings-only re-index banner added in 0.4.
export const modelsStrings: Dict = {
  // tier select
  'models.tier.label': { en: 'Model tier', fr: 'Palier de modèles' },
  'models.tier.hint': {
    en: 'Picks the search, tagging and finishing models for your machine. Auto chooses from the detected VRAM (NVIDIA) or unified memory (Mac). Bigger tiers give better search, tags and finishing, but are slower and use more memory. Max quality is the slowest and the best; Fast is the lightest.',
    fr: 'Choisit les modèles de recherche, de tags et de finition adaptés à ta machine. Auto se base sur la VRAM détectée (NVIDIA) ou la mémoire unifiée (Mac). Les paliers plus gros donnent une meilleure recherche, de meilleurs tags et une meilleure finition, mais sont plus lents et utilisent plus de mémoire. Qualité max est le plus lent et le meilleur ; Rapide est le plus léger.',
  },
  'models.tier.auto': { en: 'Auto (detected: {detected} → {tier})', fr: 'Auto (détecté : {detected} → {tier})' },
  'models.tier.autoUnknown': { en: 'Auto ({tier})', fr: 'Auto ({tier})' },
  'models.tier.fast': { en: 'Fast', fr: 'Rapide' },
  'models.tier.max': { en: 'Max quality', fr: 'Qualité max' },
  'models.tier.custom': { en: 'Custom', fr: 'Personnalisé' },
  'models.tier.gb': { en: '{gb} GB', fr: '{gb} Go' },
  'models.tier.macGb': { en: 'Mac {gb} GB', fr: 'Mac {gb} Go' },
  'models.tier.macGbPlus': { en: 'Mac {gb} GB and more', fr: 'Mac {gb} Go et plus' },
  'models.tier.overMemory': { en: '{tier} (more than detected)', fr: '{tier} (plus que détecté)' },
  'models.detected.vram': { en: '{gb} GB VRAM', fr: '{gb} Go de VRAM' },
  'models.detected.unified': { en: '{gb} GB unified memory', fr: '{gb} Go de mémoire unifiée' },
  'models.detected.ram': { en: '{gb} GB', fr: '{gb} Go' },
  'models.unavailable': { en: 'Model tiers unavailable: {message}', fr: 'Paliers de modèles indisponibles : {message}' },

  // effective models table
  'models.table.function': { en: 'Function', fr: 'Fonction' },
  'models.table.model': { en: 'Model', fr: 'Modèle' },
  'models.table.size': { en: 'Size', fr: 'Taille' },
  'models.table.license': { en: 'License', fr: 'Licence' },
  'models.table.patches': { en: '{n} patches', fr: '{n} patches' },
  'models.table.custom': {
    en: 'Custom: the models below are your own choices.',
    fr: 'Personnalisé : les modèles ci-dessous sont tes propres choix.',
  },
  'models.role.embedding': { en: 'Search embeddings', fr: 'Embeddings de recherche' },
  'models.role.tagger': { en: 'Tagger', fr: 'Tagger' },
  'models.role.upscale': { en: 'Upscale', fr: 'Upscale' },
  'models.role.interpolate': { en: 'Interpolation', fr: 'Interpolation' },
  'models.role.depth': { en: 'Depth', fr: 'Profondeur' },
  'models.size.mb': { en: '{n} MB', fr: '{n} Mo' },
  'models.size.gb': { en: '{n} GB', fr: '{n} Go' },
  'models.lockedNote': {
    en: 'The tagger, SigLIP and patches settings below are set by the tier — choose Custom to edit them.',
    fr: 'Les réglages tagger, SigLIP et patches ci-dessous sont fixés par le palier — choisis Personnalisé pour les modifier.',
  },

  // re-index banner
  'models.reindex.needed': {
    en: 'This database was indexed with {db_model}. The selected tier uses {target_model}: re-index the embeddings (cuts and tags are kept) to use it. Search keeps working with the old model meanwhile.',
    fr: 'Cette base a été indexée avec {db_model}. Le palier choisi utilise {target_model} : réindexe les embeddings (les coupes et les tags sont conservés) pour l\'utiliser. La recherche continue de fonctionner avec l\'ancien modèle en attendant.',
  },
  'models.reindex.start': { en: 'Re-index embeddings', fr: 'Réindexer les embeddings' },
  'models.reindex.starting': { en: 'Starting…', fr: 'Démarrage…' },
  'models.reindex.force': {
    en: 'Re-index anyway (drop missing videos from search)',
    fr: 'Réindexer quand même (retirer les vidéos manquantes de la recherche)',
  },
  'models.reindex.cancel': { en: 'Cancel', fr: 'Annuler' },
  'models.reindex.running': { en: 'Re-indexing embeddings… {percent}%', fr: 'Réindexation des embeddings… {percent} %' },
  'models.reindex.done': { en: 'Embeddings re-indexed.', fr: 'Embeddings réindexés.' },
  'models.reindex.error': { en: 'Re-index failed: {message}', fr: 'Échec de la réindexation : {message}' },
  'models.reindex.missing': {
    en: '{n} video file(s) can no longer be found, so their clips cannot be re-embedded.',
    fr: '{n} fichier(s) vidéo introuvable(s) : leurs clips ne peuvent pas être réindexés.',
  },
  'models.reindex.busy': {
    en: 'Indexing is running — wait for it to finish, then re-index the embeddings.',
    fr: 'Une indexation est en cours — attends qu\'elle se termine, puis réindexe les embeddings.',
  },
  'models.reindex.unsupported': {
    en: 'This sidecar version cannot re-index embeddings — update AMV Tools.',
    fr: 'Cette version du sidecar ne sait pas réindexer les embeddings — mets AMV Tools à jour.',
  },
  'models.reindex.archived': { en: 'Archived embeddings: {list}', fr: 'Embeddings archivés : {list}' },
};
