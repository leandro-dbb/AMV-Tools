import type { Dict } from './index';

// Strings for OnboardingScreen.tsx, BootstrapScreen.tsx and TutorialOverlay.tsx.
export const onboardingStrings: Dict = {
  // --- OnboardingScreen ---
  'onboarding.title': { en: 'Welcome to AMV Tools', fr: 'Bienvenue dans AMV Tools' },
  'onboarding.tagline': { en: 'One-time setup', fr: 'Configuration initiale' },
  'onboarding.intro': {
    en: 'Pick the acceleration backend that matches your hardware. This downloads ~1-3 GB of PyTorch wheels.',
    fr: 'Choisissez le backend d\'accélération adapté à votre matériel. Environ 1 à 3 Go de paquets PyTorch seront téléchargés.',
  },
  'onboarding.introChangePrefix': {
    en: 'You can change it later from ',
    fr: 'Vous pourrez le modifier plus tard depuis ',
  },
  'onboarding.introSettingsPath': { en: 'Settings → Models', fr: 'Paramètres → Modèles' },
  'onboarding.badge.gpuEverywhere': { en: 'GPU everywhere', fr: 'Tout sur GPU' },
  'onboarding.opt.cu130-trt.title': { en: 'NVIDIA CUDA 13 + TensorRT', fr: 'NVIDIA CUDA 13 + TensorRT' },
  'onboarding.opt.cu130-trt.subtitle': { en: 'RTX 20xx and newer · fastest', fr: 'RTX 20xx et plus récentes · le plus rapide' },
  'onboarding.opt.cu130-trt.detail': {
    en: 'SigLIP on CUDA with TensorRT JIT acceleration + wd-tagger on CUDA via ONNX. ~3.5 GB total download (PyTorch + TensorRT + dual CUDA 12/13 runtimes). Pick this if you want maximum speed.',
    fr: 'SigLIP sur CUDA avec accélération JIT TensorRT + wd-tagger sur CUDA via ONNX. Environ 3,5 Go à télécharger au total (PyTorch + TensorRT + runtimes CUDA 12/13). Choisissez cette option pour une vitesse maximale.',
  },
  'onboarding.opt.cu130.title': { en: 'NVIDIA CUDA 13', fr: 'NVIDIA CUDA 13' },
  'onboarding.opt.cu130.subtitle': { en: 'RTX 20xx and newer', fr: 'RTX 20xx et plus récentes' },
  'onboarding.opt.cu130.detail': {
    en: 'SigLIP + wd-tagger both on CUDA. ~2.5 GB total download. No TensorRT — slightly slower for SigLIP but a smaller install and faster first launch.',
    fr: 'SigLIP + wd-tagger tous deux sur CUDA. Environ 2,5 Go à télécharger au total. Sans TensorRT — SigLIP est légèrement plus lent, mais l\'installation est plus légère et le premier lancement plus rapide.',
  },
  'onboarding.opt.cu126.title': { en: 'NVIDIA CUDA 12.6', fr: 'NVIDIA CUDA 12.6' },
  'onboarding.opt.cu126.subtitle': { en: 'GTX 10xx · older drivers', fr: 'GTX 10xx · pilotes plus anciens' },
  'onboarding.opt.cu126.detail': {
    en: 'Use this if your driver is older than 555.',
    fr: 'À utiliser si votre pilote est antérieur à la version 555.',
  },
  'onboarding.opt.dml.title': { en: 'DirectML (AMD / Intel)', fr: 'DirectML (AMD / Intel)' },
  'onboarding.opt.dml.subtitle': { en: 'Windows only', fr: 'Windows uniquement' },
  'onboarding.opt.dml.detail': {
    en: 'AMD Radeon, Intel Arc and any DirectX 12 GPU.',
    fr: 'AMD Radeon, Intel Arc et tout GPU compatible DirectX 12.',
  },
  'onboarding.opt.rocm.title': { en: 'AMD ROCm', fr: 'AMD ROCm' },
  'onboarding.opt.rocm.subtitle': { en: 'Linux only', fr: 'Linux uniquement' },
  'onboarding.opt.rocm.detail': {
    en: 'Native AMD acceleration on Linux.',
    fr: 'Accélération AMD native sous Linux.',
  },
  'onboarding.opt.xpu.title': { en: 'Intel Arc / Xe (XPU)', fr: 'Intel Arc / Xe (XPU)' },
  'onboarding.opt.xpu.subtitle': { en: 'oneAPI', fr: 'oneAPI' },
  'onboarding.opt.xpu.detail': {
    en: 'For Intel Arc A-series and Xe.',
    fr: 'Pour les Intel Arc série A et Xe.',
  },
  'onboarding.opt.cpu.title': { en: 'CPU only', fr: 'CPU uniquement' },
  'onboarding.opt.cpu.subtitle': { en: 'No GPU acceleration', fr: 'Sans accélération GPU' },
  'onboarding.opt.cpu.detail': {
    en: 'Works everywhere, much slower for indexing.',
    fr: 'Fonctionne partout, mais l\'indexation est beaucoup plus lente.',
  },
  'onboarding.installBtn': { en: 'Install and continue', fr: 'Installer et continuer' },
  'onboarding.installingTitle': { en: 'Installing {backend} backend', fr: 'Installation du backend {backend}' },
  'onboarding.installingHint': {
    en: 'This usually takes 2-8 minutes depending on your connection.',
    fr: 'Cela prend généralement 2 à 8 minutes selon votre connexion.',
  },
  'onboarding.logPlaceholder': { en: 'Resolving PyTorch wheels...', fr: 'Résolution des paquets PyTorch...' },
  'onboarding.doneTitle': { en: 'Setup complete', fr: 'Configuration terminée' },
  'onboarding.doneHint': { en: 'Loading the app...', fr: 'Chargement de l\'application...' },
  'onboarding.errorTitle': { en: 'Installation failed', fr: 'Échec de l\'installation' },
  'onboarding.errorFallback': {
    en: 'Installation failed. Retry this backend or choose CPU to open the app.',
    fr: 'L\'installation a échoué. Réessayez ce backend ou choisissez CPU pour ouvrir l\'application.',
  },
  'onboarding.tryAnother': { en: 'Try another backend', fr: 'Essayer un autre backend' },
  'onboarding.version': { en: 'AMV Tools v0.1 alpha', fr: 'AMV Tools v0.1 alpha' },

  // Indexing preference question (speed vs quality)
  'onboarding.profile.title': { en: 'How should your episodes be indexed?', fr: 'Comment indexer tes épisodes ?' },
  'onboarding.profile.fast.title': { en: 'Fast', fr: 'Rapide' },
  'onboarding.profile.fast.detail': {
    en: 'Quicker indexing. Long uncut scenes stay in one block — searches can land less precisely inside them.',
    fr: 'Indexation plus rapide. Les longues scènes sans cut restent en un seul bloc — la recherche peut y être moins précise.',
  },
  'onboarding.profile.quality.title': { en: 'Accurate', fr: 'Précis' },
  'onboarding.profile.quality.detail': {
    en: 'Long scenes are split by content (sub-segmentation), so search and derush land on the exact shot. Indexing takes a bit longer.',
    fr: 'Les scènes longues sont découpées selon leur contenu (sub-segmentation) : la recherche et le dérushage tombent sur le bon plan. L\'indexation prend un peu plus de temps.',
  },
  'onboarding.profile.changeLater': {
    en: 'You can change this anytime in Settings → Indexing.',
    fr: 'Modifiable à tout moment dans Paramètres → Indexation.',
  },

  // --- BootstrapScreen ---
  'bootstrap.failure': { en: 'Backend failure', fr: 'Échec du backend' },
  'bootstrap.starting': { en: 'Starting backend', fr: 'Démarrage du backend' },
  'bootstrap.errorSubtitle': {
    en: 'Something went wrong while starting the Python backend.',
    fr: 'Une erreur est survenue au démarrage du backend Python.',
  },
  'bootstrap.firstLaunch': {
    en: 'First launch — fetching the Python sidecar',
    fr: 'Premier lancement — récupération du sidecar Python',
  },
  'bootstrap.firstLaunchHint': {
    en: 'This usually takes 1-3 minutes the first time. The app will open automatically when the backend is ready.',
    fr: 'Cela prend généralement 1 à 3 minutes la première fois. L\'application s\'ouvrira automatiquement dès que le backend sera prêt.',
  },
  'bootstrap.waitingOutput': { en: 'Waiting for sidecar output…', fr: 'En attente de la sortie du sidecar…' },
  'bootstrap.retryHint': {
    en: 'Close and relaunch AMV Tools to retry. If the problem persists, check the log above for clues (most often: missing network access for the initial wheel download, or insufficient disk space).',
    fr: 'Fermez puis relancez AMV Tools pour réessayer. Si le problème persiste, consultez le journal ci-dessus pour trouver des indices (le plus souvent : pas d\'accès réseau pour le téléchargement initial des paquets, ou espace disque insuffisant).',
  },
  'bootstrap.stage.responding': {
    en: 'Backend is responding — opening the app.',
    fr: 'Le backend répond — ouverture de l\'application.',
  },
  'bootstrap.stage.started': {
    en: 'FastAPI started — handshake in progress.',
    fr: 'FastAPI démarré — négociation en cours.',
  },
  'bootstrap.stage.installed': {
    en: 'Packages installed — launching server.',
    fr: 'Paquets installés — lancement du serveur.',
  },
  'bootstrap.stage.downloading': { en: 'Downloading Python wheels…', fr: 'Téléchargement des paquets Python…' },
  'bootstrap.stage.resolving': { en: 'Resolving dependencies…', fr: 'Résolution des dépendances…' },
  'bootstrap.stage.venv': { en: 'Creating the Python environment…', fr: 'Création de l\'environnement Python…' },
  'bootstrap.stage.runtime': { en: 'Preparing Python runtime…', fr: 'Préparation de l\'environnement Python…' },
  'bootstrap.stage.init': {
    en: 'Initialising — this can take a couple of minutes on first launch.',
    fr: 'Initialisation — cela peut prendre quelques minutes au premier lancement.',
  },

  // --- TutorialOverlay ---
  'tutorial.gettingStarted': { en: 'Getting started', fr: 'Premiers pas' },
  'tutorial.step1.title': { en: '1. Add your library', fr: '1. Ajoutez votre bibliothèque' },
  'tutorial.step1.body': {
    en: 'Open the Library tab and drop a folder of anime episodes into it. AMV Tools will detect cuts, sub-segment long scenes, and run wd-tagger + SigLIP 2.',
    fr: 'Ouvrez l\'onglet Bibliothèque et déposez-y un dossier d\'épisodes d\'anime. AMV Tools détectera les coupes, découpera les longues scènes en sous-segments et exécutera wd-tagger + SigLIP 2.',
  },
  'tutorial.step2.title': { en: '2. Search in plain English', fr: '2. Cherchez en langage naturel' },
  'tutorial.step2.body': {
    en: 'Type "Gojo combat" or "Charizard fire breath" in the Search tab. Drop an image to find similar scenes. Tune the threshold if you want stricter or looser matches.',
    fr: 'Tapez « Gojo combat » ou « Charizard fire breath » dans l\'onglet Recherche. Déposez une image pour trouver des scènes similaires. Ajustez le seuil pour des résultats plus stricts ou plus larges.',
  },
  'tutorial.step3.title': { en: '3. Browse by tag', fr: '3. Parcourez par tag' },
  'tutorial.step3.body': {
    en: 'In the Tags tab, pick one or several videos and a tag (e.g. "fighting") to see every occurrence with hover preview. Multi-select and bulk-export in one click.',
    fr: 'Dans l\'onglet Tags, choisissez une ou plusieurs vidéos et un tag (ex. « fighting ») pour voir chaque occurrence avec un aperçu au survol. Sélection multiple et export groupé en un clic.',
  },
  'tutorial.step4.title': { en: '4. Edit and export', fr: '4. Montez et exportez' },
  'tutorial.step4.body': {
    en: 'Click any scene to open the mini-editor. J/K/L scrub, I/O set in-out, ← → walk the list. Hit Export scene when you\'re done — frame-accurate via ffmpeg.',
    fr: 'Cliquez sur une scène pour ouvrir le mini-éditeur. J/K/L pour naviguer, I/O pour poser les points d\'entrée/sortie, ← → pour parcourir la liste. Cliquez sur Exporter la scène quand vous avez terminé — précis à l\'image près grâce à ffmpeg.',
  },
  'tutorial.skip': { en: 'Skip', fr: 'Passer' },
  'tutorial.next': { en: 'Next', fr: 'Suivant' },
  'tutorial.letsGo': { en: 'Let\'s go', fr: 'C\'est parti' },
};
