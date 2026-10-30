"""Roto editor (v06): the AI cuts the character out, the user finishes it.

``project``  one shot being rotoscoped: frames, mattes per layer, locked
             keyframes, undo / redo, autosave on disk.
``edges``    the simple edge settings (keep the ink line, soften / harden,
             grow / shrink, smoothing over time), same code for the preview
             and the export.
``suspects`` frames that probably need a look (lost, jump, specks, object
             lost, hole, rough edge), from the roto benchmark's metrics.
``engine``   the models: the AI pass (Stable v5 pipeline), a click (SAM 2.1),
             propagation (SAM 2.1 video predictor, optical flow without one).
``export``   ProRes 4444 with alpha, PNG sequence, black and white matte.
"""
