"""
ARKit 52 blendshapes -> MetaHuman face rig 'ctrl_expressions_*' curves.

Single source of truth for the mapping, shared by:
  - src/server.py           (real-time sidecar for Unreal)
  - src/ue_import_csv.py    (offline CSV -> AnimSequence importer)

This module is intentionally pure Python (no numpy at import time) so it can
also be loaded inside Unreal Engine's embedded Python interpreter, which may
not have numpy available.
"""

# The 52 ARKit blendshape channel names in the exact column order the model
# and the LiveLink Face CSV export both use.
ARKIT_CURVE_NAMES = [
    "EyeBlinkLeft", "EyeLookDownLeft", "EyeLookInLeft", "EyeLookOutLeft",
    "EyeLookUpLeft", "EyeSquintLeft", "EyeWideLeft",
    "EyeBlinkRight", "EyeLookDownRight", "EyeLookInRight", "EyeLookOutRight",
    "EyeLookUpRight", "EyeSquintRight", "EyeWideRight",
    "JawForward", "JawRight", "JawLeft", "JawOpen",
    "MouthClose", "MouthFunnel", "MouthPucker", "MouthRight", "MouthLeft",
    "MouthSmileLeft", "MouthSmileRight", "MouthFrownLeft", "MouthFrownRight",
    "MouthDimpleLeft", "MouthDimpleRight", "MouthStretchLeft", "MouthStretchRight",
    "MouthRollLower", "MouthRollUpper", "MouthShrugLower", "MouthShrugUpper",
    "MouthPressLeft", "MouthPressRight", "MouthLowerDownLeft", "MouthLowerDownRight",
    "MouthUpperUpLeft", "MouthUpperUpRight",
    "BrowDownLeft", "BrowDownRight", "BrowInnerUp",
    "BrowOuterUpLeft", "BrowOuterUpRight",
    "CheekPuff", "CheekSquintLeft", "CheekSquintRight",
    "NoseSneerLeft", "NoseSneerRight", "TongueOut",
]
assert len(ARKIT_CURVE_NAMES) == 52


# ARKit -> MetaHuman mapping. Each source channel lists the MH control
# short-names it contributes to, with a weight (1.0 = pass-through). Six
# ARKit channels have no clean MH equivalent and are dropped silently:
#   CheekPuff, JawLeft, JawRight, MouthLowerDownLeft/Right, TongueOut.
ARKIT_TO_MH = {
    # Eyes
    "EyeBlinkLeft":      [("eyeblinkl",           1.0)],
    "EyeBlinkRight":     [("eyeblinkr",           1.0)],
    "EyeWideLeft":       [("eyewidenl",           1.0)],
    "EyeWideRight":      [("eyewidenr",           1.0)],
    "EyeSquintLeft":     [("eyesquintinnerl",     1.0)],
    "EyeSquintRight":    [("eyesquintinnerr",     1.0)],
    "EyeLookUpLeft":     [("eyelookupl",          1.0)],
    "EyeLookUpRight":    [("eyelookupr",          1.0)],
    "EyeLookDownLeft":   [("eyelookdownl",        1.0)],
    "EyeLookDownRight":  [("eyelookdownr",        1.0)],
    # ARKit "in/out" is relative to the nose; translate to absolute direction.
    # Left eye "in" = looking rightward; "out" = looking leftward.
    "EyeLookInLeft":     [("eyelookrightl",       1.0)],
    "EyeLookOutLeft":    [("eyelookleftl",        1.0)],
    "EyeLookInRight":    [("eyelookleftr",        1.0)],
    "EyeLookOutRight":   [("eyelookrightr",       1.0)],

    # Brows
    "BrowDownLeft":      [("browdownl",           1.0)],
    "BrowDownRight":     [("browdownr",           1.0)],
    "BrowInnerUp":       [("browraiseinl",        1.0), ("browraiseinr",    1.0)],
    "BrowOuterUpLeft":   [("browraiseouterl",     1.0)],
    "BrowOuterUpRight":  [("browraiseouterr",     1.0)],

    # Cheeks (CheekPuff has no MH equivalent in this control set)
    "CheekSquintLeft":   [("eyecheekraisel",      1.0)],
    "CheekSquintRight":  [("eyecheekraiser",      1.0)],

    # Nose
    "NoseSneerLeft":     [("nosewrinklel",        1.0)],
    "NoseSneerRight":    [("nosewrinkler",        1.0)],

    # Jaw (JawLeft/JawRight have no lateral-jaw control in this set)
    "JawOpen":           [("jawopen",             1.0)],
    "JawForward":        [("jawfwd",              1.0)],

    # Mouth sideways
    "MouthLeft":         [("mouthleft",           1.0)],
    "MouthRight":        [("mouthright",          1.0)],

    # Mouth corners
    "MouthSmileLeft":    [("mouthcornerpulll",    1.0)],
    "MouthSmileRight":   [("mouthcornerpullr",    1.0)],
    "MouthFrownLeft":    [("mouthcornerdepressl", 1.0)],
    "MouthFrownRight":   [("mouthcornerdepressr", 1.0)],

    # Mouth dimple / stretch
    "MouthDimpleLeft":   [("mouthdimplel",        1.0)],
    "MouthDimpleRight":  [("mouthdimpler",        1.0)],
    "MouthStretchLeft":  [("mouthstretchl",       1.0)],
    "MouthStretchRight": [("mouthstretchr",       1.0)],

    # Upper lip raise (sneer toward nose)
    "MouthUpperUpLeft":  [("mouthupperlipraisel", 1.0)],
    "MouthUpperUpRight": [("mouthupperlipraiser", 1.0)],

    # Lip roll (curl inward under teeth)
    "MouthRollUpper":    [("mouthupperlipbitel",  1.0), ("mouthupperlipbiter", 1.0)],
    "MouthRollLower":    [("mouthlowerlipbitel",  1.0), ("mouthlowerlipbiter", 1.0)],

    # Lip press (per-side mouth closure)
    "MouthPressLeft":    [("mouthlipspressl",     1.0)],
    "MouthPressRight":   [("mouthlipspressr",     1.0)],

    # Mouth pucker (tight forward purse)
    "MouthPucker":       [("mouthlipspurseul",    1.0), ("mouthlipspurseur",  1.0),
                          ("mouthlipspursedl",    1.0), ("mouthlipspursedr",  1.0)],
    # Mouth funnel (wider forward O - closest MH is "thick lips")
    "MouthFunnel":       [("mouthlipsthickul",    1.0), ("mouthlipsthickur",  1.0)],

    # Mouth close (lips meeting while jaw is open)
    "MouthClose":        [("mouthlipstogetherul", 1.0), ("mouthlipstogetherur", 1.0),
                          ("mouthlipstogetherdl", 1.0), ("mouthlipstogetherdr", 1.0)],

    # Mouth shrug (pressing upper/lower lip out)
    "MouthShrugUpper":   [("mouthpressul",        1.0), ("mouthpressur",      1.0)],
    "MouthShrugLower":   [("mouthpressdl",        1.0), ("mouthpressdr",      1.0)],
}


# Build reverse map once at import time: full MH curve name -> list of
# (arkit_index, weight) contributions. Pure Python, no numpy required.
_ARKIT_INDEX = {name: i for i, name in enumerate(ARKIT_CURVE_NAMES)}
MH_TO_SOURCES = {}
for _arkit_name, _targets in ARKIT_TO_MH.items():
    _src_idx = _ARKIT_INDEX[_arkit_name]
    for _mh_short, _w in _targets:
        _full = "ctrl_expressions_" + _mh_short
        MH_TO_SOURCES.setdefault(_full, []).append((_src_idx, _w))
del _arkit_name, _targets, _src_idx, _mh_short, _w, _full

# Sorted list of all MH curve names that the model writes (for stable ordering).
MH_CURVE_NAMES = sorted(MH_TO_SOURCES.keys())


def arkit_to_mh_curves(arkit_frames):
    """
    Convert an [N][>=52] ARKit curve block to a dict of MH curve name ->
    per-frame float list. `arkit_frames` can be a numpy array, a list of
    lists, or any object supporting [frame_idx][channel_idx] indexing.
    The return type is a plain dict of lists, JSON-serializable as-is.
    """
    n_frames = len(arkit_frames)
    out = {}
    for mh_name in MH_CURVE_NAMES:
        sources = MH_TO_SOURCES[mh_name]
        if len(sources) == 1:
            src_idx, weight = sources[0]
            out[mh_name] = [
                float(weight * arkit_frames[f][src_idx]) for f in range(n_frames)
            ]
        else:
            out[mh_name] = [
                float(sum(w * arkit_frames[f][i] for i, w in sources))
                for f in range(n_frames)
            ]
    return out
