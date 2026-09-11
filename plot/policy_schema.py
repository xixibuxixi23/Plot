"""Stable M4 capability/profile routing shared by data and model code."""

FAMILIES = ("language_builder", "villager", "combat")
PROFILES = (
    "language_builder",
    "villager_peaceful",
    "zombie_melee",
    "skeleton_swordsman",
    "villager_defender",
)
FAMILY_TO_ID = {name: index for index, name in enumerate(FAMILIES)}
PROFILE_TO_ID = {name: index for index, name in enumerate(PROFILES)}

FAMILY_PROFILES = {
    FAMILY_TO_ID["language_builder"]: {PROFILE_TO_ID["language_builder"]},
    FAMILY_TO_ID["villager"]: {PROFILE_TO_ID["villager_peaceful"]},
    FAMILY_TO_ID["combat"]: {
        PROFILE_TO_ID["zombie_melee"],
        PROFILE_TO_ID["skeleton_swordsman"],
        PROFILE_TO_ID["villager_defender"],
    },
}

__all__ = [
    "FAMILIES", "PROFILES", "FAMILY_TO_ID", "PROFILE_TO_ID", "FAMILY_PROFILES"
]
