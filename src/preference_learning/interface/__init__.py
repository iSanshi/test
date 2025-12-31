"""
Interface package exposing the Tkinter UI and session controller.
"""

from .session import PreferenceSession, SessionMode, GroundTruthKind
from .ui_study import (
    AudioPreferenceStudyApp,
    main as launch_study_ui,
    user_main as launch_user_study_ui,
    test_main as launch_auto_test_ui,
)

__all__ = [
    "PreferenceSession",
    "SessionMode",
    "GroundTruthKind",
    "AudioPreferenceStudyApp",
    "launch_study_ui",
    "launch_user_study_ui",
    "launch_auto_test_ui",
]
