"""Erreurs d'appel au modèle, communes aux trois juges (EXE-131).

Un appel qui échoue n'est pas une réponse illisible : la distinction importe
pour la conduite à tenir. `JudgeCallError` arrête le juge (critère 6) ;
`ClaudeRefusal` ne concerne qu'une paire et ne l'arrête pas (critères 4 et 5).
"""

from __future__ import annotations


class JudgeCallError(Exception):
    """L'appel au modèle a échoué (code de sortie, réseau, sortie illisible
    comme JSON). Le juge s'arrête : la paire en cours n'est pas enregistrée,
    elle sera jugée à la relance."""

    def __init__(self, message: str, stdout: str = "", stderr: str = ""):
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


class ClaudeRefusal(Exception):
    """Claude a refusé de répondre à cette paire (`stop_reason == "refusal"`
    dans la sortie de Claude Code). Le message est celui rendu par l'outil."""
