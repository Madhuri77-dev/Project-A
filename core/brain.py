"""Core LLM Brain for Project JARVIS.

Authored by Jayesh (Lead) with Pair A/B integration.
Supports multilingual reasoning (English, Hindi, Marathi) with short,
conversational replies optimized for text-to-speech.
"""

from __future__ import annotations

import datetime
import os
import re
import sys
from pathlib import Path
from typing import Final

# Reconfigure stdout for Windows console UTF-8 Devanagari support
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from google import genai
except ImportError:
    genai = None  # type: ignore

from voice.lang import detect_language

SYSTEM_INSTRUCTIONS: Final[str] = (
    "You are Jarvis, an advanced desktop AI assistant. "
    "Always reply in the same language as the user (English, Hindi, Marathi, or Hinglish). "
    "When the user speaks Hindi or Marathi, reply in Devanagari script so text-to-speech can pronounce it. "
    "Keep replies to 1-2 short, crisp, natural sentences. "
    "Be helpful, polite, and direct."
)


class JarvisBrain:
    """Jarvis decision and reasoning brain."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.client = None

        if self.api_key and genai is not None:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as exc:
                print(f"Notice: Gemini client init issue: {exc}")

    def _offline_fallback(self, message: str, lang: str) -> str:
        """Intelligent offline fallback responder for standard voice commands."""
        clean = message.lower().strip()
        now = datetime.datetime.now()

        # Date query
        if any(w in clean for w in ("date", "तारीख", "दिनांक", "aaj ka date", "aajchi tarikh")):
            date_str = now.strftime("%A, %d %B %Y")
            if lang == "mr":
                return f"आजची तारीख {now.strftime('%d/%m/%Y')} आहे."
            elif lang == "hi":
                return f"आज की तारीख {now.strftime('%d/%m/%Y')} है।"
            return f"Today is {date_str}."

        # Time query
        if any(w in clean for w in ("time", "वेळ", "समय", "किती वाजले", "kitne baje")):
            time_str = now.strftime("%I:%M %p")
            if lang == "mr":
                return f"आता {time_str} झाले आहेत."
            elif lang == "hi":
                return f"अभी समय {time_str} हुआ है।"
            return f"The current time is {time_str}."

        # Open Notepad / apps
        if any(w in clean for w in ("notepad", "नोटपॅड", "नोटपैड")):
            try:
                import subprocess
                subprocess.Popen("notepad.exe")
            except Exception:
                pass
            if lang == "mr":
                return "मी आपल्यासाठी नोटपॅड उघडले आहे."
            elif lang == "hi":
                return "मैंने आपके लिए नोटपैड खोल दिया है।"
            return "Opening Notepad for you right away."

        # RAM / Memory query
        if any(w in clean for w in ("ram", "memory", "मेमरी", "रॅम", "रैम")):
            try:
                import psutil
                mem = psutil.virtual_memory()
                pct = mem.percent
                free_gb = round(mem.available / (1024 ** 3), 1)
                if lang == "mr":
                    return f"आपली रॅम वापर {pct}% आहे, आणि {free_gb} GB मेमरी मोकळी आहे."
                elif lang == "hi":
                    return f"रैम उपयोग {pct}% है, और {free_gb} GB मेमोरी उपलब्ध है।"
                return f"Your RAM usage is at {pct}%, with {free_gb} gigabytes available."
            except Exception:
                pass

        # Greetings
        if any(w in clean for w in ("hello", "hi", "hey", "नमस्कार", "नमस्ते", "जार्विस", "jarvis")):
            if lang == "mr":
                return "नमस्कार! मी जार्विस आहे. मी आपली काय मदत करू शकतो?"
            elif lang == "hi":
                return "नमस्ते! मैं जार्विस हूँ। बताइए मैं आपकी क्या सहायता करूँ?"
            return "Hello! I am Jarvis. How can I assist you today?"

        # Default fallback
        if lang == "mr":
            return f"मी आपले म्हणणे समजलो: '{message}'. आदेश पूर्ण करण्यासाठी मी तयार आहे."
        elif lang == "hi":
            return f"मैंने आपकी बात समझ ली: '{message}'. आज्ञा अनुसार कार्य करने के लिए तैयार हूँ।"
        return f"Understood: '{message}'. Ready to assist you."

    def chat(self, message: str) -> str:
        """Process user input message and return Jarvis's response.

        Args:
            message: Spoken or typed user input text.

        Returns:
            Short conversational response text.
        """
        clean_msg = message.strip()
        if not clean_msg:
            return ""

        detected_lang = detect_language(clean_msg)

        # If Gemini client is available, generate live AI response
        if self.client:
            try:
                prompt = (
                    f"{SYSTEM_INSTRUCTIONS}\n"
                    f"User language: {detected_lang}\n"
                    f"User message: {clean_msg}\n"
                    f"Assistant response:"
                )
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as exc:
                print(f"Notice: Gemini call failed ({exc}). Using internal reasoning...")

        # Fallback to local intelligent assistant responder
        return self._offline_fallback(clean_msg, detected_lang)


# Global singleton instance
_default_brain: JarvisBrain | None = None


def get_brain() -> JarvisBrain:
    """Return the shared JarvisBrain instance."""
    global _default_brain
    if _default_brain is None:
        _default_brain = JarvisBrain()
    return _default_brain
