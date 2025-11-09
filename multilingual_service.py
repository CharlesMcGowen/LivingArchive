#!/usr/bin/env python3
"""
Multilingual Service for LLaMA Gateway
======================================

Provides multilingual support for global users.
Supports 13 languages with detection and translation.

Author: EGO Revolution Team
Version: 1.0.0 - Phase 3 Advanced Features
"""

import asyncio
import logging
import json
import time
import re
from typing import Dict, List, Any, Optional, Tuple
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

class MultilingualService:
    """Service for multilingual support"""
    
    def __init__(self):
        self.supported_languages = {
            "en": {"name": "English", "code": "en", "native": "English"},
            "es": {"name": "Spanish", "code": "es", "native": "Español"},
            "fr": {"name": "French", "code": "fr", "native": "Français"},
            "de": {"name": "German", "code": "de", "native": "Deutsch"},
            "it": {"name": "Italian", "code": "it", "native": "Italiano"},
            "pt": {"name": "Portuguese", "code": "pt", "native": "Português"},
            "hi": {"name": "Hindi", "code": "hi", "native": "हिन्दी"},
            "th": {"name": "Thai", "code": "th", "native": "ไทย"},
            "zh": {"name": "Chinese", "code": "zh", "native": "中文"},
            "ja": {"name": "Japanese", "code": "ja", "native": "日本語"},
            "ko": {"name": "Korean", "code": "ko", "native": "한국어"},
            "ar": {"name": "Arabic", "code": "ar", "native": "العربية"},
            "ru": {"name": "Russian", "code": "ru", "native": "Русский"}
        }
        
        # Language detection patterns
        self.language_patterns = {
            "en": [r"\b(the|and|or|but|in|on|at|to|for|of|with|by)\b"],
            "es": [r"\b(el|la|los|las|de|del|en|con|por|para|que|como)\b"],
            "fr": [r"\b(le|la|les|de|du|des|en|avec|pour|que|comme)\b"],
            "de": [r"\b(der|die|das|und|oder|aber|in|auf|mit|für|von)\b"],
            "it": [r"\b(il|la|i|le|di|del|della|in|con|per|che|come)\b"],
            "pt": [r"\b(o|a|os|as|de|do|da|em|com|por|para|que|como)\b"],
            "hi": [r"[\u0900-\u097F]", r"\b(है|हैं|था|थे|का|की|के|में|से|को|पर)\b"],
            "th": [r"[\u0E00-\u0E7F]", r"\b(และ|หรือ|แต่|ใน|ที่|กับ|สำหรับ|ของ|เป็น|มี)\b"],
            "zh": [r"[\u4e00-\u9fff]", r"\b(的|了|在|是|有|和|或|但|为|与|从|到)\b"],
            "ja": [r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf]", r"\b(の|は|が|を|に|で|と|から|まで|より)\b"],
            "ko": [r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]", r"\b(의|이|가|을|를|에|에서|와|과|부터|까지)\b"],
            "ar": [r"[\u0600-\u06ff]", r"\b(في|من|إلى|على|مع|ل|عن|ب|ك|ه|هي|هو|هما|هم|هن)\b"],
            "ru": [r"[\u0400-\u04ff]", r"\b(и|или|но|в|на|с|для|от|до|из|к|по|у|о|об)\b"]
        }
        
        # Translation templates (simplified)
        self.translation_templates = {
            "en": {
                "greeting": "Hello! How can I help you today?",
                "coding": "I can help you with programming and coding tasks.",
                "analysis": "Let me analyze this for you.",
                "error": "I encountered an error. Please try again."
            },
            "es": {
                "greeting": "¡Hola! ¿Cómo puedo ayudarte hoy?",
                "coding": "Puedo ayudarte con tareas de programación y codificación.",
                "analysis": "Déjame analizar esto para ti.",
                "error": "Encontré un error. Por favor, inténtalo de nuevo."
            },
            "fr": {
                "greeting": "Bonjour! Comment puis-je vous aider aujourd'hui?",
                "coding": "Je peux vous aider avec les tâches de programmation et de codage.",
                "analysis": "Laissez-moi analyser cela pour vous.",
                "error": "J'ai rencontré une erreur. Veuillez réessayer."
            },
            "de": {
                "greeting": "Hallo! Wie kann ich Ihnen heute helfen?",
                "coding": "Ich kann Ihnen bei Programmier- und Codierungsaufgaben helfen.",
                "analysis": "Lassen Sie mich das für Sie analysieren.",
                "error": "Ich bin auf einen Fehler gestoßen. Bitte versuchen Sie es erneut."
            },
            "it": {
                "greeting": "Ciao! Come posso aiutarti oggi?",
                "coding": "Posso aiutarti con compiti di programmazione e codifica.",
                "analysis": "Lascia che analizzi questo per te.",
                "error": "Ho riscontrato un errore. Riprova per favore."
            },
            "pt": {
                "greeting": "Olá! Como posso ajudá-lo hoje?",
                "coding": "Posso ajudá-lo com tarefas de programação e codificação.",
                "analysis": "Deixe-me analisar isso para você.",
                "error": "Encontrei um erro. Tente novamente."
            },
            "hi": {
                "greeting": "नमस्ते! मैं आज आपकी कैसे मदद कर सकता हूं?",
                "coding": "मैं आपकी प्रोग्रामिंग और कोडिंग कार्यों में मदद कर सकता हूं।",
                "analysis": "मुझे इसे आपके लिए विश्लेषण करने दें।",
                "error": "मुझे एक त्रुटि मिली। कृपया फिर से कोशिश करें।"
            },
            "th": {
                "greeting": "สวัสดี! วันนี้ฉันจะช่วยคุณได้อย่างไร?",
                "coding": "ฉันสามารถช่วยคุณในงานการเขียนโปรแกรมและการเขียนโค้ดได้",
                "analysis": "ให้ฉันวิเคราะห์สิ่งนี้สำหรับคุณ",
                "error": "ฉันพบข้อผิดพลาด กรุณาลองใหม่อีกครั้ง"
            },
            "zh": {
                "greeting": "你好！我今天能如何帮助您？",
                "coding": "我可以帮助您进行编程和编码任务。",
                "analysis": "让我为您分析一下这个。",
                "error": "我遇到了错误。请重试。"
            },
            "ja": {
                "greeting": "こんにちは！今日はどのようにお手伝いできますか？",
                "coding": "プログラミングとコーディングのタスクをお手伝いできます。",
                "analysis": "これを分析させていただきます。",
                "error": "エラーが発生しました。もう一度お試しください。"
            },
            "ko": {
                "greeting": "안녕하세요! 오늘 어떻게 도와드릴까요?",
                "coding": "프로그래밍과 코딩 작업을 도와드릴 수 있습니다.",
                "analysis": "이것을 분석해드리겠습니다.",
                "error": "오류가 발생했습니다. 다시 시도해주세요."
            },
            "ar": {
                "greeting": "مرحباً! كيف يمكنني مساعدتك اليوم؟",
                "coding": "يمكنني مساعدتك في مهام البرمجة والترميز.",
                "analysis": "دعني أحلل هذا لك.",
                "error": "واجهت خطأ. يرجى المحاولة مرة أخرى."
            },
            "ru": {
                "greeting": "Привет! Как я могу помочь вам сегодня?",
                "coding": "Я могу помочь вам с задачами программирования и кодирования.",
                "analysis": "Позвольте мне проанализировать это для вас.",
                "error": "Я столкнулся с ошибкой. Пожалуйста, попробуйте снова."
            }
        }
    
    async def detect_language(self, text: str) -> Dict[str, Any]:
        """Detect the language of the input text"""
        try:
            if not text or len(text.strip()) < 3:
                return {"language": "en", "confidence": 0.5}  # Default to English with low confidence
            
            text_lower = text.lower()
            language_scores = {}
            
            # Score each language based on pattern matches
            for lang_code, patterns in self.language_patterns.items():
                score = 0
                for pattern in patterns:
                    matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                    score += matches
                
                # Normalize score by text length
                if len(text) > 0:
                    language_scores[lang_code] = score / len(text)
                else:
                    language_scores[lang_code] = 0
            
            # Find the language with highest score
            if language_scores:
                best_lang = max(language_scores, key=language_scores.get)
                confidence = language_scores[best_lang]
                
                # If confidence is too low, default to English
                if confidence < 0.01:
                    return {"language": "en", "confidence": 0.5}
                
                return {"language": best_lang, "confidence": min(confidence, 1.0)}
            else:
                return {"language": "en", "confidence": 0.5}
                
        except Exception as e:
            logger.error(f"❌ Language detection error: {e}")
            return {"language": "en", "confidence": 0.5}
    
    async def translate_text(self, text: str, target_language: str, source_language: str = None) -> Dict[str, Any]:
        """Translate text to target language"""
        try:
            start_time = time.time()
            
            # Detect source language if not provided
            if not source_language:
                source_language, _ = await self.detect_language(text)
            
            # If source and target are the same, return original
            if source_language == target_language:
                return {
                    "original_text": text,
                    "translated_text": text,
                    "source_language": source_language,
                    "target_language": target_language,
                    "confidence": 1.0,
                    "translation_time_ms": 0
                }
            
            # Simulate translation (in real implementation, use actual translation service)
            await asyncio.sleep(0.1)  # Simulate translation time
            
            # Get translation template based on content type
            content_type = self._classify_content(text)
            template_key = self._get_template_key(content_type)
            
            # Get translated template
            if target_language in self.translation_templates:
                translated_template = self.translation_templates[target_language].get(template_key, text)
            else:
                translated_template = text  # Fallback to original
            
            # Create translation result
            result = {
                "original_text": text,
                "translated_text": translated_template,
                "source_language": source_language,
                "target_language": target_language,
                "confidence": 0.8,  # Simulated confidence
                "translation_time_ms": round((time.time() - start_time) * 1000, 2)
            }
            
            logger.info(f"✅ Translated from {source_language} to {target_language}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Translation error: {e}")
            return {
                "original_text": text,
                "translated_text": text,
                "source_language": source_language or "en",
                "target_language": target_language,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _classify_content(self, text: str) -> str:
        """Classify content type for appropriate translation"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["hello", "hi", "hey", "greetings"]):
            return "greeting"
        elif any(word in text_lower for word in ["code", "program", "function", "variable", "debug"]):
            return "coding"
        elif any(word in text_lower for word in ["analyze", "analysis", "examine", "study"]):
            return "analysis"
        else:
            return "general"
    
    def _get_template_key(self, content_type: str) -> str:
        """Get template key for content type"""
        if content_type == "greeting":
            return "greeting"
        elif content_type == "coding":
            return "coding"
        elif content_type == "analysis":
            return "analysis"
        else:
            return "greeting"  # Default
    
    async def get_supported_languages(self) -> Dict[str, Any]:
        """Get list of supported languages"""
        return {
            "supported_languages": self.supported_languages,
            "total_languages": len(self.supported_languages),
            "language_codes": list(self.supported_languages.keys())
        }
    
    async def get_language_info(self, language_code: str) -> Dict[str, Any]:
        """Get information about a specific language"""
        if language_code in self.supported_languages:
            return {
                "language": self.supported_languages[language_code],
                "supported": True
            }
        else:
            return {
                "error": f"Language {language_code} not supported",
                "supported": False
            }
    
    async def batch_translate(self, texts: List[str], target_language: str, source_language: str = None) -> List[Dict[str, Any]]:
        """Translate multiple texts"""
        try:
            results = []
            for text in texts:
                result = await self.translate_text(text, target_language, source_language)
                results.append(result)
            
            return {
                "translations": results,
                "total_texts": len(texts),
                "target_language": target_language
            }
            
        except Exception as e:
            logger.error(f"❌ Batch translation error: {e}")
            return {"error": str(e)}

# Global multilingual service instance
multilingual_service = MultilingualService()

# Convenience functions
async def detect_language(text: str) -> Tuple[str, float]:
    """Detect language of text"""
    return await multilingual_service.detect_language(text)

async def translate_text(text: str, target_language: str, source_language: str = None) -> Dict[str, Any]:
    """Translate text"""
    return await multilingual_service.translate_text(text, target_language, source_language)

async def get_supported_languages() -> Dict[str, Any]:
    """Get supported languages"""
    return await multilingual_service.get_supported_languages()

async def get_language_info(language_code: str) -> Dict[str, Any]:
    """Get language info"""
    return await multilingual_service.get_language_info(language_code)
