# EcoVision AI — multi-object waste identification and disposal assistant
# Uses: the user's existing 8-class Keras classifier + Google's free MediaPipe EfficientDet-Lite0 detector.
# Important: no custom waste detector is trained by this app. The COCO detector locates common objects;
# the supplied Keras model classifies each crop into one of the eight existing waste categories.

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import mediapipe as mp
import numpy as np
import requests
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont, ImageOps


APP_DIR = Path(__file__).resolve().parent
CLASSIFIER_PATH = APP_DIR / "keras_model.h5"
LABELS_PATH = APP_DIR / "labels.txt"
DETECTOR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite"
)
DETECTOR_MODEL_PATH = Path(tempfile.gettempdir()) / "ecovision_efficientdet_lite0.tflite"
MIN_CATEGORY_SCORE = 0.80
HIGH_MATCH_SCORE = 0.90
MAX_UPLOAD_MB = 15

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ecovision")

st.set_page_config(page_title="EcoVision AI", page_icon="♻️", layout="wide")

LANGUAGES = {'English': 'en', 'हिन्दी': 'hi', 'தமிழ்': 'ta', 'తెలుగు': 'te', 'ಕನ್ನಡ': 'kn', 'മലയാളം': 'ml'}
TEXT = {'en': {'analyzing': 'Analyzing image...',
        'bio': 'Biodegradability',
        'bio_no': 'Usually non-biodegradable',
        'bio_var': 'Depends on the material',
        'bio_yes': 'Usually biodegradable',
        'bot_name': 'PrithviPal',
        'category': 'Predicted category',
        'caution': 'General guidance only. Follow your municipality’s collection rules. Never burn, dump or '
                   'pour waste into drains.',
        'chat_empty': 'Try asking: How do I recycle plastic? or What is composting?',
        'chat_intro': 'Ask about waste sorting, recycling, composting, safe disposal and environmental '
                      'protection.',
        'chat_placeholder': 'Ask PrithviPal a question...',
        'chat_title': 'PrithviPal — Waste & Environment Assistant',
        'confidence': 'AI efficiency / match score',
        'ideas': 'Choose a safe disposal method',
        'identify': 'Identify waste',
        'intro': 'Upload a photo. PrithviPal will locate supported objects, classify each crop and suggest '
                 'safer disposal routes.',
        'language': 'Choose language',
        'materials': 'Disposal method',
        'model_note': 'The object detector recognises common COCO objects. It may miss unusual waste or '
                      'mistake an ordinary item for potential waste. Check the result before disposal.',
        'not_waste': 'No supported potential-waste object was confidently located. Try a clearer photo with '
                     'the item visible. People are ignored by the detector.',
        'rec_no': 'Usually not accepted in regular recycling',
        'rec_var': 'Depends on material, condition and local facilities',
        'rec_yes': 'Often recyclable if clean and accepted locally',
        'recycle': 'Recyclability',
        'result': 'Identification result',
        'send': 'Send',
        'tagline': 'Multi-object waste identification • Safe disposal • Environment guidance',
        'unknown': 'I can help with waste segregation, recycling, composting and safe disposal. Try asking '
                   'about plastic, paper, glass, metal, batteries, e-waste or biodegradable waste.',
        'upload': 'Upload a waste image',
        'uploaded': 'Uploaded image'},
 'hi': {'analyzing': 'तस्वीर का विश्लेषण हो रहा है...',
        'bio': 'जैव अपघटन',
        'bio_no': 'आमतौर पर जैव अपघटनीय नहीं',
        'bio_var': 'सामग्री पर निर्भर करता है',
        'bio_yes': 'आमतौर पर जैव अपघटनीय',
        'bot_name': 'Eco FAQ',
        'category': 'अनुमानित श्रेणी',
        'caution': 'ये सामान्य सुझाव हैं। स्थानीय नियम जाँचें। खतरनाक या दूषित वस्तुओं से DIY न करें।',
        'chat_empty': 'पूछें: प्लास्टिक को कैसे रीसायकल करें?',
        'chat_intro': 'कचरा छँटाई, रीसाइक्लिंग, कम्पोस्ट या पुन: उपयोग के बारे में पूछें।',
        'chat_placeholder': 'अपना सवाल लिखें...',
        'chat_title': 'Eco FAQ चैटबॉट',
        'confidence': 'AI मिलान स्कोर',
        'ideas': 'पुन: उपयोग / निपटान के सुझाव',
        'identify': 'कचरे की पहचान करें',
        'intro': 'कचरे की तस्वीर अपलोड करें और उसकी श्रेणी, निपटान मार्गदर्शन तथा पुन: उपयोग के सुझाव पाएँ।',
        'language': 'भाषा चुनें',
        'materials': 'सरल सुझाव',
        'model_note': 'मॉडल सीमा: यह पूरी तस्वीर के लिए केवल एक श्रेणी बताता है। कई वस्तुओं की अलग पहचान के '
                      'लिए अतिरिक्त object-detection मॉडल चाहिए।',
        'not_waste': 'मॉडल किसी समर्थित कचरा श्रेणी की स्पष्ट पहचान नहीं कर पाया। एक वस्तु की साफ़, नज़दीकी '
                     'तस्वीर लें।',
        'rec_no': 'आमतौर पर सामान्य रीसाइक्लिंग में स्वीकार नहीं होता',
        'rec_var': 'सामग्री, स्थिति और स्थानीय सुविधा पर निर्भर',
        'rec_yes': 'साफ़ हो और स्थानीय केंद्र स्वीकार करे तो अक्सर रीसायकल हो सकता है',
        'recycle': 'रीसाइक्लिंग',
        'result': 'पहचान परिणाम',
        'send': 'भेजें',
        'tagline': 'कचरा पहचान • पुन: उपयोग के सुझाव • Eco FAQ',
        'unknown': 'मैं कचरा छँटाई, रीसाइक्लिंग, कम्पोस्ट और पुन: उपयोग के सामान्य सवालों का जवाब दे सकता '
                   'हूँ।',
        'upload': 'कचरे की तस्वीर अपलोड करें',
        'uploaded': 'अपलोड की गई तस्वीर'},
 'kn': {'analyzing': 'ಚಿತ್ರವನ್ನು ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ...',
        'bio': 'ಜೈವಿಕವಾಗಿ ಕೊಳೆಯುವಿಕೆ',
        'bio_no': 'ಸಾಮಾನ್ಯವಾಗಿ ಜೈವಿಕವಾಗಿ ಕೊಳೆಯುವುದಿಲ್ಲ',
        'bio_var': 'ವಸ್ತುವಿನ ಮೇಲೆ ಅವಲಂಬಿತವಾಗಿದೆ',
        'bio_yes': 'ಸಾಮಾನ್ಯವಾಗಿ ಜೈವಿಕವಾಗಿ ಕೊಳೆಯುತ್ತದೆ',
        'bot_name': 'Eco FAQ',
        'category': 'ಅಂದಾಜು ವರ್ಗ',
        'caution': 'ಇವು ಸಾಮಾನ್ಯ ಸಲಹೆಗಳು. ಸ್ಥಳೀಯ ನಿಯಮಗಳನ್ನು ಪರಿಶೀಲಿಸಿ. ಅಪಾಯಕಾರಿ ವಸ್ತುಗಳನ್ನು DIYಗೆ ಬಳಸಬೇಡಿ.',
        'chat_empty': 'ಉದಾಹರಣೆ: ಪ್ಲಾಸ್ಟಿಕ್ ಅನ್ನು ಹೇಗೆ ಮರುಬಳಕೆ ಮಾಡುವುದು?',
        'chat_intro': 'ತ್ಯಾಜ್ಯ ವಿಂಗಡಣೆ, ಮರುಬಳಕೆ, ಕಾಂಪೋಸ್ಟ್ ಅಥವಾ ಮರುಉಪಯೋಗದ ಬಗ್ಗೆ ಕೇಳಿ.',
        'chat_placeholder': 'ನಿಮ್ಮ ಪ್ರಶ್ನೆ ಬರೆಯಿರಿ...',
        'chat_title': 'Eco FAQ ಚಾಟ್\u200cಬಾಟ್',
        'confidence': 'AI ಹೊಂದಾಣಿಕೆ ಸ್ಕೋರ್',
        'ideas': 'ಮರುಬಳಕೆ / ವಿಲೇವಾರಿ ಆಲೋಚನೆಗಳು',
        'identify': 'ತ್ಯಾಜ್ಯ ಗುರುತಿಸಿ',
        'intro': 'ತ್ಯಾಜ್ಯದ ಚಿತ್ರವನ್ನು ಅಪ್\u200cಲೋಡ್ ಮಾಡಿ ಅದರ ವರ್ಗ, ನಿರ್ವಹಣೆ ಮಾರ್ಗದರ್ಶನ ಮತ್ತು ಮರುಬಳಕೆ '
                 'ಆಲೋಚನೆಗಳನ್ನು ಪಡೆಯಿರಿ.',
        'language': 'ಭಾಷೆ ಆಯ್ಕೆಮಾಡಿ',
        'materials': 'ಸರಳ ಆಲೋಚನೆ',
        'model_note': 'ಮಾದರಿಯ ಮಿತಿ: ಇದು ಸಂಪೂರ್ಣ ಚಿತ್ರಕ್ಕೆ ಒಂದು ವರ್ಗವನ್ನು ಮಾತ್ರ ಊಹಿಸುತ್ತದೆ. ಹಲವು ವಸ್ತುಗಳನ್ನು '
                      'ಪ್ರತ್ಯೇಕವಾಗಿ ಗುರುತಿಸಲು ಹೆಚ್ಚುವರಿ object-detection ಮಾದರಿ ಬೇಕು.',
        'not_waste': 'ಬೆಂಬಲಿತ ತ್ಯಾಜ್ಯ ವರ್ಗವನ್ನು ಮಾದರಿ ಸ್ಪಷ್ಟವಾಗಿ ಗುರುತಿಸಲಿಲ್ಲ. ಒಂದು ವಸ್ತುವಿನ ಸ್ಪಷ್ಟ ಹತ್ತಿರದ '
                     'ಚಿತ್ರವನ್ನು ಪ್ರಯತ್ನಿಸಿ.',
        'rec_no': 'ಸಾಮಾನ್ಯ ಮರುಬಳಕೆಯಲ್ಲಿ ಸಾಮಾನ್ಯವಾಗಿ ಸ್ವೀಕರಿಸುವುದಿಲ್ಲ',
        'rec_var': 'ವಸ್ತು, ಸ್ಥಿತಿ ಮತ್ತು ಸ್ಥಳೀಯ ಸೌಲಭ್ಯಗಳ ಮೇಲೆ ಅವಲಂಬಿತ',
        'rec_yes': 'ಸ್ವಚ್ಛವಾಗಿದ್ದು ಸ್ಥಳೀಯ ಕೇಂದ್ರ ಸ್ವೀಕರಿಸಿದರೆ ಮರುಬಳಕೆ ಸಾಧ್ಯ',
        'recycle': 'ಮರುಬಳಕೆ',
        'result': 'ಗುರುತಿಸುವಿಕೆಯ ಫಲಿತಾಂಶ',
        'send': 'ಕಳುಹಿಸಿ',
        'tagline': 'ತ್ಯಾಜ್ಯ ಗುರುತಿಸುವಿಕೆ • ಮರುಬಳಕೆ ಆಲೋಚನೆಗಳು • Eco FAQ',
        'unknown': 'ತ್ಯಾಜ್ಯ ವಿಂಗಡಣೆ, ಮರುಬಳಕೆ, ಕಾಂಪೋಸ್ಟ್ ಮತ್ತು ಮರುಉಪಯೋಗದ ಸಾಮಾನ್ಯ ಪ್ರಶ್ನೆಗಳಿಗೆ ಉತ್ತರಿಸಬಹುದು.',
        'upload': 'ತ್ಯಾಜ್ಯದ ಚಿತ್ರ ಅಪ್\u200cಲೋಡ್ ಮಾಡಿ',
        'uploaded': 'ಅಪ್\u200cಲೋಡ್ ಮಾಡಿದ ಚಿತ್ರ'},
 'ml': {'analyzing': 'ചിത്രം പരിശോധിക്കുന്നു...',
        'bio': 'ജൈവ വിഘടനം',
        'bio_no': 'സാധാരണയായി ജൈവമായി വിഘടിക്കില്ല',
        'bio_var': 'വസ്തുവിനെ ആശ്രയിച്ചിരിക്കുന്നു',
        'bio_yes': 'സാധാരണയായി ജൈവമായി വിഘടിക്കും',
        'bot_name': 'Eco FAQ',
        'category': 'കണക്കാക്കിയ വിഭാഗം',
        'caution': 'ഇവ പൊതുവായ നിർദ്ദേശങ്ങളാണ്. പ്രാദേശിക നിയമങ്ങൾ പരിശോധിക്കുക. അപകടകരമായ വസ്തുക്കൾ '
                   'DIY-യ്ക്ക് ഉപയോഗിക്കരുത്.',
        'chat_empty': 'ഉദാഹരണം: പ്ലാസ്റ്റിക് എങ്ങനെ റീസൈക്കിൾ ചെയ്യാം?',
        'chat_intro': 'മാലിന്യ വേർതിരിക്കൽ, റീസൈക്ലിംഗ്, കമ്പോസ്റ്റിംഗ്, പുനരുപയോഗം എന്നിവ ചോദിക്കാം.',
        'chat_placeholder': 'ചോദ്യം ടൈപ്പ് ചെയ്യുക...',
        'chat_title': 'Eco FAQ ചാറ്റ്ബോട്ട്',
        'confidence': 'മോഡലിന്റെ വിശ്വാസനില',
        'ideas': 'പുനരുപയോഗം / നീക്കംചെയ്യൽ ആശയങ്ങൾ',
        'identify': 'മാലിന്യം തിരിച്ചറിയുക',
        'intro': 'മാലിന്യത്തിന്റെ ചിത്രം അപ്\u200cലോഡ് ചെയ്ത് വിഭാഗവും കൈകാര്യം ചെയ്യാനുള്ള നിർദ്ദേശങ്ങളും '
                 'പുനരുപയോഗ ആശയങ്ങളും അറിയുക.',
        'language': 'ഭാഷ തിരഞ്ഞെടുക്കുക',
        'materials': 'ലളിതമായ ആശയം',
        'model_note': 'മോഡലിന്റെ പരിമിതി: മുഴുവൻ ചിത്രത്തിനും ഒരു വിഭാഗം മാത്രമാണ് പ്രവചിക്കുന്നത്. '
                      'ഒന്നിലധികം വസ്തുക്കൾ വേർതിരിച്ച് കണ്ടെത്താൻ മറ്റൊരു object-detection മോഡൽ വേണം.',
        'not_waste': 'പിന്തുണയ്ക്കുന്ന മാലിന്യ വിഭാഗം മോഡലിന് വ്യക്തമായി തിരിച്ചറിയാനായില്ല. ഒരു വസ്തുവിന്റെ '
                     'വ്യക്തമായ അടുത്ത ചിത്രം ഉപയോഗിക്കുക.',
        'rec_no': 'സാധാരണ റീസൈക്ലിംഗിൽ സാധാരണയായി സ്വീകരിക്കില്ല',
        'rec_var': 'വസ്തു, അവസ്ഥ, പ്രാദേശിക സൗകര്യങ്ങൾ എന്നിവയെ ആശ്രയിക്കുന്നു',
        'rec_yes': 'വൃത്തിയുള്ളതും പ്രാദേശിക കേന്ദ്രം സ്വീകരിക്കുന്നതുമെങ്കിൽ റീസൈക്കിൾ ചെയ്യാം',
        'recycle': 'റീസൈക്ലിംഗ്',
        'result': 'തിരിച്ചറിയൽ ഫലം',
        'send': 'അയയ്ക്കുക',
        'tagline': 'മാലിന്യ തിരിച്ചറിയൽ • പുനരുപയോഗ ആശയങ്ങൾ • Eco FAQ',
        'unknown': 'മാലിന്യ വേർതിരിക്കൽ, റീസൈക്ലിംഗ്, കമ്പോസ്റ്റിംഗ്, പുനരുപയോഗം എന്നിവയെക്കുറിച്ചുള്ള '
                   'പൊതുവായ ചോദ്യങ്ങൾക്ക് മറുപടി നൽകാം.',
        'upload': 'മാലിന്യ ചിത്രം അപ്\u200cലോഡ് ചെയ്യുക',
        'uploaded': 'അപ്\u200cലോഡ് ചെയ്ത ചിത്രം'},
 'ta': {'analyzing': 'படம் ஆய்வு செய்யப்படுகிறது...',
        'bio': 'மக்கும் தன்மை',
        'bio_no': 'பொதுவாக மக்காதது',
        'bio_var': 'பொருளைப் பொறுத்தது',
        'bio_yes': 'பொதுவாக மக்கும்',
        'bot_name': 'Eco FAQ',
        'category': 'கணிக்கப்பட்ட வகை',
        'caution': 'இவை பொதுவான ஆலோசனைகள். உள்ளூர் விதிகளைச் சரிபார்க்கவும். ஆபத்தான பொருட்களை DIY-க்கு '
                   'பயன்படுத்த வேண்டாம்.',
        'chat_empty': 'உதாரணம்: பிளாஸ்டிக்கை எப்படி மறுசுழற்சி செய்வது?',
        'chat_intro': 'கழிவு பிரித்தல், மறுசுழற்சி, உரமிடுதல் அல்லது மறுபயன்பாடு பற்றி கேளுங்கள்.',
        'chat_placeholder': 'உங்கள் கேள்வியை உள்ளிடவும்...',
        'chat_title': 'Eco FAQ உரையாடல்',
        'confidence': 'மாதிரியின் நம்பிக்கை',
        'ideas': 'மறுபயன்பாடு / அகற்றும் யோசனைகள்',
        'identify': 'கழிவை அடையாளம் காண்க',
        'intro': 'கழிவுப் படத்தைப் பதிவேற்றி அதன் வகை, கையாளும் வழிகாட்டுதல் மற்றும் மறுபயன்பாட்டு '
                 'யோசனைகளைப் பெறுங்கள்.',
        'language': 'மொழியைத் தேர்ந்தெடுக்கவும்',
        'materials': 'எளிய யோசனை',
        'model_note': 'மாதிரி வரம்பு: முழுப் படத்திற்கும் ஒரு வகையை மட்டுமே கணிக்கிறது. பல பொருட்களைத் '
                      'தனித்தனியாகக் கண்டறிய கூடுதல் object-detection மாதிரி தேவை.',
        'not_waste': 'ஆதரிக்கப்படும் கழிவு வகையை மாதிரி தெளிவாகக் கண்டறியவில்லை. ஒரு பொருளின் தெளிவான '
                     'நெருக்கமான படத்தை முயற்சிக்கவும்.',
        'rec_no': 'பொதுவாக வழக்கமான மறுசுழற்சியில் ஏற்கப்படாது',
        'rec_var': 'பொருள், நிலை மற்றும் உள்ளூர் வசதிகளைப் பொறுத்தது',
        'rec_yes': 'சுத்தமாகவும் உள்ளூர் மையம் ஏற்றுக்கொண்டாலும் மறுசுழற்சி செய்யலாம்',
        'recycle': 'மறுசுழற்சி',
        'result': 'அடையாள முடிவு',
        'send': 'அனுப்பு',
        'tagline': 'கழிவு அடையாளம் • மறுபயன்பாட்டு யோசனைகள் • Eco FAQ',
        'unknown': 'கழிவு பிரித்தல், மறுசுழற்சி, உரமிடுதல் மற்றும் மறுபயன்பாடு குறித்த பொதுவான கேள்விகளுக்கு '
                   'பதிலளிக்க முடியும்.',
        'upload': 'கழிவுப் படத்தைப் பதிவேற்றவும்',
        'uploaded': 'பதிவேற்றிய படம்'},
 'te': {'analyzing': 'చిత్రాన్ని పరిశీలిస్తోంది...',
        'bio': 'జీవ విచ్ఛిన్నత',
        'bio_no': 'సాధారణంగా జీవ విచ్ఛిన్నం కాదు',
        'bio_var': 'పదార్థంపై ఆధారపడి ఉంటుంది',
        'bio_yes': 'సాధారణంగా జీవ విచ్ఛిన్నమవుతుంది',
        'bot_name': 'PrithviPal',
        'category': 'అంచనా వేసిన వర్గం',
        'caution': 'ఇవి సాధారణ సూచనలు మాత్రమే. మీ మున్సిపాలిటీ నియమాలు పాటించండి. చెత్తను కాల్చవద్దు, '
                   'ఎక్కడపడితే అక్కడ వేయవద్దు, డ్రైన్లలో పోయవద్దు.',
        'chat_empty': 'ఉదాహరణ: ప్లాస్టిక్\u200cను ఎలా రీసైకిల్ చేయాలి?',
        'chat_intro': 'వ్యర్థాల వేర్పాటు, రీసైక్లింగ్, కంపోస్టింగ్, సురక్షిత పారవేత, పర్యావరణ పరిరక్షణ '
                      'గురించి అడగండి.',
        'chat_placeholder': 'PrithviPal\u200cను ప్రశ్న అడగండి...',
        'chat_title': 'PrithviPal — వ్యర్థాలు & పర్యావరణ సహాయకుడు',
        'confidence': 'AI సామర్థ్య / సరిపోలిక స్కోర్',
        'ideas': 'సురక్షిత పారవేత పద్ధతిని ఎంచుకోండి',
        'identify': 'వ్యర్థాన్ని గుర్తించండి',
        'intro': 'ఫోటోను అప్\u200cలోడ్ చేయండి. PrithviPal గుర్తించగల వస్తువులను విడిగా గుర్తించి, ప్రతి '
                 'వస్తువుకు వర్గం మరియు సురక్షిత పారవేత సూచనలు అందిస్తుంది.',
        'language': 'భాషను ఎంచుకోండి',
        'materials': 'పారవేత పద్ధతి',
        'model_note': 'ఆబ్జెక్ట్ డిటెక్టర్ సాధారణ COCO వస్తువులను గుర్తిస్తుంది. ప్రత్యేకమైన వ్యర్థాలను '
                      'గుర్తించకపోవచ్చు లేదా సాధారణ వస్తువును వ్యర్థంగా పొరబడవచ్చు. పారవేయడానికి ముందు '
                      'ఫలితాన్ని పరిశీలించండి.',
        'not_waste': 'మద్దతు ఉన్న వ్యర్థ వస్తువు స్పష్టంగా గుర్తించబడలేదు. వస్తువు స్పష్టంగా కనిపించే ఫోటోను '
                     'ప్రయత్నించండి. వ్యక్తులను డిటెక్టర్ పట్టించుకోదు.',
        'rec_no': 'సాధారణ రీసైక్లింగ్\u200cలో సాధారణంగా అంగీకరించరు',
        'rec_var': 'పదార్థం, స్థితి, స్థానిక సదుపాయాలపై ఆధారపడి ఉంటుంది',
        'rec_yes': 'శుభ్రంగా ఉంటే, స్థానికంగా అంగీకరిస్తే రీసైకిల్ చేయవచ్చు',
        'recycle': 'రీసైక్లింగ్',
        'result': 'గుర్తింపు ఫలితం',
        'send': 'పంపండి',
        'tagline': 'బహుళ వస్తువుల వ్యర్థ గుర్తింపు • సురక్షిత పారవేత • పర్యావరణ సూచనలు',
        'unknown': 'వ్యర్థాల వేర్పాటు, రీసైక్లింగ్, కంపోస్టింగ్, సురక్షిత పారవేత గురించి సహాయం చేయగలను. '
                   'ప్లాస్టిక్, కాగితం, గాజు, లోహం, బ్యాటరీలు లేదా ఈ-వ్యర్థాల గురించి అడగండి.',
        'upload': 'వ్యర్థాల ఫోటోను అప్\u200cలోడ్ చేయండి',
        'uploaded': 'అప్\u200cలోడ్ చేసిన చిత్రం'}}
CATEGORY_NAMES = {'E-Waste & Others': {'hi': 'ई-कचरा और अन्य',
                      'kn': 'ಇ-ತ್ಯಾಜ್ಯ ಮತ್ತು ಇತರೆ',
                      'ml': 'ഇ-മാലിന്യം & മറ്റുള്ളവ',
                      'ta': 'மின்னணுக் கழிவு மற்றும் பிற',
                      'te': 'ఈ-వ్యర్థాలు & ఇతరాలు'},
 'Glass': {'hi': 'काँच', 'kn': 'ಗಾಜು', 'ml': 'ഗ്ലാസ്', 'ta': 'கண்ணாடி', 'te': 'గాజు'},
 'Hazardous Waste': {'hi': 'खतरनाक कचरा',
                     'kn': 'ಅಪಾಯಕಾರಿ ತ್ಯಾಜ್ಯ',
                     'ml': 'അപകടകരമായ മാലിന്യം',
                     'ta': 'அபாயகரமான கழிவு',
                     'te': 'ప్రమాదకర వ్యర్థాలు'},
 'Metal & Scrap': {'hi': 'धातु और कबाड़',
                   'kn': 'ಲೋಹ ಮತ್ತು ಸ್ಕ್ರ್ಯಾಪ್',
                   'ml': 'ലോഹം & സ്ക്രാപ്പ്',
                   'ta': 'உலோகம் மற்றும் பழைய பொருட்கள்',
                   'te': 'లోహం & స్క్రాప్'},
 'Organic Waste': {'hi': 'जैविक कचरा',
                   'kn': 'ಸಾವಯವ ತ್ಯಾಜ್ಯ',
                   'ml': 'ജൈവ മാലിന്യം',
                   'ta': 'கரிமக் கழிவு',
                   'te': 'సేంద్రీయ వ్యర్థాలు'},
 'Paper & Cardboard': {'hi': 'कागज़ और कार्डबोर्ड',
                       'kn': 'ಕಾಗದ ಮತ್ತು ಕಾರ್ಡ್\u200cಬೋರ್ಡ್',
                       'ml': 'പേപ്പർ & കാർഡ്ബോർഡ്',
                       'ta': 'காகிதம் மற்றும் அட்டை',
                       'te': 'కాగితం & కార్డ్\u200cబోర్డ్'},
 'Plastic': {'hi': 'प्लास्टिक',
             'kn': 'ಪ್ಲಾಸ್ಟಿಕ್',
             'ml': 'പ്ലാസ്റ്റിക്',
             'ta': 'பிளாஸ்டிக்',
             'te': 'ప్లాస్టిక్'},
 'Textile & Fabric Waste': {'hi': 'कपड़ा',
                            'kn': 'ಜವಳಿ ಮತ್ತು ಬಟ್ಟೆ',
                            'ml': 'തുണി',
                            'ta': 'துணி',
                            'te': 'వస్త్రాలు & ఫాబ్రిక్'}}

# The detector uses general object labels. Person is deliberately excluded.
# Electronics are shown for confirmation because a working classroom computer is not automatically e-waste.
WASTE_CANDIDATE_LABELS = {
    "bottle", "cup", "wine glass", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "book", "scissors",
    "cell phone", "laptop", "mouse", "keyboard", "tv", "remote", "microwave", "oven",
    "toaster", "refrigerator", "vase", "backpack", "handbag", "suitcase", "umbrella",
    "teddy bear", "toothbrush",
}
ELECTRONIC_LABELS = {
    "cell phone", "laptop", "mouse", "keyboard", "tv", "remote", "microwave",
    "oven", "toaster", "refrigerator",
}

CATEGORY_INFO = {
    "Plastic": {"bio": "Usually non-biodegradable", "recycle": "Depends on plastic type, cleanliness and local facilities."},
    "Paper & Cardboard": {"bio": "Usually biodegradable when untreated", "recycle": "Often recyclable when clean and dry."},
    "Organic Waste": {"bio": "Usually biodegradable", "recycle": "Usually handled through composting or wet-waste collection."},
    "Glass": {"bio": "Non-biodegradable", "recycle": "Often recyclable where glass collection is available."},
    "Metal & Scrap": {"bio": "Non-biodegradable", "recycle": "Often recyclable through an authorised scrap or dry-waste collector."},
    "Hazardous Waste": {"bio": "Depends on the material; treat as hazardous", "recycle": "Do not place in regular recycling. Use an authorised collection route."},
    "Textile & Fabric Waste": {"bio": "Depends on whether fibres are natural or synthetic", "recycle": "Donation or textile recycling may be available for suitable items."},
    "E-Waste & Others": {"bio": "Generally non-biodegradable", "recycle": "Use an authorised e-waste collection or producer take-back route."},
}

# These are illustrative category-based handling-priority indices (0–100), NOT measured damage percentages.
# Real effects depend on composition, quantity, condition, exposure and local environment.
RISK_INDEX = {
    "Plastic": {"environment": 75, "wildlife": 85, "people": 45},
    "Paper & Cardboard": {"environment": 25, "wildlife": 15, "people": 10},
    "Organic Waste": {"environment": 30, "wildlife": 25, "people": 35},
    "Glass": {"environment": 35, "wildlife": 55, "people": 55},
    "Metal & Scrap": {"environment": 50, "wildlife": 50, "people": 55},
    "Hazardous Waste": {"environment": 95, "wildlife": 90, "people": 95},
    "Textile & Fabric Waste": {"environment": 55, "wildlife": 60, "people": 40},
    "E-Waste & Others": {"environment": 90, "wildlife": 85, "people": 90},
}

# Three disposal routes per category. Suggestions avoid burning, dumping, dismantling or unsafe handling.
DISPOSAL_GUIDES = {
    "Plastic": [
        {"title": "Local dry-waste collection / recycler", "title_te": "స్థానిక పొడి వ్యర్థాల సేకరణ / రీసైక్లర్", "summary": "For clean, accepted plastic containers and bottles.", "summary_te": "శుభ్రంగా ఉన్న, కేంద్రం అంగీకరించే ప్లాస్టిక్ బాటిళ్లు లేదా డబ్బాల కోసం.", "steps": ["Keep the item separate from food and wet waste.", "Empty it and follow the collection centre's sorting rules.", "Hand it to the municipal dry-waste collector or an authorised recycler."], "steps_te": ["ఆహార, తడి వ్యర్థాల నుంచి వేరుగా ఉంచండి.", "లోపల ఖాళీ చేసి సేకరణ కేంద్రం సూచించిన విధంగా వేరు చేయండి.", "మున్సిపల్ పొడి వ్యర్థాల సేకరణకు లేదా అనుమతిపొందిన రీసైక్లర్‌కు ఇవ్వండి."]},
        {"title": "Producer take-back / deposit scheme", "title_te": "తయారీదారు తిరిగి స్వీకరణ పథకం", "summary": "Use a brand or store take-back route when one is available locally.", "summary_te": "మీ ప్రాంతంలో బ్రాండ్ లేదా దుకాణం తిరిగి స్వీకరణ పథకం ఉంటే దాన్ని ఉపయోగించండి.", "steps": ["Check whether the brand or store accepts this packaging.", "Keep the packaging separate and follow its return instructions.", "Use the official collection point; availability differs by area."], "steps_te": ["ఆ బ్రాండ్ లేదా దుకాణం ఈ ప్యాకింగ్‌ను స్వీకరిస్తుందో చూడండి.", "ప్యాకింగ్‌ను వేరుగా ఉంచి వారి సూచనలు పాటించండి.", "అధికారిక సేకరణ కేంద్రాన్ని ఉపయోగించండి; సదుపాయం ప్రాంతాన్ని బట్టి మారుతుంది."]},
        {"title": "Designated residual-waste collection", "title_te": "నిర్దేశిత మిగిలిన వ్యర్థాల సేకరణ", "summary": "For multilayer, dirty or plastic items not accepted by local recycling.", "summary_te": "బహుళ పొరల ప్యాకింగ్, మురికిగా ఉన్న లేదా రీసైక్లింగ్‌కు అంగీకరించని ప్లాస్టిక్ కోసం.", "steps": ["Do not mix it with clean recyclable plastic.", "Keep it in the designated residual-waste stream.", "Follow municipal collection guidance. Never burn or litter it."], "steps_te": ["శుభ్రమైన రీసైకిల్ ప్లాస్టిక్‌తో కలపవద్దు.", "నిర్దేశించిన మిగిలిన వ్యర్థాల విభాగంలో ఉంచండి.", "మున్సిపల్ సూచనలు పాటించండి. కాల్చవద్దు, ఎక్కడపడితే అక్కడ వేయవద్దు."]},
    ],
    "Paper & Cardboard": [
        {"title": "Paper recycling collection", "title_te": "పేపర్ రీసైక్లింగ్ సేకరణ", "summary": "Best for clean, dry paper and cardboard.", "summary_te": "శుభ్రంగా, పొడిగా ఉన్న కాగితం మరియు కార్డ్‌బోర్డ్‌కు అనుకూలం.", "steps": ["Keep paper dry and separate from food or oil.", "Flatten cardboard if local rules request it.", "Give it to a paper collector or an accepted recycling centre."], "steps_te": ["కాగితాన్ని పొడిగా ఉంచి ఆహారం లేదా నూనె నుంచి వేరు చేయండి.", "స్థానిక నియమాలు చెబితే కార్డ్‌బోర్డ్‌ను మడవండి.", "పేపర్ సేకరణదారునికి లేదా అంగీకరించిన రీసైక్లింగ్ కేంద్రానికి ఇవ్వండి."]},
        {"title": "Municipal dry-waste collection", "title_te": "మున్సిపల్ పొడి వ్యర్థాల సేకరణ", "summary": "Use the local dry-waste stream for sorted paper products.", "summary_te": "వేరు చేసిన కాగితపు వస్తువులను స్థానిక పొడి వ్యర్థాల సేకరణకు ఇవ్వండి.", "steps": ["Separate paper and cardboard from wet waste.", "Keep the bundle clean and dry.", "Hand it to the municipal dry-waste collector."], "steps_te": ["కాగితం, కార్డ్‌బోర్డ్‌ను తడి వ్యర్థాల నుంచి వేరు చేయండి.", "కట్టను శుభ్రంగా, పొడిగా ఉంచండి.", "మున్సిపల్ పొడి వ్యర్థాల సేకరణదారునికి ఇవ్వండి."]},
        {"title": "Soiled-paper local guidance", "title_te": "మురికిగా ఉన్న కాగితానికి స్థానిక సూచనలు", "summary": "Food- or oil-soiled paper may not belong in paper recycling.", "summary_te": "ఆహారం లేదా నూనె అంటిన కాగితాన్ని పేపర్ రీసైక్లింగ్‌లో అంగీకరించకపోవచ్చు.", "steps": ["Keep heavily soiled paper separate from clean paper.", "Check whether your local wet-waste service accepts it for composting.", "Otherwise use the local residual-waste route."], "steps_te": ["ఎక్కువగా మురికిగా ఉన్న కాగితాన్ని శుభ్రమైన కాగితం నుంచి వేరు చేయండి.", "స్థానిక తడి వ్యర్థాల సేవ దాన్ని కంపోస్టింగ్‌కు అంగీకరిస్తుందో చూడండి.", "అంగీకరించకపోతే స్థానిక మిగిలిన వ్యర్థాల మార్గాన్ని అనుసరించండి."]},
    ],
    "Organic Waste": [
        {"title": "Municipal wet-waste collection", "title_te": "మున్సిపల్ తడి వ్యర్థాల సేకరణ", "summary": "For food scraps accepted by your local wet-waste service.", "summary_te": "స్థానిక తడి వ్యర్థాల సేవ అంగీకరించే ఆహార మిగతాలకు.", "steps": ["Keep food scraps separate from plastic and other dry waste.", "Use the wet-waste container provided by your municipality.", "Put it out at the collection time and avoid leaving it accessible to animals."], "steps_te": ["ఆహార మిగతాలను ప్లాస్టిక్, పొడి వ్యర్థాల నుంచి వేరు చేయండి.", "మున్సిపాలిటీ ఇచ్చిన తడి వ్యర్థాల డబ్బాను ఉపయోగించండి.", "సేకరణ సమయానికి ఇవ్వండి; జంతువులకు అందుబాటులో వదలవద్దు."]},
        {"title": "Community composting programme", "title_te": "సామూహిక కంపోస్టింగ్ కార్యక్రమం", "summary": "Use a community compost site that confirms which food scraps it accepts.", "summary_te": "ఏ ఆహార మిగతాలను స్వీకరిస్తుందో తెలిపే సామూహిక కంపోస్టింగ్ కేంద్రాన్ని ఉపయోగించండి.", "steps": ["Check the site's accepted-material list first.", "Keep accepted organic scraps free from plastic and packaging.", "Deliver them to the designated compost collection point."], "steps_te": ["ముందుగా ఆ కేంద్రం అంగీకరించే పదార్థాల జాబితా చూడండి.", "అంగీకరించిన సేంద్రీయ మిగతాల్లో ప్లాస్టిక్ లేదా ప్యాకింగ్ కలపవద్దు.", "నిర్దేశించిన కంపోస్ట్ సేకరణ కేంద్రానికి ఇవ్వండి."]},
        {"title": "Approved home compost bin", "title_te": "అనుమతించిన ఇంటి కంపోస్ట్ బిన్", "summary": "Only for suitable fruit and vegetable scraps if your household can maintain a compost bin.", "summary_te": "ఇంట్లో కంపోస్ట్ బిన్‌ను సరిగ్గా నిర్వహించగలిగితే పండ్లు, కూరగాయల సరైన మిగతాలకు మాత్రమే.", "steps": ["Follow a trusted local composting guide for accepted materials.", "Keep plastics, packaging and unsuitable food items out.", "If home composting is not suitable, use municipal wet-waste collection."], "steps_te": ["అంగీకరించే పదార్థాల కోసం నమ్మదగిన స్థానిక కంపోస్టింగ్ సూచనలు పాటించండి.", "ప్లాస్టిక్, ప్యాకింగ్, సరిపోని ఆహార పదార్థాలు వేయవద్దు.", "ఇంట్లో కంపోస్టింగ్ సాధ్యం కాకపోతే మున్సిపల్ తడి వ్యర్థాల సేకరణ ఉపయోగించండి."]},
    ],
    "Glass": [
        {"title": "Glass bottle / jar collection", "title_te": "గాజు బాటిల్ / జార్ సేకరణ", "summary": "For intact glass accepted by a local collection programme.", "summary_te": "స్థానిక సేకరణ కార్యక్రమం అంగీకరించే పగలని గాజు వస్తువులకు.", "steps": ["Check that the local centre accepts this type and colour of glass.", "Keep intact glass separate from other waste.", "Use the designated glass collection point."], "steps_te": ["స్థానిక కేంద్రం ఈ రకం, రంగు గాజును స్వీకరిస్తుందో చూడండి.", "పగలని గాజును ఇతర వ్యర్థాల నుంచి వేరు చేయండి.", "నిర్దేశించిన గాజు సేకరణ కేంద్రాన్ని ఉపయోగించండి."]},
        {"title": "Municipal special collection", "title_te": "మున్సిపల్ ప్రత్యేక సేకరణ", "summary": "Ask the municipality how glass should be handed over in your area.", "summary_te": "మీ ప్రాంతంలో గాజును ఎలా అందించాలో మున్సిపాలిటీని అడగండి.", "steps": ["Keep glass separate and clearly identify it as glass waste.", "Do not put loose glass where people or animals can be injured.", "Follow the collection team's instructions."], "steps_te": ["గాజును వేరుగా ఉంచి గాజు వ్యర్థమని స్పష్టంగా తెలియజేయండి.", "మనుషులు లేదా జంతువులకు గాయం అయ్యేలా విడిగా పడేయవద్దు.", "సేకరణ బృందం సూచనలు పాటించండి."]},
        {"title": "Broken-glass safe handover", "title_te": "పగిలిన గాజును సురక్షితంగా అప్పగించడం", "summary": "Broken glass needs special care; do not touch it directly.", "summary_te": "పగిలిన గాజుకు ప్రత్యేక జాగ్రత్త అవసరం; నేరుగా తాకవద్దు.", "steps": ["Keep people and pets away from the broken glass.", "Do not pick it up with bare hands; ask an adult or collection worker for help.", "Use the municipality's approved broken-glass handover method."], "steps_te": ["పగిలిన గాజు దగ్గరకు మనుషులు, పెంపుడు జంతువులు వెళ్లకుండా చూడండి.", "చేతులతో తీసుకోవద్దు; పెద్దవారి లేదా సేకరణ సిబ్బంది సహాయం పొందండి.", "మున్సిపాలిటీ ఆమోదించిన పగిలిన గాజు అప్పగింపు విధానాన్ని అనుసరించండి."]},
    ],
    "Metal & Scrap": [
        {"title": "Authorised scrap-metal collector", "title_te": "అనుమతిపొందిన స్క్రాప్ మెటల్ సేకరణదారు", "summary": "For sorted metal items accepted by a registered or authorised collector.", "summary_te": "అధికారిక సేకరణదారు అంగీకరించే వేరు చేసిన లోహ వస్తువులకు.", "steps": ["Keep metal separate from wet and organic waste.", "Do not handle sharp, rusty or unknown scrap yourself.", "Ask an adult to arrange handover to an authorised collector."], "steps_te": ["లోహాన్ని తడి, సేంద్రీయ వ్యర్థాల నుంచి వేరు చేయండి.", "పదునైన, తుప్పుపట్టిన లేదా తెలియని స్క్రాప్‌ను మీరే తాకవద్దు.", "అనుమతిపొందిన సేకరణదారునికి అప్పగించేందుకు పెద్దవారి సహాయం తీసుకోండి."]},
        {"title": "Municipal dry-waste centre", "title_te": "మున్సిపల్ పొడి వ్యర్థాల కేంద్రం", "summary": "Use a local dry-waste centre that accepts metal packaging or items.", "summary_te": "లోహ ప్యాకింగ్ లేదా వస్తువులను అంగీకరించే స్థానిక పొడి వ్యర్థాల కేంద్రాన్ని ఉపయోగించండి.", "steps": ["Check whether the centre accepts the item.", "Keep it separate and away from children and pets if it has sharp edges.", "Follow the centre's handover instructions."], "steps_te": ["ఆ కేంద్రం ఆ వస్తువును అంగీకరిస్తుందో చూడండి.", "పదునైన అంచులు ఉంటే పిల్లలు, జంతువులకు దూరంగా వేరుగా ఉంచండి.", "కేంద్రం అప్పగింపు సూచనలు పాటించండి."]},
        {"title": "Metal packaging collection", "title_te": "లోహ ప్యాకింగ్ సేకరణ", "summary": "For cans and other metal packaging accepted by local recycling services.", "summary_te": "స్థానిక రీసైక్లింగ్ సేవ అంగీకరించే క్యాన్లు, లోహ ప్యాకింగ్ కోసం.", "steps": ["Separate accepted metal packaging from mixed waste.", "Follow local guidance on emptying and sorting it.", "Place it in the designated metal or dry-recycling collection."], "steps_te": ["అంగీకరించే లోహ ప్యాకింగ్‌ను మిశ్రమ వ్యర్థాల నుంచి వేరు చేయండి.", "ఖాళీ చేయడం, వేరు చేయడం గురించి స్థానిక సూచనలు పాటించండి.", "నిర్దేశించిన లోహ లేదా పొడి రీసైక్లింగ్ సేకరణలో ఇవ్వండి."]},
    ],
    "Hazardous Waste": [
        {"title": "Authorised hazardous-waste facility", "title_te": "అనుమతిపొందిన ప్రమాదకర వ్యర్థాల కేంద్రం", "summary": "For chemicals, paint, unknown substances and other hazardous household waste.", "summary_te": "రసాయనాలు, పెయింట్, తెలియని పదార్థాలు మరియు ఇతర ప్రమాదకర ఇంటి వ్యర్థాలకు.", "steps": ["Do not open, mix, burn or pour the material into a drain.", "Keep it in its original labelled container where possible and away from children and pets.", "Ask an adult to contact the municipality or an authorised collection facility."], "steps_te": ["వాటిని తెరవవద్దు, కలపవద్దు, కాల్చవద్దు, డ్రైన్లో పోయవద్దు.", "సాధ్యమైతే అసలు లేబుల్ ఉన్న డబ్బాలోనే ఉంచి పిల్లలు, జంతువులకు దూరంగా ఉంచండి.", "మున్సిపాలిటీ లేదా అనుమతిపొందిన కేంద్రాన్ని సంప్రదించేందుకు పెద్దవారి సహాయం తీసుకోండి."]},
        {"title": "Battery collection / take-back", "title_te": "బ్యాటరీ సేకరణ / తిరిగి స్వీకరణ", "summary": "Use a dedicated battery collection point or an approved take-back programme.", "summary_te": "ప్రత్యేక బ్యాటరీ సేకరణ కేంద్రం లేదా ఆమోదిత తిరిగి స్వీకరణ పథకాన్ని ఉపయోగించండి.", "steps": ["Do not open, crush, puncture or place batteries in regular bins.", "Keep batteries separate and dry; do not mix damaged batteries with other items.", "Ask an adult to take them to an approved battery collection point."], "steps_te": ["బ్యాటరీలను తెరవవద్దు, నలపవద్దు, గుచ్చవద్దు, సాధారణ చెత్తలో వేయవద్దు.", "బ్యాటరీలను వేరుగా, పొడిగా ఉంచండి; దెబ్బతిన్న బ్యాటరీలను ఇతర వస్తువులతో కలపవద్దు.", "ఆమోదిత బ్యాటరీ సేకరణ కేంద్రానికి ఇవ్వడానికి పెద్దవారి సహాయం తీసుకోండి."]},
        {"title": "Special municipal collection", "title_te": "మున్సిపల్ ప్రత్యేక సేకరణ", "summary": "Check local arrangements for household hazardous or medical waste.", "summary_te": "ఇంటి ప్రమాదకర లేదా వైద్య వ్యర్థాల కోసం స్థానిక ఏర్పాట్లు తెలుసుకోండి.", "steps": ["Keep the waste separate from household recycling and food waste.", "Do not handle unknown or leaking material; keep others away.", "Contact the municipality or an authorised service for safe collection."], "steps_te": ["ఇంటి రీసైక్లింగ్, ఆహార వ్యర్థాల నుంచి వేరుగా ఉంచండి.", "తెలియని లేదా లీక్ అవుతున్న పదార్థాన్ని తాకవద్దు; ఇతరులను దూరంగా ఉంచండి.", "సురక్షిత సేకరణ కోసం మున్సిపాలిటీ లేదా అనుమతిపొందిన సేవను సంప్రదించండి."]},
    ],
    "Textile & Fabric Waste": [
        {"title": "Donation of clean wearable items", "title_te": "శుభ్రమైన ధరించగల దుస్తుల దానం", "summary": "For wearable clothing accepted by a local donation group.", "summary_te": "స్థానిక దాన సంస్థ అంగీకరించే ధరించగల దుస్తులకు.", "steps": ["Separate clean, dry and wearable clothing.", "Check the donation group's acceptance rules.", "Deliver it through an established donation point."], "steps_te": ["శుభ్రమైన, పొడిగా ఉన్న, ధరించగల దుస్తులను వేరు చేయండి.", "దాన సంస్థ అంగీకార నియమాలు తెలుసుకోండి.", "స్థిరమైన దాన కేంద్రం ద్వారా అందించండి."]},
        {"title": "Textile recovery / collection point", "title_te": "వస్త్రాల రికవరీ / సేకరణ కేంద్రం", "summary": "For worn fabric that a textile-recovery programme accepts.", "summary_te": "వస్త్రాల రికవరీ కార్యక్రమం అంగీకరించే పాత బట్టలకు.", "steps": ["Keep textiles dry and separate from wet waste.", "Check whether the collection point accepts this fabric type.", "Hand it over through the approved textile collection."], "steps_te": ["బట్టలను పొడిగా, తడి వ్యర్థాల నుంచి వేరుగా ఉంచండి.", "ఆ సేకరణ కేంద్రం ఈ రకం బట్టను అంగీకరిస్తుందో చూడండి.", "ఆమోదిత వస్త్రాల సేకరణ ద్వారా అప్పగించండి."]},
        {"title": "Residual-waste collection", "title_te": "మిగిలిన వ్యర్థాల సేకరణ", "summary": "For textiles not accepted for donation or textile recovery.", "summary_te": "దానం లేదా వస్త్రాల రికవరీకి అంగీకరించని బట్టలకు.", "steps": ["Do not burn or dump unwanted textiles.", "Keep them separate from clean reusable clothes.", "Use the local residual-waste route and follow municipal rules."], "steps_te": ["వద్దనుకున్న బట్టలను కాల్చవద్దు లేదా ఎక్కడపడితే అక్కడ వేయవద్దు.", "వాటిని శుభ్రమైన, మళ్లీ ఉపయోగించగల దుస్తుల నుంచి వేరు చేయండి.", "స్థానిక మిగిలిన వ్యర్థాల మార్గాన్ని, మున్సిపల్ నియమాలను పాటించండి."]},
    ],
    "E-Waste & Others": [
        {"title": "Authorised e-waste collection centre", "title_te": "అనుమతిపొందిన ఈ-వ్యర్థాల సేకరణ కేంద్రం", "summary": "For discarded electronics, cables and accessories.", "summary_te": "పారవేయాల్సిన ఎలక్ట్రానిక్ పరికరాలు, కేబుళ్లు, ఉపకరణాలకు.", "steps": ["Do not dismantle, break or burn electronic devices.", "Keep devices dry and separate from regular household waste.", "Ask an adult to use an authorised e-waste collection centre."], "steps_te": ["ఎలక్ట్రానిక్ పరికరాలను విడదీయవద్దు, పగలగొట్టవద్దు, కాల్చవద్దు.", "పరికరాలను పొడిగా, సాధారణ ఇంటి చెత్త నుంచి వేరుగా ఉంచండి.", "అనుమతిపొందిన ఈ-వ్యర్థాల కేంద్రాన్ని ఉపయోగించేందుకు పెద్దవారి సహాయం తీసుకోండి."]},
        {"title": "Brand / producer take-back", "title_te": "బ్రాండ్ / తయారీదారు తిరిగి స్వీకరణ", "summary": "Use an official manufacturer collection or take-back programme where available.", "summary_te": "అందుబాటులో ఉంటే తయారీదారు అధికారిక సేకరణ లేదా తిరిగి స్వీకరణ పథకాన్ని ఉపయోగించండి.", "steps": ["Check the manufacturer's official take-back instructions.", "Keep the device and accessories together if requested.", "Hand them over only through the official collection channel."], "steps_te": ["తయారీదారు అధికారిక తిరిగి స్వీకరణ సూచనలు చూడండి.", "అవసరమైతే పరికరం, ఉపకరణాలను కలిపి ఉంచండి.", "అధికారిక సేకరణ మార్గం ద్వారానే అప్పగించండి."]},
        {"title": "Separate battery handover", "title_te": "బ్యాటరీలను వేరుగా అప్పగించడం", "summary": "For devices with removable batteries or loose batteries; follow approved collection guidance.", "summary_te": "తీయగల బ్యాటరీలు లేదా విడిగా ఉన్న బ్యాటరీల కోసం; ఆమోదిత సేకరణ సూచనలు పాటించండి.", "steps": ["Do not remove a built-in battery or open the device yourself.", "Keep loose batteries separate and do not put them in regular bins.", "Ask an adult or authorised e-waste service for the correct handover route."], "steps_te": ["లోపల అమర్చిన బ్యాటరీని మీరే తీయవద్దు, పరికరాన్ని తెరవవద్దు.", "విడిగా ఉన్న బ్యాటరీలను వేరుగా ఉంచి సాధారణ చెత్తలో వేయవద్దు.", "సరైన అప్పగింపు మార్గం కోసం పెద్దవారి లేదా అనుమతిపొందిన ఈ-వ్యర్థాల సేవ సహాయం పొందండి."]},
    ],
}

# A small local knowledge base; it works without paid APIs or API keys.
FAQ_KB = {
    "plastic": {
        "en": "Separate plastic from food and wet waste. Check the item type and local acceptance rules; not every plastic is recyclable everywhere. Use a municipal dry-waste collector or authorised recycler. Never burn plastic or leave it in the environment.",
        "te": "ప్లాస్టిక్‌ను ఆహారం, తడి వ్యర్థాల నుంచి వేరు చేయండి. ప్లాస్టిక్ రకం, స్థానిక అంగీకార నియమాలు చూడండి; అన్ని రకాల ప్లాస్టిక్‌ను ప్రతి చోటా రీసైకిల్ చేయరు. మున్సిపల్ పొడి వ్యర్థాల సేకరణ లేదా అనుమతిపొందిన రీసైక్లర్‌కు ఇవ్వండి. ప్లాస్టిక్‌ను కాల్చవద్దు, బయట పారేయవద్దు.",
    },
    "paper": {
        "en": "Keep paper and cardboard clean and dry. Separate oily or food-soiled paper because it may not be accepted in paper recycling. Flatten cardboard if your local collector asks, then use a paper or dry-waste collection point.",
        "te": "కాగితం, కార్డ్‌బోర్డ్‌ను శుభ్రంగా, పొడిగా ఉంచండి. నూనె లేదా ఆహారం అంటిన కాగితాన్ని వేరు చేయండి; దాన్ని పేపర్ రీసైక్లింగ్‌లో అంగీకరించకపోవచ్చు. స్థానిక సేకరణదారు సూచిస్తే కార్డ్‌బోర్డ్‌ను మడిచి పేపర్ లేదా పొడి వ్యర్థాల కేంద్రానికి ఇవ్వండి.",
    },
    "glass": {
        "en": "Glass is not biodegradable, but many bottles and jars can be recycled where collection exists. Keep glass separate. Never touch broken glass directly; keep people and pets away and ask an adult or collection worker for the approved handover method.",
        "te": "గాజు జీవ విచ్ఛిన్నం కాదు, కానీ సేకరణ సదుపాయం ఉన్నచోట బాటిళ్లు, జార్లను రీసైకిల్ చేయవచ్చు. గాజును వేరుగా ఉంచండి. పగిలిన గాజును నేరుగా తాకవద్దు; మనుషులు, జంతువులను దూరంగా ఉంచి పెద్దవారి లేదా సేకరణ సిబ్బంది సహాయం పొందండి.",
    },
    "metal": {
        "en": "Separate accepted metal cans and scrap for an authorised recycler or municipal dry-waste centre. Do not handle sharp, rusty or unknown metal pieces; ask an adult or trained collector to manage them.",
        "te": "అంగీకరించే లోహ క్యాన్లు, స్క్రాప్‌ను అనుమతిపొందిన రీసైక్లర్ లేదా మున్సిపల్ పొడి వ్యర్థాల కేంద్రానికి వేరుగా ఇవ్వండి. పదునైన, తుప్పుపట్టిన లేదా తెలియని లోహ ముక్కలను తాకవద్దు; పెద్దవారి లేదా శిక్షణ పొందిన సేకరణదారుని సహాయం తీసుకోండి.",
    },
    "organic": {
        "en": "Fruit and vegetable scraps can often go to municipal wet-waste collection or an approved compost programme. Keep plastics out. Follow the facility's accepted-material list; simple home compost systems may not accept meat, dairy or oily food.",
        "te": "పండ్లు, కూరగాయల మిగతాలను సాధారణంగా మున్సిపల్ తడి వ్యర్థాల సేకరణ లేదా ఆమోదిత కంపోస్టింగ్ కార్యక్రమానికి ఇవ్వవచ్చు. ప్లాస్టిక్ కలపవద్దు. కేంద్రం అంగీకరించే పదార్థాల జాబితా పాటించండి; ఇంటి సాధారణ కంపోస్ట్‌లో మాంసం, పాల పదార్థాలు, ఎక్కువ నూనె ఉన్న ఆహారం సరిపోకపోవచ్చు.",
    },
    "compost": {
        "en": "Composting is the controlled breakdown of suitable organic materials into a soil-like amendment. Use only materials accepted by the compost service or bin instructions. Keep plastic, packaging and unsuitable food out.",
        "te": "కంపోస్టింగ్ అంటే సరైన సేంద్రీయ పదార్థాలు నియంత్రితంగా కుళ్లి మట్టిలాంటి పదార్థంగా మారే ప్రక్రియ. కంపోస్టింగ్ సేవ లేదా బిన్ సూచనలు అంగీకరించే పదార్థాలనే వాడండి. ప్లాస్టిక్, ప్యాకింగ్, సరిపోని ఆహారం వేయవద్దు.",
    },
    "hazard": {
        "en": "Batteries, chemicals, paint, medicines and medical waste should not go into ordinary recycling or drains. Do not open, mix, burn or pour them away. Keep them separate and ask an adult to contact an authorised collection service.",
        "te": "బ్యాటరీలు, రసాయనాలు, పెయింట్, మందులు, వైద్య వ్యర్థాలను సాధారణ రీసైక్లింగ్ లేదా డ్రైన్లలో వేయకూడదు. వాటిని తెరవవద్దు, కలపవద్దు, కాల్చవద్దు, పారబోయవద్దు. వేరుగా ఉంచి అనుమతిపొందిన సేకరణ సేవను సంప్రదించేందుకు పెద్దవారి సహాయం తీసుకోండి.",
    },
    "ewaste": {
        "en": "Discarded phones, computers, cables and appliances belong in an authorised e-waste collection or official manufacturer take-back programme. Do not dismantle devices or batteries and do not put e-waste in regular bins.",
        "te": "పారవేయాల్సిన ఫోన్లు, కంప్యూటర్లు, కేబుళ్లు, పరికరాలను అనుమతిపొందిన ఈ-వ్యర్థాల కేంద్రం లేదా తయారీదారు అధికారిక తిరిగి స్వీకరణ పథకానికి ఇవ్వండి. పరికరాలు లేదా బ్యాటరీలను విడదీయవద్దు; సాధారణ చెత్తలో వేయవద్దు.",
    },
    "biodegradable": {
        "en": "Biodegradable means microorganisms can break a material down under suitable conditions. It does not automatically mean the item belongs in home compost; check the material and local collection rules.",
        "te": "సరైన పరిస్థితుల్లో సూక్ష్మజీవులు ఒక పదార్థాన్ని విచ్ఛిన్నం చేయగలిగితే దాన్ని జీవ విచ్ఛిన్న పదార్థం అంటారు. అందుకే అది ఇంటి కంపోస్ట్‌లో వేయవచ్చని కాదు; పదార్థం, స్థానిక సేకరణ నియమాలు చూడాలి.",
    },
    "recyclable": {
        "en": "Recyclable means a material can be processed into a new product when a suitable collection and processing facility exists. Cleanliness, material type and local programmes matter; the recycling symbol alone does not guarantee acceptance.",
        "te": "రీసైకిల్ చేయగలదంటే తగిన సేకరణ, ప్రాసెసింగ్ సదుపాయం ఉన్నప్పుడు ఆ పదార్థాన్ని కొత్త ఉత్పత్తిగా మార్చవచ్చని అర్థం. శుభ్రత, పదార్థ రకం, స్థానిక పథకాలు ముఖ్యం; రీసైక్లింగ్ గుర్తు ఉందని ప్రతి చోటా అంగీకరిస్తారని కాదు.",
    },
    "segregation": {
        "en": "A useful basic separation is wet/organic waste, dry recyclables, and domestic hazardous or e-waste. Follow the exact bin colours and collection categories used by your municipality, because rules differ by place.",
        "te": "ప్రాథమికంగా తడి/సేంద్రీయ వ్యర్థాలు, పొడి రీసైకిల్ పదార్థాలు, ఇంటి ప్రమాదకర లేదా ఈ-వ్యర్థాలను వేరుచేయవచ్చు. బిన్ రంగులు, సేకరణ విభాగాలు ప్రాంతాన్ని బట్టి మారుతాయి కాబట్టి మీ మున్సిపాలిటీ నియమాలు పాటించండి.",
    },
    "reduce": {
        "en": "Reduce waste by choosing durable products, avoiding unnecessary single-use packaging, carrying reusable bags or bottles, repairing items when practical, and separating materials for the correct local collection route.",
        "te": "మన్నికైన వస్తువులు ఎంచుకోవడం, అవసరం లేని ఒక్కసారి వాడే ప్యాకింగ్ తగ్గించడం, మళ్లీ వాడగల సంచులు లేదా బాటిళ్లు తీసుకెళ్లడం, సాధ్యమైనప్పుడు మరమ్మతు చేయడం, వ్యర్థాలను సరైన స్థానిక సేకరణకు వేరు చేయడం ద్వారా చెత్తను తగ్గించవచ్చు.",
    },
    "animals": {
        "en": "Keep food waste, plastic, strings and small discarded items away from animals. Do not leave waste in open places or waterways. Use closed collection bins and report dumped hazardous waste to local authorities.",
        "te": "ఆహార వ్యర్థాలు, ప్లాస్టిక్, దారాలు, చిన్న పారేసిన వస్తువులు జంతువులకు అందకుండా ఉంచండి. చెత్తను బహిరంగ ప్రదేశాల్లో లేదా నీటి మార్గాల్లో వదలవద్దు. మూత ఉన్న సేకరణ డబ్బాలు వాడండి; ప్రమాదకర చెత్తను పారేసినట్లు కనిపిస్తే స్థానిక అధికారులకు తెలియజేయండి.",
    },
    "general_disposal": {
        "en": "First identify the material, then separate it from wet and dry waste. Check your municipality's accepted collection categories. Use an authorised recycler or special collection for e-waste and hazardous items. Never burn waste, dump it outdoors or pour it into drains.",
        "te": "ముందుగా పదార్థాన్ని గుర్తించి, తడి మరియు పొడి వ్యర్థాల నుంచి వేరు చేయండి. మీ మున్సిపాలిటీ అంగీకరించే సేకరణ విభాగాలు చూడండి. ఈ-వ్యర్థాలు, ప్రమాదకర వస్తువులకు అనుమతిపొందిన రీసైక్లర్ లేదా ప్రత్యేక సేకరణ ఉపయోగించండి. చెత్తను కాల్చవద్దు, బయట పారేయవద్దు, డ్రైన్లలో పోయవద్దు.",
    },
    "impact": {
        "en": "Environmental impact depends on the material, quantity, condition and where it ends up. Loose plastic can persist and harm wildlife; broken glass can injure; hazardous chemicals and e-waste need special collection. A photo cannot measure an exact harm percentage, so this app shows only an illustrative category-based priority index.",
        "te": "పర్యావరణ ప్రభావం పదార్థం, పరిమాణం, స్థితి మరియు అది ఎక్కడ పడుతుందనే అంశాలపై ఆధారపడి ఉంటుంది. ప్లాస్టిక్ ఎక్కువకాలం ఉండి జంతువులకు హాని చేయవచ్చు; పగిలిన గాజు గాయపరచవచ్చు; ప్రమాదకర రసాయనాలు, ఈ-వ్యర్థాలకు ప్రత్యేక సేకరణ అవసరం. ఫోటో ఆధారంగా ఖచ్చితమైన హాని శాతాన్ని కొలవలేం; అందుకే ఈ యాప్ వర్గం ఆధారిత సూచిక మాత్రమే చూపిస్తుంది.",
    },
}

FAQ_KEYWORDS = [
    ("hazard", ["hazard", "chemical", "paint", "medicine", "medical waste", "ప్రమాదకర", "రసాయన", "మందులు", "వైద్య వ్యర్థ"]),
    ("ewaste", ["e-waste", "ewaste", "electronic", "laptop", "computer", "mobile", "phone", "ఈ-వ్యర్థ", "ఎలక్ట్రానిక్", "కంప్యూటర్", "ఫోన్"]),
    ("battery", ["battery", "బ్యాటరీ"]),
    ("glass", ["glass", "గాజు"]),
    ("metal", ["metal", "scrap", "can", "లోహ", "స్క్రాప్", "క్యాన్"]),
    ("paper", ["paper", "cardboard", "కాగితం", "కార్డ్‌బోర్డ్"]),
    ("plastic", ["plastic", "ప్లాస్టిక్"]),
    ("compost", ["compost", "కంపోస్ట్"]),
    ("organic", ["organic", "food waste", "wet waste", "సేంద్రీయ", "ఆహార వ్యర్థ", "తడి వ్యర్థ"]),
    ("biodegradable", ["biodegradable", "non-biodegradable", "జీవ విచ్ఛిన్న", "జీవవిచ్ఛిన్న"]),
    ("recyclable", ["recyclable", "recycling", "recycle", "రీసైకిల్", "రీసైక్లింగ్"]),
    ("segregation", ["segregate", "segregation", "separate waste", "waste sorting", "వేరు చేయ", "వర్గీకరణ"]),
    ("animals", ["animal", "animals", "wildlife", "పశువులు", "జంతువులు", "వన్యప్రాణ"]),
    ("impact", ["environment impact", "harmful to environment", "harm to human", "harm to people", "pollution", "impact percentage", "పర్యావరణ హాని", "మనుషులకు హాని", "కాలుష్యం"]),
    ("general_disposal", ["dispose", "disposal", "throw away", "where to throw", "how to discard", "పారవేయ", "ఎక్కడ వేయాలి", "ఎలా వేయాలి"]),
    ("reduce", ["reduce waste", "less waste", "waste reduction", "చెత్త తగ్గ", "వ్యర్థాలు తగ్గ"]),
]


def tx(key: str, lang: str) -> str:
    return TEXT.get(lang, TEXT["en"]).get(key, TEXT["en"].get(key, key))


def category_display(category: str, lang: str) -> str:
    return CATEGORY_NAMES.get(category, {}).get(lang, category)


@st.cache_data(show_spinner=False)
def load_labels() -> list[str]:
    if not LABELS_PATH.exists():
        raise FileNotFoundError("labels.txt is missing")
    labels: list[str] = []
    with LABELS_PATH.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            labels.append(parts[1] if len(parts) == 2 and parts[0].isdigit() else line)
    if not labels:
        raise ValueError("labels.txt contains no class labels")
    return labels


class CompatibleDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    @classmethod
    def from_config(cls, config: dict[str, Any]):
        config.pop("groups", None)
        return super().from_config(config)


@st.cache_resource(show_spinner="Loading waste classification model...")
def load_classifier():
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError("keras_model.h5 is missing")
    return tf.keras.models.load_model(
        str(CLASSIFIER_PATH),
        custom_objects={
            "DepthwiseConv2D": CompatibleDepthwiseConv2D,
            "CompatibleDepthwiseConv2D": CompatibleDepthwiseConv2D,
        },
        compile=False,
    )


def download_detector_model() -> Path:
    if DETECTOR_MODEL_PATH.exists() and DETECTOR_MODEL_PATH.stat().st_size > 1_000_000:
        return DETECTOR_MODEL_PATH

    temp_path = DETECTOR_MODEL_PATH.with_suffix(".download")
    try:
        response = requests.get(DETECTOR_MODEL_URL, stream=True, timeout=(20, 180))
        response.raise_for_status()
        with temp_path.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
        if temp_path.stat().st_size < 1_000_000:
            raise ValueError("Downloaded object detector model is unexpectedly small")
        temp_path.replace(DETECTOR_MODEL_PATH)
        return DETECTOR_MODEL_PATH
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


@st.cache_resource(show_spinner="Preparing multi-object detector (first launch downloads a free model)...")
def load_object_detector():
    model_path = download_detector_model()
    options = mp.tasks.vision.ObjectDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        score_threshold=0.35,
        max_results=25,
    )
    return mp.tasks.vision.ObjectDetector.create_from_options(options)


def predict_category(crop: Image.Image, model, labels: list[str]) -> tuple[str, float]:
    image_shape = model.input_shape
    target_height = int(image_shape[1] or 224)
    target_width = int(image_shape[2] or 224)
    prepared = ImageOps.fit(
        crop.convert("RGB"),
        (target_width, target_height),
        method=Image.Resampling.LANCZOS,
    )
    array = np.asarray(prepared, dtype=np.float32)
    array = (array / 127.5) - 1.0
    prediction = np.asarray(model.predict(np.expand_dims(array, axis=0), verbose=0)).flatten()
    if prediction.size == 0:
        return "Unknown", 0.0
    index = int(np.argmax(prediction))
    category = labels[index] if index < len(labels) else "Unknown"
    return category, float(prediction[index])


def locate_and_classify(image: Image.Image, detector, classifier, labels: list[str]) -> tuple[list[dict[str, Any]], int]:
    rgb = np.asarray(image.convert("RGB"))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection_result = detector.detect(mp_image)
    image_width, image_height = image.size
    records: list[dict[str, Any]] = []
    ignored_people = 0

    for detection in detection_result.detections:
        if not detection.categories:
            continue
        detected_class = detection.categories[0]
        object_name = (detected_class.category_name or "").strip().lower()
        object_score = float(detected_class.score or 0.0)

        if object_name == "person":
            ignored_people += 1
            continue
        if object_name not in WASTE_CANDIDATE_LABELS:
            continue

        box = detection.bounding_box
        left = max(0, int(box.origin_x))
        top = max(0, int(box.origin_y))
        right = min(image_width, int(box.origin_x + box.width))
        bottom = min(image_height, int(box.origin_y + box.height))
        if right <= left or bottom <= top:
            continue

        # Add a small border so the classifier sees enough of the object's context.
        pad_x = int((right - left) * 0.06)
        pad_y = int((bottom - top) * 0.06)
        crop_box = (
            max(0, left - pad_x), max(0, top - pad_y),
            min(image_width, right + pad_x), min(image_height, bottom + pad_y),
        )
        crop = image.crop(crop_box)
        category, category_score = predict_category(crop, classifier, labels)

        records.append({
            "object_name": object_name,
            "object_score": object_score,
            "category": category,
            "category_score": category_score,
            "crop": crop,
            "bbox": (left, top, right, bottom),
            "requires_confirmation": object_name in ELECTRONIC_LABELS,
        })

    # Stable reading order: top-to-bottom, then left-to-right.
    records.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return records, ignored_people


def draw_detection_boxes(image: Image.Image, records: list[dict[str, Any]]) -> Image.Image:
    marked = image.copy().convert("RGB")
    draw = ImageDraw.Draw(marked)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(14, min(24, marked.width // 40)))
    except OSError:
        font = ImageFont.load_default()
    for index, record in enumerate(records, start=1):
        x1, y1, x2, y2 = record["bbox"]
        draw.rectangle((x1, y1, x2, y2), outline="#16a34a", width=max(3, marked.width // 350))
        label = f"{index}. {record['object_name']}"
        text_box = draw.textbbox((x1, max(0, y1 - 30)), label, font=font)
        draw.rectangle(text_box, fill="#166534")
        draw.text((x1, max(0, y1 - 30)), label, fill="white", font=font)
    return marked


def make_disposal_visual(category: str, method_title: str) -> Image.Image:
    """Create a simple local infographic (no image API or external image service)."""
    width, height = 1200, 470
    canvas = Image.new("RGB", (width, height), "#f4f8f4")
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 31)
        step_font = ImageFont.truetype("DejaVuSans.ttf", 24)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        title_font = step_font = small_font = ImageFont.load_default()

    draw.rounded_rectangle((24, 20, width - 24, 96), radius=20, fill="#14532d")
    draw.text((52, 39), "SAFE DISPOSAL GUIDE", font=title_font, fill="white")
    draw.text((width - 440, 46), category[:28], font=small_font, fill="#dcfce7")

    cards = [
        ("1", "SEPARATE", "Keep this waste apart"),
        ("2", "KEEP SAFE", "Avoid spills and exposure"),
        ("3", "HAND OVER", "Use an approved route"),
    ]
    card_w = 340
    xs = [35, 430, 825]
    for x, (number, heading, detail) in zip(xs, cards):
        draw.rounded_rectangle((x, 142, x + card_w, 392), radius=22, fill="white", outline="#cbd5d1", width=3)
        draw.ellipse((x + 24, 164, x + 82, 222), fill="#16a34a")
        draw.text((x + 44, 174), number, font=step_font, fill="white")
        draw.text((x + 102, 180), heading, font=step_font, fill="#14532d")
        # Simple visual symbols: three bins/box, closed container, and collection point.
        if number == "1":
            draw.rectangle((x + 110, 254, x + 230, 330), fill="#bbf7d0", outline="#166534", width=4)
            draw.rectangle((x + 100, 244, x + 240, 260), fill="#166534")
            for xx in (x + 140, x + 175, x + 210):
                draw.line((xx, 267, xx, 315), fill="#166534", width=4)
        elif number == "2":
            draw.rounded_rectangle((x + 115, 250, x + 225, 330), radius=10, fill="#dbeafe", outline="#1d4ed8", width=4)
            draw.arc((x + 140, 218, x + 200, 280), 180, 360, fill="#1d4ed8", width=5)
        else:
            draw.rectangle((x + 105, 270, x + 235, 330), fill="#dcfce7", outline="#166534", width=4)
            draw.polygon([(x + 95, 270), (x + 170, 225), (x + 245, 270)], fill="#86efac", outline="#166534")
            draw.rectangle((x + 155, 292, x + 185, 330), fill="#166534")
        draw.text((x + 30, 350), detail, font=small_font, fill="#334155")
    draw.text((45, 420), "Always follow your local municipal collection rules.", font=small_font, fill="#475569")
    return canvas


def risk_explanation(lang: str) -> str:
    if lang == "te":
        return "ఇవి వాస్తవ నష్టం శాతాలు కావు. పదార్థ వర్గం ఆధారంగా చూపే సూచికలు మాత్రమే. నిజమైన ప్రభావం పదార్థం, పరిమాణం, స్థితి, పరిసరాలు మరియు పరిచయంపై ఆధారపడి మారుతుంది."
    return "These are not measured damage percentages. They are illustrative category-based priority indices only. Actual impact depends on material, quantity, condition, exposure and local environment."


def get_guide_text(guide: dict[str, Any], field: str, lang: str):
    if lang == "te":
        return guide.get(f"{field}_te", guide.get(field, ""))
    return guide.get(field, "")


def classify_risk_label(score: int) -> str:
    if score >= 75:
        return "Higher priority"
    if score >= 45:
        return "Moderate priority"
    return "Lower priority"


def answer_prithvipal(question: str, lang: str) -> tuple[str, str]:
    q = question.lower().strip()
    previous = st.session_state.get("prithvi_last_topic")
    follow_up_terms = ["more", "explain", "details", "steps", "examples", "మరింత", "ఇంకా", "వివరాలు", "చెప్పండి"]
    if previous and any(term in q for term in follow_up_terms):
        topic = previous
    else:
        topic = None
        for key, phrases in FAQ_KEYWORDS:
            if any(phrase in q for phrase in phrases):
                topic = "hazard" if key == "battery" else key
                break
    if topic in FAQ_KB:
        st.session_state["prithvi_last_topic"] = topic
        answer = FAQ_KB[topic].get(lang, FAQ_KB[topic]["en"])
        return answer, "Knowledge base"
    fallback = tx("unknown", lang)
    return fallback, "Knowledge base"


# Sidebar and first-visit welcome screen.
st.sidebar.title("♻️ EcoVision AI")
language_name = st.sidebar.selectbox(tx("language", "en"), list(LANGUAGES.keys()), index=0)
lang = LANGUAGES[language_name]

st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 1.5rem; padding-bottom: 2.5rem;}
.eco-hero {padding: 1.25rem 1.5rem; border-radius: 18px; background: linear-gradient(120deg,#e8f5e9,#f1f8e9); border: 1px solid #c8e6c9; margin-bottom: 1rem;}
.eco-hero h1 {margin: 0; color: #14532d;}
.eco-hero p {margin: .4rem 0 0 0; color: #365b40;}
.small-note {font-size: .88rem; color: #64748b;}
.welcome-wrap {padding: 1.5rem 1.2rem 1rem 1.2rem; border-radius: 26px; background: linear-gradient(135deg,#e7f8ec 0%,#f0fdf4 45%,#eff6ff 100%); border: 1px solid #ccebd5; margin: .5rem 0 1rem 0;}
.welcome-kicker {display:inline-block; padding:.35rem .75rem; border-radius:999px; background:#dcfce7; color:#166534; font-weight:700; font-size:.82rem; letter-spacing:.02em;}
.welcome-title {color:#14532d; font-size:2.5rem; line-height:1.12; font-weight:800; margin:.85rem 0 .5rem 0;}
.welcome-subtitle {color:#365b40; font-size:1.08rem; max-width:780px; line-height:1.65;}
.eco-card {height:100%; padding:1rem; border:1px solid #d7e9dc; border-radius:18px; background:rgba(255,255,255,.86); box-shadow:0 4px 16px rgba(22,101,52,.05);}
.eco-card-icon {font-size:2rem; margin-bottom:.35rem;}
.eco-card h3 {font-size:1.12rem; color:#14532d; margin:.15rem 0 .35rem 0;}
.eco-card p {font-size:.94rem; line-height:1.5; color:#475569; margin:0;}
.welcome-foot {text-align:center; color:#64748b; font-size:.88rem; margin-top:.5rem;}
@keyframes eco-float {0%,100%{transform:translateY(0)}50%{transform:translateY(-7px)}}
.eco-floating {display:inline-block; animation:eco-float 3.4s ease-in-out infinite; font-size:3.6rem;}
@media (prefers-reduced-motion: reduce) {.eco-floating {animation:none;}}
</style>
""", unsafe_allow_html=True)

WELCOME_TEXT = {
    "en": {
        "kicker": "A SMALL STEP FOR YOU • A BIG STEP FOR OUR PLANET",
        "title": "A cleaner world starts with one better choice.",
        "subtitle": "Welcome to EcoVision AI. Explore waste, understand what it may be made of, and choose a safer way to separate and dispose of it. Every thoughtful action helps protect people, animals and the places we share.",
        "how": "Your eco journey",
        "reduce_title": "Reduce",
        "reduce": "Choose durable items and avoid unnecessary single-use packaging.",
        "separate_title": "Separate",
        "separate": "Keep wet, dry, recyclable and special-care waste apart as local rules advise.",
        "dispose_title": "Dispose safely",
        "dispose": "Use the correct collection service. Never burn, litter or pour waste into drains.",
        "tip_label": "Choose a small green action to learn about",
        "tips": {
            "Carry a reusable bottle": "Refill a durable bottle where safe drinking water is available. This can reduce the need for single-use bottles.",
            "Sort waste at home": "Keep food scraps separate from clean, dry recyclables. Follow your municipality’s bin and collection rules.",
            "Keep batteries separate": "Do not put batteries in ordinary household waste. Ask an authorised collection point or a responsible adult for the correct route.",
            "Protect animals": "Keep plastic, strings and food waste inside closed bins so animals cannot reach or swallow them.",
        },
        "button": "🌱 Start with EcoVision AI",
        "no_login": "No account or sign-in needed. Choose a language, upload a photo and explore the result.",
        "footer": "Learn • Identify • Separate • Dispose responsibly",
    },
    "te": {
        "kicker": "మీ చిన్న అడుగు • భూమికి పెద్ద మేలు",
        "title": "శుభ్రమైన ప్రపంచం ఒక మంచి నిర్ణయంతో ప్రారంభమవుతుంది.",
        "subtitle": "EcoVision AIకి స్వాగతం. వ్యర్థాల గురించి తెలుసుకోండి, ఫోటోలోని గుర్తించగల వస్తువులను పరిశీలించండి, వాటిని వేరు చేసి సురక్షితంగా పారవేసే మార్గాన్ని ఎంచుకోండి. మనుషులు, జంతువులు, పర్యావరణాన్ని కాపాడటంలో ప్రతి మంచి చర్య ఉపయోగపడుతుంది.",
        "how": "మీ పర్యావరణ ప్రయాణం",
        "reduce_title": "తగ్గించండి",
        "reduce": "మన్నికైన వస్తువులు ఎంచుకోండి. అవసరం లేని ఒక్కసారి వాడే ప్యాకింగ్‌ను తగ్గించండి.",
        "separate_title": "వేరు చేయండి",
        "separate": "స్థానిక నియమాల ప్రకారం తడి, పొడి, రీసైకిల్ చేయగల, ప్రత్యేక జాగ్రత్త అవసరమైన వ్యర్థాలను వేరుగా ఉంచండి.",
        "dispose_title": "సురక్షితంగా పారవేయండి",
        "dispose": "సరైన సేకరణ సేవను ఉపయోగించండి. చెత్తను కాల్చవద్దు, ఎక్కడపడితే అక్కడ వేయవద్దు, డ్రైన్లలో పోయవద్దు.",
        "tip_label": "తెలుసుకోవడానికి ఒక చిన్న పర్యావరణ చర్యను ఎంచుకోండి",
        "tips": {
            "మళ్లీ నింపుకునే బాటిల్ వాడండి": "సురక్షితమైన తాగునీరు లభించే చోట మన్నికైన బాటిల్‌ను మళ్లీ నింపుకోండి. ఒక్కసారి వాడే బాటిళ్ల అవసరం తగ్గుతుంది.",
            "ఇంట్లోనే వ్యర్థాలను వేరు చేయండి": "ఆహార మిగతాలను శుభ్రమైన పొడి రీసైకిల్ వస్తువుల నుంచి వేరుగా ఉంచండి. మీ మున్సిపాలిటీ బిన్, సేకరణ నియమాలను పాటించండి.",
            "బ్యాటరీలను వేరుగా ఉంచండి": "బ్యాటరీలను సాధారణ ఇంటి చెత్తలో వేయవద్దు. సరైన సేకరణ కేంద్రం లేదా బాధ్యతగల పెద్దవారిని సంప్రదించండి.",
            "జంతువులను కాపాడండి": "ప్లాస్టిక్, దారాలు, ఆహార వ్యర్థాలను మూత ఉన్న డబ్బాల్లో ఉంచండి. జంతువులు వాటిని తినకుండా జాగ్రత్తపడండి.",
        },
        "button": "🌱 EcoVision AI ప్రారంభించండి",
        "no_login": "లాగిన్ లేదా అకౌంట్ అవసరం లేదు. భాష ఎంచుకుని, ఫోటో అప్‌లోడ్ చేసి ఫలితాన్ని చూడండి.",
        "footer": "తెలుసుకోండి • గుర్తించండి • వేరు చేయండి • బాధ్యతగా పారవేయండి",
    },
}
welcome = WELCOME_TEXT.get(lang, WELCOME_TEXT["en"])

if st.session_state.get("app_started", False):
    if st.sidebar.button("🏡 Welcome screen", use_container_width=True):
        st.session_state["app_started"] = False
        st.rerun()
else:
    st.markdown(
        f"""
        <div class="welcome-wrap">
          <div class="welcome-kicker">{welcome['kicker']}</div>
          <div style="display:flex; flex-wrap:wrap; align-items:center; gap:1.2rem; justify-content:space-between;">
            <div style="flex:1 1 580px;">
              <div class="welcome-title">{welcome['title']}</div>
              <div class="welcome-subtitle">{welcome['subtitle']}</div>
            </div>
            <div style="flex:0 0 135px; text-align:center;"><span class="eco-floating">🌍</span><div style="font-size:2.2rem;">🌱 ♻️</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader(welcome["how"])
    card_cols = st.columns(3)
    cards = [
        ("🌿", welcome["reduce_title"], welcome["reduce"]),
        ("♻️", welcome["separate_title"], welcome["separate"]),
        ("🌎", welcome["dispose_title"], welcome["dispose"]),
    ]
    for col, (icon, title, description) in zip(card_cols, cards):
        with col:
            st.markdown(
                f'<div class="eco-card"><div class="eco-card-icon">{icon}</div><h3>{title}</h3><p>{description}</p></div>',
                unsafe_allow_html=True,
            )
    st.write("")
    st.markdown(f"**{welcome['tip_label']}**")
    selected_tip = st.selectbox("Eco action", list(welcome["tips"].keys()), label_visibility="collapsed")
    st.info(welcome["tips"][selected_tip], icon="🌱")
    st.button(
        welcome["button"],
        type="primary",
        use_container_width=True,
        on_click=lambda: st.session_state.update({"app_started": True}),
    )
    st.markdown(f'<div class="welcome-foot">{welcome["no_login"]}<br>{welcome["footer"]}</div>', unsafe_allow_html=True)
    st.stop()

st.markdown("""
<div class="eco-hero">
  <h1>♻️ EcoVision AI</h1>
  <p>Smart waste identification • Safe disposal • PrithviPal assistant</p>
</div>
""", unsafe_allow_html=True)
st.caption(tx("tagline", lang))
st.write(tx("intro", lang))

# Load both models and labels with clear, friendly failure messages.
try:
    classifier = load_classifier()
    labels = load_labels()
except Exception:
    logger.exception("Could not load the existing Keras classifier or its labels")
    st.error("The waste classification model could not be loaded. Please check that keras_model.h5 and labels.txt are present in the repository root.")
    st.stop()

st.subheader("📷 " + tx("upload", lang))
uploaded_file = st.file_uploader(tx("upload", lang), type=["jpg", "jpeg", "png"], key="waste_photo")

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    upload_hash = hashlib.sha256(image_bytes).hexdigest()
    if len(image_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"Please upload an image smaller than {MAX_UPLOAD_MB} MB.")
    else:
        try:
            original_image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
        except Exception:
            st.error("This image could not be opened. Please choose a valid JPG or PNG image.")
            st.stop()

        st.image(original_image, caption=tx("uploaded", lang), use_container_width=True)

        if st.session_state.get("active_upload_hash") != upload_hash:
            st.session_state["active_upload_hash"] = upload_hash
            st.session_state["object_results"] = None
            st.session_state["ignored_people"] = 0

        if st.button(tx("identify", lang), type="primary", use_container_width=True):
            try:
                with st.spinner(tx("analyzing", lang)):
                    object_detector = load_object_detector()
                    object_results, ignored_people = locate_and_classify(
                        original_image, object_detector, classifier, labels
                    )
                    st.session_state["object_results"] = object_results
                    st.session_state["ignored_people"] = ignored_people
                    st.session_state["analysis_upload_hash"] = upload_hash
            except Exception:
                logger.exception("Image detection failed")
                st.error("The image could not be analysed. Please check your internet connection (the free object model downloads on first use) and try again.")
                st.session_state["object_results"] = None

        if (
            st.session_state.get("analysis_upload_hash") == upload_hash
            and st.session_state.get("object_results") is not None
        ):
            results = st.session_state["object_results"]
            ignored_people = st.session_state.get("ignored_people", 0)
            st.subheader("🌱 " + tx("result", lang))
            if ignored_people:
                st.success(f"Person detection ignored: {ignored_people}")
            if not results:
                st.info(tx("not_waste", lang))
            else:
                st.image(draw_detection_boxes(original_image, results), caption="Separate detected objects", use_container_width=True)
                st.caption(tx("model_note", lang))

                displayed_count = 0
                for index, item in enumerate(results, start=1):
                    st.markdown("---")
                    st.markdown(f"### Object {index}: {item['object_name'].replace('_', ' ').title()}")
                    left_col, right_col = st.columns([1, 2])
                    with left_col:
                        st.image(item["crop"], caption=f"Object {index} crop", use_container_width=True)
                    with right_col:
                        st.write(f"**Object detector score:** {item['object_score'] * 100:.1f}%")
                        if item["requires_confirmation"]:
                            confirmation_key = f"confirm_e_waste_{upload_hash[:10]}_{index}"
                            confirmed = st.checkbox(
                                "This electronic device is actually discarded (not in use)",
                                value=False,
                                key=confirmation_key,
                            )
                            if not confirmed:
                                st.info("A computer or electronic device is not automatically waste. Confirm only when it is genuinely being discarded.")
                                continue
                        category = item["category"]
                        score = item["category_score"]
                        if score < MIN_CATEGORY_SCORE:
                            st.warning("The category match is uncertain. Please upload a closer, clearer picture of this item.")
                            continue

                        displayed_count += 1
                        st.markdown(f"**{tx('category', lang)}:** {category_display(category, lang)}")
                        st.metric(tx("confidence", lang), f"{score * 100:.1f}%")
                        st.progress(max(0.0, min(1.0, score)))
                        if score >= HIGH_MATCH_SCORE:
                            st.success("Strong prediction match (90% or above). This is not a verified accuracy or efficiency rate.")
                        else:
                            st.caption("This percentage is the model prediction score, not a measured efficiency or test accuracy. Results below 80% are not shown as confirmed categories.")

                    if item["requires_confirmation"] and not confirmed:
                        continue
                    if item["category_score"] < MIN_CATEGORY_SCORE:
                        continue

                    category = item["category"]
                    st.markdown(f"**{tx('bio', lang)}:** {CATEGORY_INFO.get(category, CATEGORY_INFO['E-Waste & Others'])['bio']}")
                    st.markdown(f"**{tx('recycle', lang)}:** {CATEGORY_INFO.get(category, CATEGORY_INFO['E-Waste & Others'])['recycle']}")

                    st.markdown("#### 🌍 Potential environmental and safety impact — priority index")
                    scores = RISK_INDEX.get(category, RISK_INDEX["E-Waste & Others"])
                    risk_cols = st.columns(3)
                    for col, (risk_key, label) in zip(
                        risk_cols,
                        [("environment", "Environment"), ("wildlife", "Animals / wildlife"), ("people", "People")],
                    ):
                        with col:
                            col.metric(label, f"{scores[risk_key]} / 100")
                            col.caption(classify_risk_label(scores[risk_key]))
                    st.caption(risk_explanation(lang))

                    st.markdown("#### ♻️ " + tx("ideas", lang))
                    guides = DISPOSAL_GUIDES.get(category, DISPOSAL_GUIDES["E-Waste & Others"])
                    guide_titles = [get_guide_text(guide, "title", lang) for guide in guides]
                    selection_key = f"disposal_choice_{upload_hash[:10]}_{index}"
                    selected_title = st.selectbox(
                        tx("materials", lang),
                        guide_titles,
                        key=selection_key,
                    )
                    selected_guide_index = guide_titles.index(selected_title)
                    selected_guide = guides[selected_guide_index]
                    st.image(
                        make_disposal_visual(category, selected_guide["title"]),
                        caption="Visual disposal guide — illustration, not a photograph",
                        use_container_width=True,
                    )
                    st.write(get_guide_text(selected_guide, "summary", lang))
                    steps = get_guide_text(selected_guide, "steps", lang)
                    for step_number, step in enumerate(steps, start=1):
                        st.markdown(f"**{step_number}.** {step}")

                if displayed_count == 0:
                    st.info("No item passed the category match threshold. Try a closer photo, better lighting, or a less cluttered background.")

                st.caption(tx("caution", lang))

st.divider()
st.subheader("💬 " + tx("chat_title", lang))
st.write(tx("chat_intro", lang))

if "prithvi_chat" not in st.session_state:
    st.session_state["prithvi_chat"] = []

for message in st.session_state["prithvi_chat"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            st.caption("Source: local waste-safety knowledge base")

user_question = st.chat_input(tx("chat_placeholder", lang))
if user_question:
    st.session_state["prithvi_chat"].append({"role": "user", "content": user_question})
    answer, answer_source = answer_prithvipal(user_question, lang)
    st.session_state["prithvi_chat"].append({"role": "assistant", "content": answer, "source": answer_source})
    st.rerun()

if st.button("Clear PrithviPal chat"):
    st.session_state["prithvi_chat"] = []
    st.session_state.pop("prithvi_last_topic", None)
    st.rerun()

st.caption("PrithviPal uses a free, local FAQ knowledge base. It does not use a paid chatbot API and may not understand every open-ended question.")
