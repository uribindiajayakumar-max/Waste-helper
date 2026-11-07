import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from PIL import Image, ImageOps

# EcoVision - updated Streamlit app
# The included Keras model is an IMAGE CLASSIFIER (8 classes), not an object detector.
# Therefore this version classifies the uploaded image as a whole. It does not claim
# to detect/crop multiple separate objects in one image.

st.set_page_config(page_title="EcoVision", page_icon="♻️", layout="centered")

LANGUAGES = {
    "English": "en",
    "తెలుగు": "te",
    "हिन्दी": "hi",
    "தமிழ்": "ta",
    "മലയാളം": "ml",
    "ಕನ್ನಡ": "kn",
}

TEXT = {
    "en": {
        "tagline": "Waste identification • Reuse ideas • Eco FAQ",
        "intro": "Upload a waste image to get a category, handling guidance and practical reuse ideas.",
        "language": "Choose language",
        "upload": "Upload a waste image",
        "identify": "Identify waste",
        "uploaded": "Uploaded image",
        "analyzing": "Analyzing image...",
        "result": "Identification result",
        "category": "Predicted category",
        "confidence": "Model confidence",
        "bio": "Biodegradability",
        "recycle": "Recyclability",
        "bio_yes": "Usually biodegradable",
        "bio_no": "Usually non-biodegradable",
        "bio_var": "Depends on the material",
        "rec_yes": "Often recyclable if clean and accepted locally",
        "rec_no": "Usually not accepted in regular recycling",
        "rec_var": "Depends on material, condition and local facilities",
        "ideas": "Reuse / disposal ideas",
        "caution": "Guidance is general. Check local collection rules. Do not reuse hazardous or contaminated items for crafts.",
        "chat_title": "Eco FAQ Chatbot",
        "chat_intro": "Ask a question about waste sorting, recycling, composting or reuse.",
        "chat_placeholder": "Type your question...",
        "send": "Send",
        "chat_empty": "Try asking: How do I recycle plastic? or What is composting?",
        "bot_name": "Eco FAQ",
        "unknown": "I can answer common questions about waste sorting, recycling, composting and reuse. Try asking about plastic, paper, glass, metal, organic waste, e-waste or hazardous waste.",
        "not_waste": "The model could not confidently identify a supported waste category. Try a clear, close-up photo of one item.",
        "model_note": "Model limitation: this model predicts one category for the whole image. Separate detection of multiple objects needs an additional object-detection model.",
        "materials": "Simple idea",
    },
    "te": {
        "tagline": "వ్యర్థాల గుర్తింపు • మళ్లీ వినియోగ ఆలోచనలు • Eco FAQ",
        "intro": "వ్యర్థాల ఫోటోను అప్‌లోడ్ చేసి దాని వర్గం, నిర్వహణ సూచనలు, మళ్లీ వినియోగ ఆలోచనలు తెలుసుకోండి.",
        "language": "భాషను ఎంచుకోండి",
        "upload": "వ్యర్థాల ఫోటోను అప్‌లోడ్ చేయండి",
        "identify": "వ్యర్థాన్ని గుర్తించండి",
        "uploaded": "అప్‌లోడ్ చేసిన చిత్రం",
        "analyzing": "చిత్రాన్ని పరిశీలిస్తోంది...",
        "result": "గుర్తింపు ఫలితం",
        "category": "అంచనా వేసిన వర్గం",
        "confidence": "మోడల్ నమ్మక స్థాయి",
        "bio": "జీవ విచ్ఛిన్నత",
        "recycle": "రీసైక్లింగ్",
        "bio_yes": "సాధారణంగా జీవ విచ్ఛిన్నమవుతుంది",
        "bio_no": "సాధారణంగా జీవ విచ్ఛిన్నం కాదు",
        "bio_var": "పదార్థంపై ఆధారపడి ఉంటుంది",
        "rec_yes": "శుభ్రంగా ఉంటే, స్థానికంగా అంగీకరిస్తే రీసైకిల్ చేయవచ్చు",
        "rec_no": "సాధారణ రీసైక్లింగ్‌లో సాధారణంగా అంగీకరించరు",
        "rec_var": "పదార్థం, స్థితి, స్థానిక సదుపాయాలపై ఆధారపడి ఉంటుంది",
        "ideas": "మళ్లీ వినియోగం / పారవేత ఆలోచనలు",
        "caution": "ఇవి సాధారణ సూచనలు మాత్రమే. స్థానిక నియమాలు తెలుసుకోండి. ప్రమాదకరమైన లేదా కలుషితమైన వస్తువులతో DIY చేయవద్దు.",
        "chat_title": "Eco FAQ చాట్‌బాట్",
        "chat_intro": "వ్యర్థాల వర్గీకరణ, రీసైక్లింగ్, కంపోస్టింగ్ లేదా మళ్లీ వినియోగం గురించి అడగండి.",
        "chat_placeholder": "మీ ప్రశ్న టైప్ చేయండి...",
        "send": "పంపండి",
        "chat_empty": "ఉదాహరణ: ప్లాస్టిక్‌ను ఎలా రీసైకిల్ చేయాలి?",
        "bot_name": "Eco FAQ",
        "unknown": "వ్యర్థాల వర్గీకరణ, రీసైక్లింగ్, కంపోస్టింగ్, మళ్లీ వినియోగం గురించి సాధారణ ప్రశ్నలకు సమాధానం ఇవ్వగలను. ప్లాస్టిక్, కాగితం, గాజు, లోహం, సేంద్రీయ వ్యర్థాలు లేదా ఈ-వ్యర్థాల గురించి అడగండి.",
        "not_waste": "మద్దతు ఉన్న వ్యర్థ వర్గాన్ని మోడల్ స్పష్టంగా గుర్తించలేకపోయింది. ఒక వస్తువు దగ్గరగా కనిపించే ఫోటోను ప్రయత్నించండి.",
        "model_note": "మోడల్ పరిమితి: ఇది మొత్తం చిత్రానికి ఒక వర్గాన్ని మాత్రమే అంచనా వేస్తుంది. అనేక వస్తువులను విడిగా గుర్తించడానికి అదనపు object-detection మోడల్ అవసరం.",
        "materials": "సులభమైన ఆలోచన",
    },
    "hi": {
        "tagline": "कचरा पहचान • पुन: उपयोग के सुझाव • Eco FAQ",
        "intro": "कचरे की तस्वीर अपलोड करें और उसकी श्रेणी, निपटान मार्गदर्शन तथा पुन: उपयोग के सुझाव पाएँ।",
        "language": "भाषा चुनें", "upload": "कचरे की तस्वीर अपलोड करें", "identify": "कचरे की पहचान करें",
        "uploaded": "अपलोड की गई तस्वीर", "analyzing": "तस्वीर का विश्लेषण हो रहा है...", "result": "पहचान परिणाम",
        "category": "अनुमानित श्रेणी", "confidence": "मॉडल का विश्वास स्तर", "bio": "जैव अपघटन",
        "recycle": "रीसाइक्लिंग", "bio_yes": "आमतौर पर जैव अपघटनीय", "bio_no": "आमतौर पर जैव अपघटनीय नहीं",
        "bio_var": "सामग्री पर निर्भर करता है", "rec_yes": "साफ़ हो और स्थानीय केंद्र स्वीकार करे तो अक्सर रीसायकल हो सकता है",
        "rec_no": "आमतौर पर सामान्य रीसाइक्लिंग में स्वीकार नहीं होता", "rec_var": "सामग्री, स्थिति और स्थानीय सुविधा पर निर्भर",
        "ideas": "पुन: उपयोग / निपटान के सुझाव", "caution": "ये सामान्य सुझाव हैं। स्थानीय नियम जाँचें। खतरनाक या दूषित वस्तुओं से DIY न करें।",
        "chat_title": "Eco FAQ चैटबॉट", "chat_intro": "कचरा छँटाई, रीसाइक्लिंग, कम्पोस्ट या पुन: उपयोग के बारे में पूछें।",
        "chat_placeholder": "अपना सवाल लिखें...", "send": "भेजें", "chat_empty": "पूछें: प्लास्टिक को कैसे रीसायकल करें?",
        "bot_name": "Eco FAQ", "unknown": "मैं कचरा छँटाई, रीसाइक्लिंग, कम्पोस्ट और पुन: उपयोग के सामान्य सवालों का जवाब दे सकता हूँ।",
        "not_waste": "मॉडल किसी समर्थित कचरा श्रेणी की स्पष्ट पहचान नहीं कर पाया। एक वस्तु की साफ़, नज़दीकी तस्वीर लें।",
        "model_note": "मॉडल सीमा: यह पूरी तस्वीर के लिए केवल एक श्रेणी बताता है। कई वस्तुओं की अलग पहचान के लिए अतिरिक्त object-detection मॉडल चाहिए।",
        "materials": "सरल सुझाव",
    },
    "ta": {
        "tagline": "கழிவு அடையாளம் • மறுபயன்பாட்டு யோசனைகள் • Eco FAQ",
        "intro": "கழிவுப் படத்தைப் பதிவேற்றி அதன் வகை, கையாளும் வழிகாட்டுதல் மற்றும் மறுபயன்பாட்டு யோசனைகளைப் பெறுங்கள்.",
        "language": "மொழியைத் தேர்ந்தெடுக்கவும்", "upload": "கழிவுப் படத்தைப் பதிவேற்றவும்", "identify": "கழிவை அடையாளம் காண்க",
        "uploaded": "பதிவேற்றிய படம்", "analyzing": "படம் ஆய்வு செய்யப்படுகிறது...", "result": "அடையாள முடிவு",
        "category": "கணிக்கப்பட்ட வகை", "confidence": "மாதிரியின் நம்பிக்கை", "bio": "மக்கும் தன்மை",
        "recycle": "மறுசுழற்சி", "bio_yes": "பொதுவாக மக்கும்", "bio_no": "பொதுவாக மக்காதது", "bio_var": "பொருளைப் பொறுத்தது",
        "rec_yes": "சுத்தமாகவும் உள்ளூர் மையம் ஏற்றுக்கொண்டாலும் மறுசுழற்சி செய்யலாம்",
        "rec_no": "பொதுவாக வழக்கமான மறுசுழற்சியில் ஏற்கப்படாது", "rec_var": "பொருள், நிலை மற்றும் உள்ளூர் வசதிகளைப் பொறுத்தது",
        "ideas": "மறுபயன்பாடு / அகற்றும் யோசனைகள்", "caution": "இவை பொதுவான ஆலோசனைகள். உள்ளூர் விதிகளைச் சரிபார்க்கவும். ஆபத்தான பொருட்களை DIY-க்கு பயன்படுத்த வேண்டாம்.",
        "chat_title": "Eco FAQ உரையாடல்", "chat_intro": "கழிவு பிரித்தல், மறுசுழற்சி, உரமிடுதல் அல்லது மறுபயன்பாடு பற்றி கேளுங்கள்.",
        "chat_placeholder": "உங்கள் கேள்வியை உள்ளிடவும்...", "send": "அனுப்பு", "chat_empty": "உதாரணம்: பிளாஸ்டிக்கை எப்படி மறுசுழற்சி செய்வது?",
        "bot_name": "Eco FAQ", "unknown": "கழிவு பிரித்தல், மறுசுழற்சி, உரமிடுதல் மற்றும் மறுபயன்பாடு குறித்த பொதுவான கேள்விகளுக்கு பதிலளிக்க முடியும்.",
        "not_waste": "ஆதரிக்கப்படும் கழிவு வகையை மாதிரி தெளிவாகக் கண்டறியவில்லை. ஒரு பொருளின் தெளிவான நெருக்கமான படத்தை முயற்சிக்கவும்.",
        "model_note": "மாதிரி வரம்பு: முழுப் படத்திற்கும் ஒரு வகையை மட்டுமே கணிக்கிறது. பல பொருட்களைத் தனித்தனியாகக் கண்டறிய கூடுதல் object-detection மாதிரி தேவை.",
        "materials": "எளிய யோசனை",
    },
    "ml": {
        "tagline": "മാലിന്യ തിരിച്ചറിയൽ • പുനരുപയോഗ ആശയങ്ങൾ • Eco FAQ",
        "intro": "മാലിന്യത്തിന്റെ ചിത്രം അപ്‌ലോഡ് ചെയ്ത് വിഭാഗവും കൈകാര്യം ചെയ്യാനുള്ള നിർദ്ദേശങ്ങളും പുനരുപയോഗ ആശയങ്ങളും അറിയുക.",
        "language": "ഭാഷ തിരഞ്ഞെടുക്കുക", "upload": "മാലിന്യ ചിത്രം അപ്‌ലോഡ് ചെയ്യുക", "identify": "മാലിന്യം തിരിച്ചറിയുക",
        "uploaded": "അപ്‌ലോഡ് ചെയ്ത ചിത്രം", "analyzing": "ചിത്രം പരിശോധിക്കുന്നു...", "result": "തിരിച്ചറിയൽ ഫലം",
        "category": "കണക്കാക്കിയ വിഭാഗം", "confidence": "മോഡലിന്റെ വിശ്വാസനില", "bio": "ജൈവ വിഘടനം",
        "recycle": "റീസൈക്ലിംഗ്", "bio_yes": "സാധാരണയായി ജൈവമായി വിഘടിക്കും", "bio_no": "സാധാരണയായി ജൈവമായി വിഘടിക്കില്ല",
        "bio_var": "വസ്തുവിനെ ആശ്രയിച്ചിരിക്കുന്നു", "rec_yes": "വൃത്തിയുള്ളതും പ്രാദേശിക കേന്ദ്രം സ്വീകരിക്കുന്നതുമെങ്കിൽ റീസൈക്കിൾ ചെയ്യാം",
        "rec_no": "സാധാരണ റീസൈക്ലിംഗിൽ സാധാരണയായി സ്വീകരിക്കില്ല", "rec_var": "വസ്തു, അവസ്ഥ, പ്രാദേശിക സൗകര്യങ്ങൾ എന്നിവയെ ആശ്രയിക്കുന്നു",
        "ideas": "പുനരുപയോഗം / നീക്കംചെയ്യൽ ആശയങ്ങൾ", "caution": "ഇവ പൊതുവായ നിർദ്ദേശങ്ങളാണ്. പ്രാദേശിക നിയമങ്ങൾ പരിശോധിക്കുക. അപകടകരമായ വസ്തുക്കൾ DIY-യ്ക്ക് ഉപയോഗിക്കരുത്.",
        "chat_title": "Eco FAQ ചാറ്റ്ബോട്ട്", "chat_intro": "മാലിന്യ വേർതിരിക്കൽ, റീസൈക്ലിംഗ്, കമ്പോസ്റ്റിംഗ്, പുനരുപയോഗം എന്നിവ ചോദിക്കാം.",
        "chat_placeholder": "ചോദ്യം ടൈപ്പ് ചെയ്യുക...", "send": "അയയ്ക്കുക", "chat_empty": "ഉദാഹരണം: പ്ലാസ്റ്റിക് എങ്ങനെ റീസൈക്കിൾ ചെയ്യാം?",
        "bot_name": "Eco FAQ", "unknown": "മാലിന്യ വേർതിരിക്കൽ, റീസൈക്ലിംഗ്, കമ്പോസ്റ്റിംഗ്, പുനരുപയോഗം എന്നിവയെക്കുറിച്ചുള്ള പൊതുവായ ചോദ്യങ്ങൾക്ക് മറുപടി നൽകാം.",
        "not_waste": "പിന്തുണയ്ക്കുന്ന മാലിന്യ വിഭാഗം മോഡലിന് വ്യക്തമായി തിരിച്ചറിയാനായില്ല. ഒരു വസ്തുവിന്റെ വ്യക്തമായ അടുത്ത ചിത്രം ഉപയോഗിക്കുക.",
        "model_note": "മോഡലിന്റെ പരിമിതി: മുഴുവൻ ചിത്രത്തിനും ഒരു വിഭാഗം മാത്രമാണ് പ്രവചിക്കുന്നത്. ഒന്നിലധികം വസ്തുക്കൾ വേർതിരിച്ച് കണ്ടെത്താൻ മറ്റൊരു object-detection മോഡൽ വേണം.",
        "materials": "ലളിതമായ ആശയം",
    },
    "kn": {
        "tagline": "ತ್ಯಾಜ್ಯ ಗುರುತಿಸುವಿಕೆ • ಮರುಬಳಕೆ ಆಲೋಚನೆಗಳು • Eco FAQ",
        "intro": "ತ್ಯಾಜ್ಯದ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ ಅದರ ವರ್ಗ, ನಿರ್ವಹಣೆ ಮಾರ್ಗದರ್ಶನ ಮತ್ತು ಮರುಬಳಕೆ ಆಲೋಚನೆಗಳನ್ನು ಪಡೆಯಿರಿ.",
        "language": "ಭಾಷೆ ಆಯ್ಕೆಮಾಡಿ", "upload": "ತ್ಯಾಜ್ಯದ ಚಿತ್ರ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ", "identify": "ತ್ಯಾಜ್ಯ ಗುರುತಿಸಿ",
        "uploaded": "ಅಪ್‌ಲೋಡ್ ಮಾಡಿದ ಚಿತ್ರ", "analyzing": "ಚಿತ್ರವನ್ನು ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ...", "result": "ಗುರುತಿಸುವಿಕೆಯ ಫಲಿತಾಂಶ",
        "category": "ಅಂದಾಜು ವರ್ಗ", "confidence": "ಮಾದರಿಯ ವಿಶ್ವಾಸ ಮಟ್ಟ", "bio": "ಜೈವಿಕವಾಗಿ ಕೊಳೆಯುವಿಕೆ",
        "recycle": "ಮರುಬಳಕೆ", "bio_yes": "ಸಾಮಾನ್ಯವಾಗಿ ಜೈವಿಕವಾಗಿ ಕೊಳೆಯುತ್ತದೆ", "bio_no": "ಸಾಮಾನ್ಯವಾಗಿ ಜೈವಿಕವಾಗಿ ಕೊಳೆಯುವುದಿಲ್ಲ",
        "bio_var": "ವಸ್ತುವಿನ ಮೇಲೆ ಅವಲಂಬಿತವಾಗಿದೆ", "rec_yes": "ಸ್ವಚ್ಛವಾಗಿದ್ದು ಸ್ಥಳೀಯ ಕೇಂದ್ರ ಸ್ವೀಕರಿಸಿದರೆ ಮರುಬಳಕೆ ಸಾಧ್ಯ",
        "rec_no": "ಸಾಮಾನ್ಯ ಮರುಬಳಕೆಯಲ್ಲಿ ಸಾಮಾನ್ಯವಾಗಿ ಸ್ವೀಕರಿಸುವುದಿಲ್ಲ", "rec_var": "ವಸ್ತು, ಸ್ಥಿತಿ ಮತ್ತು ಸ್ಥಳೀಯ ಸೌಲಭ್ಯಗಳ ಮೇಲೆ ಅವಲಂಬಿತ",
        "ideas": "ಮರುಬಳಕೆ / ವಿಲೇವಾರಿ ಆಲೋಚನೆಗಳು", "caution": "ಇವು ಸಾಮಾನ್ಯ ಸಲಹೆಗಳು. ಸ್ಥಳೀಯ ನಿಯಮಗಳನ್ನು ಪರಿಶೀಲಿಸಿ. ಅಪಾಯಕಾರಿ ವಸ್ತುಗಳನ್ನು DIYಗೆ ಬಳಸಬೇಡಿ.",
        "chat_title": "Eco FAQ ಚಾಟ್‌ಬಾಟ್", "chat_intro": "ತ್ಯಾಜ್ಯ ವಿಂಗಡಣೆ, ಮರುಬಳಕೆ, ಕಾಂಪೋಸ್ಟ್ ಅಥವಾ ಮರುಉಪಯೋಗದ ಬಗ್ಗೆ ಕೇಳಿ.",
        "chat_placeholder": "ನಿಮ್ಮ ಪ್ರಶ್ನೆ ಬರೆಯಿರಿ...", "send": "ಕಳುಹಿಸಿ", "chat_empty": "ಉದಾಹರಣೆ: ಪ್ಲಾಸ್ಟಿಕ್ ಅನ್ನು ಹೇಗೆ ಮರುಬಳಕೆ ಮಾಡುವುದು?",
        "bot_name": "Eco FAQ", "unknown": "ತ್ಯಾಜ್ಯ ವಿಂಗಡಣೆ, ಮರುಬಳಕೆ, ಕಾಂಪೋಸ್ಟ್ ಮತ್ತು ಮರುಉಪಯೋಗದ ಸಾಮಾನ್ಯ ಪ್ರಶ್ನೆಗಳಿಗೆ ಉತ್ತರಿಸಬಹುದು.",
        "not_waste": "ಬೆಂಬಲಿತ ತ್ಯಾಜ್ಯ ವರ್ಗವನ್ನು ಮಾದರಿ ಸ್ಪಷ್ಟವಾಗಿ ಗುರುತಿಸಲಿಲ್ಲ. ಒಂದು ವಸ್ತುವಿನ ಸ್ಪಷ್ಟ ಹತ್ತಿರದ ಚಿತ್ರವನ್ನು ಪ್ರಯತ್ನಿಸಿ.",
        "model_note": "ಮಾದರಿಯ ಮಿತಿ: ಇದು ಸಂಪೂರ್ಣ ಚಿತ್ರಕ್ಕೆ ಒಂದು ವರ್ಗವನ್ನು ಮಾತ್ರ ಊಹಿಸುತ್ತದೆ. ಹಲವು ವಸ್ತುಗಳನ್ನು ಪ್ರತ್ಯೇಕವಾಗಿ ಗುರುತಿಸಲು ಹೆಚ್ಚುವರಿ object-detection ಮಾದರಿ ಬೇಕು.",
        "materials": "ಸರಳ ಆಲೋಚನೆ",
    },
}

CATEGORY_NAMES = {
    "Plastic": {"te": "ప్లాస్టిక్", "hi": "प्लास्टिक", "ta": "பிளாஸ்டிக்", "ml": "പ്ലാസ്റ്റിക്", "kn": "ಪ್ಲಾಸ್ಟಿಕ್"},
    "Paper & Cardboard": {"te": "కాగితం & కార్డ్‌బోర్డ్", "hi": "कागज़ और कार्डबोर्ड", "ta": "காகிதம் மற்றும் அட்டை", "ml": "പേപ്പർ & കാർഡ്ബോർഡ്", "kn": "ಕಾಗದ ಮತ್ತು ಕಾರ್ಡ್‌ಬೋರ್ಡ್"},
    "Organic Waste": {"te": "సేంద్రీయ వ్యర్థాలు", "hi": "जैविक कचरा", "ta": "கரிமக் கழிவு", "ml": "ജൈവ മാലിന്യം", "kn": "ಸಾವಯವ ತ್ಯಾಜ್ಯ"},
    "Glass": {"te": "గాజు", "hi": "काँच", "ta": "கண்ணாடி", "ml": "ഗ്ലാസ്", "kn": "ಗಾಜು"},
    "Metal & Scrap": {"te": "లోహం & స్క్రాప్", "hi": "धातु और कबाड़", "ta": "உலோகம் மற்றும் பழைய பொருட்கள்", "ml": "ലോഹം & സ്ക്രാപ്പ്", "kn": "ಲೋಹ ಮತ್ತು ಸ್ಕ್ರ್ಯಾಪ್"},
    "Hazardous Waste": {"te": "ప్రమాదకర వ్యర్థాలు", "hi": "खतरनाक कचरा", "ta": "அபாயகரமான கழிவு", "ml": "അപകടകരമായ മാലിന്യം", "kn": "ಅಪಾಯಕಾರಿ ತ್ಯಾಜ್ಯ"},
    "Textile & Fabric Waste": {"te": "వస్త్రాలు & ఫాబ్రిక్", "hi": "कपड़ा", "ta": "துணி", "ml": "തുണി", "kn": "ಜವಳಿ ಮತ್ತು ಬಟ್ಟೆ"},
    "E-Waste & Others": {"te": "ఈ-వ్యర్థాలు & ఇతరాలు", "hi": "ई-कचरा और अन्य", "ta": "மின்னணுக் கழிவு மற்றும் பிற", "ml": "ഇ-മാലിന്യം & മറ്റുള്ളവ", "kn": "ಇ-ತ್ಯಾಜ್ಯ ಮತ್ತು ಇತರೆ"},
}

IDEAS = {
    "Plastic": {
        "en": ["Reuse a clean bottle as a small plant-watering container.", "Turn a clean, safe plastic container into a desk organiser.", "Use a sturdy bottle as a drip-irrigation bottle for a garden pot."],
        "te": ["శుభ్రమైన బాటిల్‌ను మొక్కలకు నీళ్లు పోసే చిన్న పాత్రగా వాడండి.", "శుభ్రమైన ప్లాస్టిక్ డబ్బాను పెన్ స్టాండ్‌గా మార్చండి.", "తోటలో కుండకు నెమ్మదిగా నీరు అందించే బాటిల్‌గా వాడండి."],
    },
    "Paper & Cardboard": {
        "en": ["Reuse clean cardboard as drawer dividers or storage organisers.", "Use blank paper on the reverse side for notes.", "Flatten clean cardboard and send it to a paper recycler."],
        "te": ["శుభ్రమైన కార్డ్‌బోర్డ్‌తో డ్రాయర్ విభజకాలు తయారు చేయండి.", "ఖాళీ వైపు ఉన్న కాగితాన్ని నోట్స్ కోసం వాడండి.", "శుభ్రమైన కార్డ్‌బోర్డ్‌ను మడిచి పేపర్ రీసైక్లింగ్‌కు ఇవ్వండి."],
    },
    "Organic Waste": {
        "en": ["Compost suitable fruit and vegetable scraps in a ventilated compost bin.", "Use dry leaves as a brown layer in a compost bin.", "Keep meat, dairy and oily food out of a simple home compost bin."],
        "te": ["పండ్లు, కూరగాయల మిగతాలను గాలి వచ్చే కంపోస్ట్ బిన్‌లో కంపోస్ట్ చేయండి.", "ఎండిన ఆకులను కంపోస్ట్‌లో పొడి పదార్థంగా కలపండి.", "సాధారణ ఇంటి కంపోస్ట్‌లో మాంసం, పాల పదార్థాలు, నూనె ఎక్కువగా ఉన్న ఆహారం వేయవద్దు."],
    },
    "Glass": {
        "en": ["Reuse an intact, clean jar for storage if it is food-safe.", "Use an intact jar as a desk organiser.", "Keep broken glass separate, wrapped securely, and follow local disposal rules; do not craft with it."],
        "te": ["పగలని, శుభ్రమైన జార్‌ను సురక్షిత నిల్వకు వాడండి.", "పగలని జార్‌ను చిన్న వస్తువుల ఆర్గనైజర్‌గా వాడండి.", "పగిలిన గాజును జాగ్రత్తగా వేరుగా ఉంచి స్థానిక పారవేత నియమాలు పాటించండి; DIY చేయవద్దు."],
    },
    "Metal & Scrap": {
        "en": ["Reuse a clean, smooth tin as a pen holder after removing sharp edges safely.", "Use a clean metal container as a small storage organiser.", "Send sorted metal scrap to an authorised recycler."],
        "te": ["పదునైన అంచులు లేని శుభ్రమైన టిన్‌ను పెన్ హోల్డర్‌గా వాడండి.", "శుభ్రమైన లోహపు డబ్బాను చిన్న వస్తువుల నిల్వకు వాడండి.", "వేరు చేసిన లోహ స్క్రాప్‌ను అనుమతిపొందిన రీసైక్లర్‌కు ఇవ్వండి."],
    },
    "Hazardous Waste": {
        "en": ["Do not make DIY crafts from batteries, chemicals, paint, or medical waste.", "Keep hazardous items in their original labelled packaging where possible.", "Take them to an authorised hazardous-waste or e-waste collection point."],
        "te": ["బ్యాటరీలు, రసాయనాలు, పెయింట్ లేదా వైద్య వ్యర్థాలతో DIY వస్తువులు తయారు చేయవద్దు.", "సాధ్యమైనంతవరకు వాటిని అసలు లేబుల్ ఉన్న ప్యాకింగ్‌లోనే ఉంచండి.", "అనుమతిపొందిన ప్రమాదకర వ్యర్థాల సేకరణ కేంద్రానికి ఇవ్వండి."],
    },
    "Textile & Fabric Waste": {
        "en": ["Reuse clean fabric as cleaning cloths.", "Turn intact fabric pieces into simple reusable gift-wrap or pouch material.", "Donate wearable, clean clothes where accepted."],
        "te": ["శుభ్రమైన పాత బట్టలను తుడిచే గుడ్డలుగా వాడండి.", "మంచి స్థితిలో ఉన్న బట్ట ముక్కలతో చిన్న సంచి తయారు చేయండి.", "ధరించగలిగే శుభ్రమైన దుస్తులను అవసరమైన వారికి ఇవ్వండి."],
    },
    "E-Waste & Others": {
        "en": ["Do not open batteries, screens or electronic devices for DIY.", "Keep small electronics and cables dry and separate.", "Use an authorised e-waste collection or take-back centre."],
        "te": ["బ్యాటరీలు, స్క్రీన్లు లేదా ఎలక్ట్రానిక్ పరికరాలను తెరిచి DIY చేయవద్దు.", "చిన్న ఎలక్ట్రానిక్ వస్తువులు, కేబుళ్లను పొడిగా వేరుగా ఉంచండి.", "అనుమతిపొందిన ఈ-వ్యర్థాల సేకరణ కేంద్రానికి ఇవ్వండి."],
    },
}

FAQ = {
    "plastic": {
        "en": "Keep plastic items clean and dry, check the resin/recycling mark if available, and follow your local recycler's rules. Not every plastic type is accepted everywhere.",
        "te": "ప్లాస్టిక్ వస్తువులను శుభ్రంగా, పొడిగా ఉంచండి. రీసైక్లింగ్ గుర్తు ఉంటే చూడండి; స్థానిక రీసైక్లర్ నియమాలు పాటించండి. అన్ని రకాల ప్లాస్టిక్‌ను ప్రతి చోటా అంగీకరించరు.",
    },
    "paper": {
        "en": "Keep paper and cardboard dry and free from food or oil. Flatten cardboard before placing it in paper recycling.",
        "te": "కాగితం, కార్డ్‌బోర్డ్‌ను తడవకుండా, ఆహారం లేదా నూనె అంటకుండా ఉంచండి. రీసైక్లింగ్‌కు ఇచ్చే ముందు కార్డ్‌బోర్డ్‌ను మడవండి.",
    },
    "glass": {
        "en": "Separate glass from other waste. Do not handle broken glass with bare hands; wrap and label it according to local collection guidance.",
        "te": "గాజును ఇతర వ్యర్థాల నుంచి వేరుగా ఉంచండి. పగిలిన గాజును చేతులతో తాకవద్దు; స్థానిక సూచనల ప్రకారం సురక్షితంగా చుట్టి ఇవ్వండి.",
    },
    "metal": {
        "en": "Separate clean metal cans and scrap. Sharp or rusty pieces should not be used for crafts; use an authorised collection point.",
        "te": "శుభ్రమైన లోహపు క్యాన్లు, స్క్రాప్‌ను వేరుగా ఉంచండి. పదునైన లేదా తుప్పుపట్టిన ముక్కలతో DIY చేయవద్దు; అనుమతిపొందిన సేకరణ కేంద్రానికి ఇవ్వండి.",
    },
    "organic": {
        "en": "Fruit and vegetable scraps, coffee grounds and dry leaves can often be composted. Avoid meat, dairy and oily food in a basic home compost bin.",
        "te": "పండ్లు, కూరగాయల మిగతాలు, కాఫీ పొడి, ఎండిన ఆకులను సాధారణంగా కంపోస్ట్ చేయవచ్చు. ఇంటి సాధారణ కంపోస్ట్‌లో మాంసం, పాల పదార్థాలు, నూనె ఎక్కువగా ఉన్న ఆహారం వేయవద్దు.",
    },
    "compost": {
        "en": "Composting is the controlled breakdown of suitable organic material into a soil-like amendment. Balance moist green scraps with dry brown material and allow airflow.",
        "te": "కంపోస్టింగ్ అంటే సరైన సేంద్రీయ పదార్థాలు కుళ్లి మట్టిలాంటి పదార్థంగా మారే ప్రక్రియ. తడి ఆకుపచ్చ వ్యర్థాలకు పొడి ఆకులు వంటి పదార్థాలు కలిపి గాలి వచ్చేలా ఉంచండి.",
    },
    "hazard": {
        "en": "Do not reuse chemicals, batteries, medical waste or unknown containers for crafts. Keep them separate and use an authorised collection facility.",
        "te": "రసాయనాలు, బ్యాటరీలు, వైద్య వ్యర్థాలు లేదా తెలియని డబ్బాలను DIY కోసం వాడవద్దు. వేరుగా ఉంచి అనుమతిపొందిన సేకరణ కేంద్రానికి ఇవ్వండి.",
    },
    "ewaste": {
        "en": "Do not dismantle electronics or batteries. Keep e-waste dry and take it to an authorised e-waste collection or manufacturer take-back point.",
        "te": "ఎలక్ట్రానిక్ వస్తువులు లేదా బ్యాటరీలను విడదీయవద్దు. ఈ-వ్యర్థాలను పొడిగా ఉంచి అనుమతిపొందిన సేకరణ కేంద్రానికి ఇవ్వండి.",
    },
    "biodegradable": {
        "en": "Biodegradable materials can be broken down by microorganisms under suitable conditions. Some items need industrial facilities, so the label alone does not mean home-compostable.",
        "te": "సరైన పరిస్థితుల్లో సూక్ష్మజీవులు విచ్ఛిన్నం చేయగల పదార్థాలను జీవ విచ్ఛిన్న పదార్థాలు అంటారు. కొన్ని వస్తువులకు పారిశ్రామిక సదుపాయాలు అవసరం; కాబట్టి ఇంట్లో కంపోస్ట్ చేయవచ్చని అనుకోవద్దు.",
    },
}

def get_text(key, lang):
    return TEXT.get(lang, TEXT["en"]).get(key, TEXT["en"].get(key, key))

@st.cache_data
def load_labels():
    with open("labels.txt", "r", encoding="utf-8") as f:
        labels = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            labels.append(parts[1] if len(parts) == 2 and parts[0].isdigit() else line)
        return labels

class CompatibleDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    @classmethod
    def from_config(cls, config):
        config.pop("groups", None)
        return super().from_config(config)

@st.cache_resource
def load_waste_model():
    return tf.keras.models.load_model(
        "keras_model.h5",
        custom_objects={
            "DepthwiseConv2D": CompatibleDepthwiseConv2D,
            "CompatibleDepthwiseConv2D": CompatibleDepthwiseConv2D,
        },
        compile=False,
    )

def predict_waste(image, model, labels):
    image = image.convert("RGB")
    shape = model.input_shape
    height, width = shape[1] or 224, shape[2] or 224
    prepared = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
    arr = np.asarray(prepared, dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    prediction = np.asarray(model.predict(np.expand_dims(arr, axis=0), verbose=0)).flatten()
    idx = int(np.argmax(prediction))
    label = labels[idx] if idx < len(labels) else f"Class {idx + 1}"
    top_indices = np.argsort(prediction)[::-1][:3]
    top_predictions = [
        (labels[int(i)] if int(i) < len(labels) else f"Class {int(i) + 1}", float(prediction[int(i)]))
        for i in top_indices
    ]
    return label, float(prediction[idx]), top_predictions

def category_guidance(category, lang):
    c = category.lower()
    if "organic" in c:
        bio, rec = "bio_yes", "rec_var"
    elif "paper" in c:
        bio, rec = "bio_yes", "rec_yes"
    elif "plastic" in c:
        bio, rec = "bio_no", "rec_var"
    elif "glass" in c or "metal" in c:
        bio, rec = "bio_no", "rec_var"
    elif "textile" in c or "fabric" in c:
        bio, rec = "rec_var", "rec_var"
    else:
        bio, rec = "bio_var", "rec_var"
    return get_text(bio, lang), get_text(rec, lang)

def get_ideas(category, lang):
    source = IDEAS.get(category, IDEAS["E-Waste & Others"])
    return source.get(lang, source.get("en", []))

def faq_answer(question, lang):
    q = question.lower()
    keys = [
        (["plastic", "ప్లాస్టిక్", "प्लास्टिक", "பிளாஸ்டிக்", "പ്ലാസ്റ്റിക്", "ಪ್ಲಾಸ್ಟಿಕ್"], "plastic"),
        (["paper", "cardboard", "కాగితం", "కార్డ్", "कागज़", "காகித", "പേപ്പർ", "ಕಾಗದ"], "paper"),
        (["glass", "గాజు", "काँच", "கண்ணாடி", "ഗ്ലാസ്", "ಗಾಜು"], "glass"),
        (["metal", "can", "లోహ", "धातु", "உலோகம்", "ലോഹ", "ಲೋಹ"], "metal"),
        (["organic", "food waste", "కంపోస్ట్", "సేంద్రీయ", "जैविक", "கரிம", "ജൈവ", "ಸಾವಯವ"], "organic"),
        (["compost", "కంపోస్ట", "कम्पोस्ट", "உரம்", "കമ്പോസ്റ്റ്", "ಕಾಂಪೋಸ್ಟ್"], "compost"),
        (["hazard", "battery", "chemical", "ప్రమాదకర", "బ్యాటరీ", "खतरनाक", "அபாய", "അപകട", "ಅಪಾಯಕಾರಿ"], "hazard"),
        (["e-waste", "ewaste", "electronic", "ఈ-వ్యర్థ", "ई-कचरा", "மின்னணுக்", "ഇ-മാലിന്യം", "ಇ-ತ್ಯಾಜ್ಯ"], "ewaste"),
        (["biodegradable", "జీవ విచ్ఛిన్న", "जैव अपघटनीय", "மக்கும்", "ജൈവ വിഘട", "ಜೈವಿಕವಾಗಿ"], "biodegradable"),
    ]
    for words, key in keys:
        if any(word in q for word in words):
            return FAQ[key].get(lang, FAQ[key]["en"])
    return get_text("unknown", lang)


def detect_faq_topic(question):
    q = question.lower()
    topic_terms = {
        "plastic": ["plastic", "ప్లాస్టిక్", "प्लास्टिक", "பிளாஸ்டிக்", "പ്ലാസ്റ്റിക്", "ಪ್ಲಾಸ್ಟಿಕ್"],
        "paper": ["paper", "cardboard", "కాగితం", "కార్డ్", "कागज़", "காகித", "പേപ്പർ", "ಕಾಗದ"],
        "glass": ["glass", "గాజు", "काँच", "கண்ணாடி", "ഗ്ലാസ്", "ಗಾಜು"],
        "metal": ["metal", "can", "లోహ", "धातु", "உலோகம்", "ലോഹ", "ಲೋಹ"],
        "organic": ["organic", "food waste", "సేంద్రీయ", "जैविक", "கரிம", "ജൈവ", "ಸಾವಯವ"],
        "compost": ["compost", "కంపోస్ట", "कम्पोस्ट", "உரம்", "കമ്പോസ്റ്റ്", "ಕಾಂಪೋಸ್ಟ್"],
        "hazard": ["hazard", "battery", "chemical", "ప్రమాదకర", "బ్యాటరీ", "खतरनाक", "அபாய", "അപകട", "ಅಪಾಯಕಾರಿ"],
        "ewaste": ["e-waste", "ewaste", "electronic", "ఈ-వ్యర్థ", "ई-कचरा", "மின்னணுக்", "ഇ-മാലിന്യം", "ಇ-ತ್ಯಾಜ್ಯ"],
        "biodegradable": ["biodegradable", "జీవ విచ్ఛిన్న", "जैव अपघटनीय", "மக்கும்", "ജൈവ വിഘട", "ಜೈವಿಕವಾಗಿ"],
    }
    for topic, words in topic_terms.items():
        if any(word in q for word in words):
            return topic
    return None


def faq_topic_answer(topic, lang):
    return FAQ.get(topic, {}).get(lang, FAQ.get(topic, {}).get("en", get_text("unknown", lang)))


def get_optional_ai_settings():
    """Read an optional OpenAI-compatible endpoint from Streamlit secrets.
    No API key is required for the default free FAQ mode.
    """
    try:
        return {
            "api_url": st.secrets.get("ECO_CHAT_API_URL", ""),
            "api_key": st.secrets.get("ECO_CHAT_API_KEY", ""),
            "model": st.secrets.get("ECO_CHAT_MODEL", ""),
        }
    except Exception:
        return {"api_url": "", "api_key": "", "model": ""}


def ai_chat_response(chat_history, language_name):
    """Optional AI-ready adapter. Returns None when no provider is configured."""
    settings = get_optional_ai_settings()
    if not all(settings.values()):
        return None
    try:
        headers = {
            "Authorization": f"Bearer {settings['api_key']}",
            "Content-Type": "application/json",
        }
        system_prompt = (
            "You are EcoVision, a careful waste segregation and reuse assistant. "
            "Answer in " + language_name + ". Give practical, safe, concise advice. "
            "Do not suggest DIY reuse for batteries, chemicals, medical waste, or hazardous items. "
            "Explain that recyclability depends on cleanliness and local facilities. "
            "Do not claim certainty about image recognition or environmental rules."
        )
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(chat_history[-8:])
        response = requests.post(
            settings["api_url"],
            headers=headers,
            json={"model": settings["model"], "messages": messages, "temperature": 0.4},
            timeout=25,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def hybrid_chat_answer(question, lang, language_name):
    history = st.session_state.get("eco_chat", [])
    ai_answer = ai_chat_response(history + [{"role": "user", "content": question}], language_name)
    if ai_answer:
        return ai_answer, "AI"

    topic = detect_faq_topic(question)
    if topic:
        st.session_state["eco_last_topic"] = topic
        return faq_topic_answer(topic, lang), "FAQ"

    follow_up_words = [
        "more", "details", "steps", "examples", "what about it", "how about it",
        "దాని గురించి", "ఇంకా", "వివరాలు", "ఎలా చేయాలి", "మరిన్ని",
        "और बताओ", "विस्तार", "மேலும்", "விவரம்", "കൂടുതൽ", "ವಿವರ",
    ]
    if any(word in question.lower() for word in follow_up_words):
        previous_topic = st.session_state.get("eco_last_topic")
        if previous_topic:
            return faq_topic_answer(previous_topic, lang), "FAQ"

    return faq_answer(question, lang), "FAQ"


# Sidebar language selection
st.sidebar.title("♻️ EcoVision")
language_name = st.sidebar.selectbox(get_text("language", "en"), list(LANGUAGES.keys()))
lang = LANGUAGES[language_name]
t = lambda key: get_text(key, lang)

st.markdown("""
<style>
    .block-container {max-width: 1050px; padding-top: 1.8rem; padding-bottom: 2.5rem;}
    .eco-hero {padding: 1.25rem 1.4rem; border-radius: 18px; background: linear-gradient(120deg, #e8f5e9, #f1f8e9); border: 1px solid #c8e6c9; margin-bottom: 1rem;}
    .eco-hero h1 {margin: 0; color: #1b5e20;}
    .eco-muted {color: #45634a; margin-top: .35rem;}
    div[data-testid="stMetric"] {background: #f7fbf7; border: 1px solid #dcebdd; padding: 12px; border-radius: 12px;}
</style>
<div class="eco-hero">
  <h1>♻️ EcoVision</h1>
  <div class="eco-muted">Smart waste classification • Reuse guidance • Hybrid FAQ assistant</div>
</div>
""", unsafe_allow_html=True)
st.caption(t("tagline"))
st.write(t("intro"))

# Load model and labels
try:
    model = load_waste_model()
    labels = load_labels()
except Exception as exc:
    st.error("Could not load keras_model.h5 or labels.txt. Keep both files beside app.py.")
    st.code(str(exc))
    st.stop()

st.subheader("📷 " + t("upload"))
uploaded = st.file_uploader(t("upload"), type=["jpg", "jpeg", "png"])

if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption=t("uploaded"), use_container_width=True)
        if st.button(t("identify"), type="primary", use_container_width=True):
            with st.spinner(t("analyzing")):
                category, confidence, top_predictions = predict_waste(image, model, labels)

            st.subheader("🌱 " + t("result"))
            translated_category = CATEGORY_NAMES.get(category, {}).get(lang, category)
            st.write(f"**{t('category')}:** {translated_category}")
            st.write(f"**{t('confidence')}:** {confidence * 100:.1f}%")
            st.progress(max(0.0, min(1.0, confidence)))
            if confidence < 0.55:
                st.warning("Low model confidence. Treat this result as uncertain and try a clear, close-up photo.")
            with st.expander("View top model predictions"):
                for predicted_label, predicted_score in top_predictions:
                    st.write(f"{predicted_label}: {predicted_score * 100:.1f}%")

            bio_text, recycle_text = category_guidance(category, lang)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{t('bio')}**")
                st.info(bio_text)
            with col2:
                st.markdown(f"**{t('recycle')}**")
                st.info(recycle_text)

            st.subheader("♻️ " + t("ideas"))
            for i, idea in enumerate(get_ideas(category, lang), start=1):
                st.markdown(f"**{i}. {idea}**")

            st.caption(t("caution"))
            st.warning(t("model_note"))
    except Exception as exc:
        st.error("Could not read or analyze this image.")
        st.code(str(exc))

st.divider()
st.subheader("💬 " + t("chat_title"))
st.write(t("chat_intro"))

if "eco_chat" not in st.session_state:
    st.session_state.eco_chat = []

for message in st.session_state.eco_chat:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            st.caption("Response mode: " + message.get("mode", "FAQ"))

user_question = st.chat_input(t("chat_placeholder"))
if user_question:
    st.session_state.eco_chat.append({"role": "user", "content": user_question})
    answer, answer_mode = hybrid_chat_answer(user_question, lang, language_name)
    st.session_state.eco_chat.append({"role": "assistant", "content": answer, "mode": answer_mode})
    st.rerun()

if st.button("Clear chat"):
    st.session_state.eco_chat = []
    st.rerun()

st.info("Hybrid chatbot status: free FAQ mode is active. To enable a compatible AI provider later, add ECO_CHAT_API_URL, ECO_CHAT_API_KEY and ECO_CHAT_MODEL to Streamlit Secrets. Without these settings, the app stays on the free FAQ assistant.")
st.caption("EcoVision | Free FAQ first • Optional AI integration later")
