import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
load_dotenv()
project_id = os.getenv("OPENAI_PROJECT_ID")
client = OpenAI(api_key=api_key, project=project_id)


st.title("Domain-Specific-Language Translator")


medical_keywords = [
    "blood pressure", "bp","dizziness", "patient", "treatment", "diagnosis","doctor","nurse","hr","ecg","temp",
    "symptom", "therapy", "medicine", "clinical", "prognosis","prescription","disease","heart rate","fever","infection","surgery","hospital"
]
legal_keywords = [
    "contract", "agreement", "law", "court", "obligation", "clause","statute","regulation","compliance","settlement","evidence","verdict","appeal","notary",
    "legal", "rights", "liability", "jurisdiction","land","property","estate","agrmt","ctrct","juris"
]


medical_abbreviations = {
    "bp": "Blood Pressure",
    "hr": "Heart Rate",
    "ecg": "Electrocardiogram",
    "temp": "Temperature"
}

legal_abbreviations = {
    "agrmt": "Agreement",
    "ctrct": "Contract",
    "juris": "Jurisdiction"
}

def expand_abbreviations(text: str, domain: str) -> str:
    text_lower = text.lower().strip()
    if domain == "Medical" and text_lower in medical_abbreviations:
        return medical_abbreviations[text_lower]
    elif domain == "Legal" and text_lower in legal_abbreviations:
        return legal_abbreviations[text_lower]
    return text


def detect_domain(text: str) -> str:
    text_lower = text.lower()
 
    if any(word in text_lower for word in medical_keywords):
        return "Medical"
    elif any(word in text_lower for word in legal_keywords):
        return "Legal"
    else:
       
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a domain classifier. Classify text as Medical, Legal, or Unknown."},
                    {"role": "user", "content": text}
                ],
                temperature=0
            )
            classification = response.choices[0].message.content.strip()
            
            if "medical" in classification.lower():
                return "Medical"
            elif "legal" in classification.lower():
                return "Legal"
            else:
                return "Unknown"
        except Exception:
            return "Unknown"

# --- User inputs ---
text_input = st.text_area("Enter the text to translate")
source_lang = st.selectbox("Source Language", ["English", "French", "German", "Hindi"])
target_lang = st.selectbox("Target Language", ["English", "French", "German", "Hindi"])
domain = st.selectbox("Domain", ["Legal", "Medical"])

# --- Translate button ---
if st.button("Translate"):
    if text_input.strip() == "":
        st.warning("Please enter some text")
    else:
        # Detect domain of the input text
        detected_domain = detect_domain(text_input)

        # If mismatch, block translation and show error
        if detected_domain != "Unknown" and detected_domain != domain:
            st.error(
                f"❌ The text appears to belong to the **{detected_domain}** domain, "
                f"but you selected **{domain}**. It is not related to {domain}."
            )
        elif detected_domain == "Unknown":
            st.error(
                f"❌ The text does not appear to belong to the **{domain}** domain."
            )
        else:
            # Expand abbreviations before translation
            expanded_text = expand_abbreviations(text_input, domain)

            # Build translation prompt
            prompt = f"""
            Translate the following text from {source_lang} to {target_lang} in the {domain} domain.
            Maintain domain-specific terminology.

            Text:
            {expanded_text}
            """
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a professional translator"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )

                translated_text = response.choices[0].message.content

                st.subheader("Translated Output")
                st.success(translated_text)
            except Exception as e:
                st.error("Translation failed. Please check your API key or try again later.")
                st.text(str(e))
