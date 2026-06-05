import streamlit as st
import os
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# --- Setup ---
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

st.title("Domain-Specific-Language Translator with ChromaDB")

# --- Keywords ---
medical_keywords = [
    "blood pressure","bp","dizziness","patient","treatment","diagnosis","doctor","nurse","hr","ecg","temp",
    "symptom","therapy","medicine","clinical","prognosis","prescription","disease","heart rate","fever","infection","surgery","hospital"
]
legal_keywords = [
    "contract","agreement","law","court","obligation","clause","statute","regulation","compliance","settlement",
    "evidence","verdict","appeal","notary","legal","rights","liability","jurisdiction","land","property","estate","agrmt","ctrct","juris"
]

# --- Abbreviations ---
medical_abbreviations = {"bp":"Blood Pressure","hr":"Heart Rate","ecg":"Electrocardiogram","temp":"Temperature"}
legal_abbreviations = {"agrmt":"Agreement","ctrct":"Contract","juris":"Jurisdiction"}

def expand_abbreviations(text: str, domain: str) -> str:
    text_lower = text.lower().strip()
    if domain == "Medical" and text_lower in medical_abbreviations:
        return medical_abbreviations[text_lower]
    elif domain == "Legal" and text_lower in legal_abbreviations:
        return legal_abbreviations[text_lower]
    return text

# --- Domain Detection ---
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
                    {"role":"system","content":"You are a domain classifier. Classify text as Medical, Legal, or Unknown."},
                    {"role":"user","content":text}
                ],
                temperature=0
            )
            classification = response.choices[0].message.content.strip()
            if "legal" in classification.lower():
                return "Legal"
            #elif "medical" in classification.lower():
               # return "Medical"
            else:
                return "Unknown"
        except Exception:
            return "Unknown"

# --- Load persisted DBs ---
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
#medical_db = Chroma(persist_directory="medical_chroma", embedding_function=embeddings)
legal_db = Chroma(persist_directory="legal_chroma", embedding_function=embeddings)

# --- Validation: show DB sizes ---
st.write("📊 Legal DB count:", len(legal_db.get()["ids"]))
#st.write("📊 Medical DB count:", len(medical_db.get()["ids"]))

def retrieve_context(query, domain):
    if domain == "Medical":
        results = medical_db.similarity_search(query, k=3)
    elif domain == "Legal":
        results = legal_db.similarity_search(query, k=3)
    else:
        results = []
    return "\n".join([r.page_content for r in results])

# --- Translator UI ---
st.subheader("🌐 Translator")
text_input = st.text_area("Enter the text to translate")
source_lang = st.selectbox("Source Language", ["English","French","German","Hindi"])
target_lang = st.selectbox("Target Language", ["English","French","German","Hindi"])
domain = st.selectbox("Domain", ["Legal","Medical"])

# --- Translate Button ---
if st.button("Translate"):
    if text_input.strip() == "":
        st.warning("Please enter some text")
    else:
        expanded_text = expand_abbreviations(text_input, domain)
        detected_domain = detect_domain(expanded_text)

        if detected_domain != "Unknown" and detected_domain != domain:
            st.error(f"❌ The text appears to belong to the **{detected_domain}** domain, but you selected **{domain}**.")
        elif detected_domain == "Unknown":
            st.error(f"❌ The text does not appear to belong to the **{domain}** domain.")
        else:
            context = retrieve_context(expanded_text, domain)
            
    
            prompt = f"""
            Translate the following text from {source_lang} to {target_lang} in the {domain} domain.
            Maintain domain-specific terminology.

            Context (for reference):
            {context}

            Text:
            {expanded_text}
            """
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role":"system","content":"You are a professional translator"},
                        {"role":"user","content":prompt}
                    ],
                    temperature=0.3
                )
                translated_text = response.choices[0].message.content
                st.subheader("Translated Output")
                st.success(translated_text)
            except Exception as e:
                st.error("Translation failed. Please check your API key or try again later.")
                st.text(str(e))
