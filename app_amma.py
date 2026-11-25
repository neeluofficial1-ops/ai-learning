import streamlit as st
from gtts import gTTS
import io

st.title("Amma pesum app 🗣️")

phrases = {
    "💧 thanni":"எனக்கு தண்ணீர் வேண்டும்",
    "🍚 Saapadu": "எனக்கு பசிக்கிறது",
    "🤕 Valikudhu": "எனக்கு வலிக்கிறது",
    "🏥 Doctor": "டாக்டரிடம் போக வேண்டும்",
    "🚻 Bathroom": "நான் பாத்ரூம் போக வேண்டும்",
    "😴 Thookam": "எனக்கு தூக்கம் வருகிறது",
}

def speak(text):
    tts = gTTS(text=text,lang="ta")
    audio = io.BytesIO()
    tts.write_to_fp(audio)
    st.audio(audio.getvalue(),format="audio/mp3",autoplay=True)

cols = st.columns(2)
for i , (label,sentence) in enumerate(phrases.items()):
     if cols[i % 2].button(label,use_container_width=True):
         st.success(sentence)
         speak(sentence)

st.divider()
st.subheader("✍️ Amma type pannunga")

text = st.text_area("Inga Tamil la type pannunga:")

if st.button("🔊 Pesu", use_container_width=True):
    if text.strip()=="":
        st.warning("mudhalla edhavadhu type pannunga")
    else:
        st.success(text)
        speak(text)