import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.say("Text to speech test. If you hear this, TTS is working.")
engine.runAndWait()
