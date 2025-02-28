import os
from datetime import datetime
import streamlit as st
from werkzeug.utils import secure_filename
import pretty_midi
import importlib

# Import your existing backend components
from EmotionDetectionSystem import EmotionDetectionSystem
import MusicAlgorithmFunciton
import MusicPostProcess

# Dynamic paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MUSIC_FOLDER = os.path.join(BASE_PATH, "static", "music")
MODEL_FOLDER = os.path.join(BASE_PATH, "Model")
MODEL_PATH = os.path.join(MODEL_FOLDER, "Emotion_Detection_System.joblib")
os.makedirs(MUSIC_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Load CSS from external file
def load_css(file_path):
    with open(file_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize emotion detection system with a loading spinner
with st.spinner("Initializing Emotion Detection System..."):
    emotion_system = EmotionDetectionSystem()
    if not os.path.exists(MODEL_PATH):
        st.info("First-time setup: Loading and saving models...")
        emotion_system.load_models()
        emotion_system.save_system()
    else:
        st.info("Loading system from saved model...")
        emotion_system.load_system()
st.success("System ready!")

# Keep the generate_midi function unchanged
def generate_midi(emotion):
    importlib.reload(MusicAlgorithmFunciton)
    importlib.reload(MusicPostProcess)
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    midi_filename = f"{emotion}_{timestamp}.mid"
    output_path = os.path.join(MUSIC_FOLDER, midi_filename)

    # MIDI generation logic
    midi_data = pretty_midi.PrettyMIDI()
    min_v = 100
    max_v = 120
    drum_track = pretty_midi.Instrument(program=0, is_drum=True)

    if emotion == 'happy':
        piano_track = pretty_midi.Instrument(program=1)
        chord_track = pretty_midi.Instrument(program=0)
        tempo = 100
    elif emotion == 'sad':
        piano_track = pretty_midi.Instrument(program=1)
        chord_track = pretty_midi.Instrument(program=0)
        tempo = 60
    elif emotion == 'angry':
        piano_track = pretty_midi.Instrument(program=10)
        chord_track = pretty_midi.Instrument(program=0)
        tempo = 120
    elif emotion == 'fear':
        piano_track = pretty_midi.Instrument(program=11)
        chord_track = pretty_midi.Instrument(program=52)
        tempo = 70
    elif emotion == 'awe':
        piano_track = pretty_midi.Instrument(program=48)
        chord_track = pretty_midi.Instrument(program=60)
        tempo = 65

    total_duration = (tempo / 60) * 30
    rhythm = MusicAlgorithmFunciton.generate_rhythm(total_duration, emotion, tempo)
    MusicAlgorithmFunciton.generate_melody_with_chordsV2(
        rhythm, piano_track, chord_track, drum_track, min_v, max_v, emotion, total_duration, tempo
    )

    MusicPostProcess.add_channel_volume(piano_track, 127)
    MusicPostProcess.set_expression(piano_track, 127)

    midi_data.instruments.append(piano_track)
    midi_data.instruments.append(chord_track)
    midi_data.instruments.append(drum_track)
    midi_data.write(output_path)

    return output_path

def main():
    # Set page config for a wider layout and custom title
    st.set_page_config(page_title="Emotion to Music", layout="centered", page_icon="🎶")

    # Load external CSS
    load_css(os.path.join(BASE_PATH, "styles.css"))

    # Title and description
    st.markdown('<div class="main-title">Emotion-Based Music Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Upload an image and let us compose music inspired by its emotion!</div>', unsafe_allow_html=True)

    # File uploader with custom styling
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop your image here or click to upload", 
                                    type=["png", "jpg", "jpeg", "gif"], 
                                    help="Supported formats: PNG, JPG, JPEG, GIF")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        # Add a progress spinner during processing
        with st.spinner("Analyzing emotion and generating music..."):
            try:
                # Save uploaded file temporarily
                temp_path = os.path.join(MUSIC_FOLDER, secure_filename(uploaded_file.name))
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Process image with emotion detection
                result = emotion_system.predict_emotion(temp_path)
                emotion = result['emotion']

                # Generate MIDI
                midi_path = generate_midi(emotion)

                # Display result with better formatting
                st.markdown(f'<p class="emotion-text">Detected Emotion: {emotion.capitalize()}</p>', unsafe_allow_html=True)
                st.success("Music generated successfully! 🎵")

                # Preview the uploaded image
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

                # Provide download button with a friendly label
                with open(midi_path, "rb") as file:
                    st.download_button(
                        label=f"Download {emotion.capitalize()} Music",
                        data=file,
                        file_name=os.path.basename(midi_path),
                        mime="audio/midi",
                        key="download-midi"
                    )

                # Clean up temporary file
                os.remove(temp_path)

            except Exception as e:
                st.error(f"Oops! Something went wrong: {str(e)}")
                st.write("Please try uploading a different image.")

    # Add a footer
    st.markdown("---")
    st.markdown('<p class="footer">Built with ❤️ using Streamlit</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()