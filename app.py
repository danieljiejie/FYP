import os
from datetime import datetime
import streamlit as st
from werkzeug.utils import secure_filename
import pretty_midi
import importlib
import requests

# Import your existing backend components
from EmotionDetectionSystem import EmotionDetectionSystem
import MusicAlgorithmFunciton
import MusicPostProcess

# Dynamic paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MUSIC_FOLDER = os.path.join(BASE_PATH, "static", "music")
MODEL_FOLDER = os.path.join(BASE_PATH, "Model")
MODEL_PATH = os.path.join(MODEL_FOLDER, "Emotion_Detection_System_cpu.joblib")
os.makedirs(MUSIC_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)


# GitHub URLs for models (replace with your actual URLs)
FACIAL_MODEL_URL = "https://github.com/danieljiejie/FYP/releases/download/v1.0.1/FacialEmotionModel.h5"
GENERAL_MODEL_URL = "https://github.com/danieljiejie/FYP/releases/download/v1.0.2/NonFacialEmotionModelV2.pth"
JOBLIB_MODEL_URL = "https://github.com/danieljiejie/FYP/releases/download/v2.0.0/Emotion_Detection_System_cpu.joblib"

# Emotion-specific options
emotion_options = {
    'happy': {
        'melody_instruments': [
            ("Piano", 1),
            ("Acoustic Guitar", 24),
            ("Electric Guitar", 26),
            ("String", 48),
            ("Trumpet", 56),
            ("Flute", 73)
        ],
        'chord_instruments': [
            ("Piano", 0),
            ("Harp", 46),
            ("Acoustic Guitar", 24),
            ("Glockenspiel", 9)
        ],
        'tempos': [90, 100, 110, 120]  # Bright, upbeat tempos
    },
    'sad': {
        'melody_instruments': [
            ("Piano", 1),
            ("Organ", 19),
            ("Violin", 40),
            ("Oboe", 68),
            ("String", 48),
            ("Clarinet", 71)
        ],
        'chord_instruments': [
            ("Piano", 0),
            ("String", 49),
            ("Cello", 42)
        ],
        'tempos': [50, 60, 70, 80]  # Slow, melancholic tempos
    },
    'angry': {
        'melody_instruments': [
            ("Piano", 1),
            ("Acoustic Guitar", 24),
            ("Electric Guitar", 28),
            ("Distortion Guitar", 30),
            ("Saxophone", 64)
        ],
        'chord_instruments': [
            ("Piano", 0),
            ("Electric Guitar", 27),
            ("OverDriven Guitar", 29),
            ("French Horn", 60)
        ],
        'tempos': [110, 120, 130, 140]  # Fast, intense tempos
    },
    'fear': {
        'melody_instruments': [
            ("Piano", 1),
            ("Tubular Bells", 11),
            ("Violin", 40),
            ("Viola", 41),
            ("Xylophone", 13),
            ("Oboe", 68)
        ],
        'chord_instruments': [
            ("Piano",0),
            ("Choir Aahs", 52),
            ("Synth Strings", 50),
            ("String Ensemble", 49),
            ("Pad 5", 92)
        ],
        'tempos': [60, 70, 80, 90]  # Tense, eerie tempos
    },
    'awe': {
        'melody_instruments': [
            ("Piano", 1),
            ("String Ensemble 1", 48),
            ("String Ensemble 2", 49),
            ("Synth Voice", 54),
            ("French Horn", 60)
        ],
        'chord_instruments': [
            ("Piano", 1),
            ("French Horn", 60),
            ("String Ensemble", 49),
            ("Organ", 19)
        ],
        'tempos': [60, 70, 80, 90]  # Majestic, flowing tempos
    }
}

# Universal duration options (kept the same for simplicity)
duration_options = [15, 30, 45, 60]  # in seconds

# Set page config for a wider layout and custom title
st.set_page_config(page_title="Emotion to Music", layout="centered", page_icon="🎶")

# Load CSS from external file
def load_css(file_path):
    with open(file_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Download model files from GitHub if not present
@st.cache_resource
def download_models(force_download=False):
    model_files = {
        "FacialEmotionModel.h5": FACIAL_MODEL_URL,
        "NonFacialEmotionModelV2.pth": GENERAL_MODEL_URL,
        "Emotion_Detection_System_cpu.joblib": JOBLIB_MODEL_URL
    }
    for filename, url in model_files.items():
        local_path = os.path.join(MODEL_FOLDER, filename)
        if not os.path.exists(local_path) or force_download:
            with st.spinner(f"Downloading {filename} from GitHub..."):
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(local_path, "wb") as f:
                        f.write(response.content)
                    st.info(f"Downloaded {filename} successfully!")
                else:
                    st.error(f"Failed to download {filename}. Status code: {response.status_code}")
                    raise Exception(f"Model download failed: {filename}")

# Initialize emotion detection system with a loading spinner
@st.cache_resource
def init_emotion_system():
    download_models()
    with st.spinner("Initializing Emotion Detection System..."):
        emotion_system = EmotionDetectionSystem()
        if not os.path.exists(MODEL_PATH):
            st.info("First-time setup: Loading and saving models...")
            emotion_system.load_models()
            emotion_system.save_system()
        else:
            st.info("Loading system from saved model...")
            emotion_system.load_system()
    return emotion_system

# Keep the generate_midi function unchanged
@st.cache_data
def generate_midi(emotion,melody_instrument_program,chord_instrument_program,total_duration,tempo,_timestamp):
    importlib.reload(MusicAlgorithmFunciton)
    importlib.reload(MusicPostProcess)
    
    midi_filename = f"{emotion}_{_timestamp}.mid"
    output_path = os.path.join(MUSIC_FOLDER, midi_filename)

    # MIDI generation logic
    midi_data = pretty_midi.PrettyMIDI()
    min_v = 100
    max_v = 120

    drum_track = pretty_midi.Instrument(program=0, is_drum=True)

    melody_track = pretty_midi.Instrument(program=melody_instrument_program)
    chord_track = pretty_midi.Instrument(program=chord_instrument_program)

    total_duration = total_duration * (tempo/60)
    rhythm = MusicAlgorithmFunciton.generate_rhythm(total_duration, emotion, tempo)
    MusicAlgorithmFunciton.generate_melody_with_chordsV2(
        rhythm, melody_track, chord_track, drum_track, min_v, max_v, emotion, total_duration, tempo
    )

    MusicPostProcess.add_channel_volume(melody_track, 127)
    MusicPostProcess.set_expression(melody_track, 127)

    midi_data.instruments.append(melody_track)
    midi_data.instruments.append(chord_track)
    midi_data.instruments.append(drum_track)
    midi_data.write(output_path)

    return output_path

def main():
    # Load external CSS
    load_css(os.path.join(BASE_PATH, "styles.css"))

    # Initialize emotion system only once
    if 'emotion_system' not in st.session_state:
        st.session_state.emotion_system = init_emotion_system()
        st.success("System ready!")
    emotion_system = st.session_state.emotion_system

    # Optional: Button to force model refresh
    if st.button("Refresh Models"):
        download_models(force_download=True)
        st.cache_resource.clear()  # Clear the cached emotion system
        st.session_state.emotion_system = init_emotion_system()
        st.success("Models refreshed and system reloaded!")

    # Title and description
    st.markdown('<div class="main-title">Emotion-Based Music Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Upload an image and let us compose music inspired by its emotion!</div>', unsafe_allow_html=True)

    # File uploader
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop your image here or click to upload", 
                                    type=["png", "jpg", "jpeg", "gif"], 
                                    help="Supported formats: PNG, JPG, JPEG, GIF")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        # Process image only if it’s new
        if 'last_file' not in st.session_state or st.session_state.last_file != uploaded_file.name:
            with st.spinner("Analyzing emotion..."):
                try:
                    # Save uploaded file temporarily
                    temp_path = os.path.join(MUSIC_FOLDER, secure_filename(uploaded_file.name))
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Detect emotion
                    result = emotion_system.predict_emotion(temp_path)
                    emotion = result['emotion']
                    
                    # Store emotion and file info in session state
                    st.session_state.emotion = emotion
                    st.session_state.last_file = uploaded_file.name
                    st.session_state.temp_path = temp_path  # Store temp path for cleanup later
                    
                    # Reset generation state
                    if 'midi_path' in st.session_state:
                        del st.session_state.midi_path

                except Exception as e:
                    st.error(f"Oops! Something went wrong: {str(e)}")
                    st.write("Please try uploading a different image.")
                    return

        # Use cached emotion
        emotion = st.session_state.emotion

        # Display detected emotion
        st.markdown(f'<p class="emotion-text">Detected Emotion: {emotion.capitalize()}</p>', unsafe_allow_html=True)

        # Get emotion-specific options
        options = emotion_options.get(emotion, emotion_options['happy'])  # Fallback to 'happy'

        # Parameter selection with session state persistence
        if 'melody_name' not in st.session_state:
            st.session_state.melody_name = options['melody_instruments'][0][0]
        melody_names = [name for name, _ in options['melody_instruments']]
        melody_name = st.selectbox(
            "Choose a melody instrument:",
            options=melody_names,
            index=melody_names.index(st.session_state.melody_name),
            key="melody_select"
        )
        st.session_state.melody_name = melody_name
        melody_program = next(prog for name, prog in options['melody_instruments'] if name == melody_name)

        if 'chord_name' not in st.session_state:
            st.session_state.chord_name = options['chord_instruments'][0][0]
        chord_names = [name for name, _ in options['chord_instruments']]
        chord_name = st.selectbox(
            "Choose a chord instrument:",
            options=chord_names,
            index=chord_names.index(st.session_state.chord_name),
            key="chord_select"
        )
        st.session_state.chord_name = chord_name
        chord_program = next(prog for name, prog in options['chord_instruments'] if name == chord_name)

        if 'total_duration' not in st.session_state:
            st.session_state.total_duration = duration_options[1]  # Default to 30
        total_duration = st.selectbox(
            "Choose the duration (seconds):",
            options=duration_options,
            index=duration_options.index(st.session_state.total_duration),
            key="duration_select"
        )
        st.session_state.total_duration = total_duration

        if 'tempo' not in st.session_state:
            st.session_state.tempo = options['tempos'][len(options['tempos']) // 2]  # Middle tempo
        tempo = st.selectbox(
            "Choose the tempo (BPM):",
            options=options['tempos'],
            index=options['tempos'].index(st.session_state.tempo),
            key="tempo_select"
        )
        st.session_state.tempo = tempo

        # Generate music only on button click and if parameters changed
        generate_key = f"{emotion}_{melody_program}_{chord_program}_{total_duration}_{tempo}"
        if st.button("Generate Music") or ('generate_key' in st.session_state and st.session_state.generate_key != generate_key):
            with st.spinner("Generating music..."):
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                midi_path = generate_midi(emotion, melody_program, chord_program, total_duration, tempo, timestamp)
                st.session_state.midi_path = midi_path
                st.session_state.generate_key = generate_key
                st.success("Music generated successfully! 🎵")

        # Display results if MIDI has been generated
        if 'midi_path' in st.session_state:
            midi_path = st.session_state.midi_path
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            with open(midi_path, "rb") as file:
                st.download_button(
                    label=f"Download {emotion.capitalize()} Music",
                    data=file,
                    file_name=os.path.basename(midi_path),
                    mime="audio/midi",
                    key="download-midi"
                )

        # Clean up temporary file only once after processing
        if 'temp_path' in st.session_state:
            os.remove(st.session_state.temp_path)
            del st.session_state.temp_path

    # Add a footer
    st.markdown("---")
    st.markdown('<p class="footer">Built with ❤️ using Streamlit</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
