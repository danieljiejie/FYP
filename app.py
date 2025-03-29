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

import json
import shutil

import sqlitecloud
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
#JOBLIB_MODEL_URL = "https://github.com/danieljiejie/FYP/releases/download/v2.0.0/Emotion_Detection_System_cpu.joblib"

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
        "best_swin_emotion_model.pth": GENERAL_MODEL_URL,
        #"Emotion_Detection_System_cpu.joblib": JOBLIB_MODEL_URL
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
def generate_midi(emotion,melody_instrument_program,chord_instrument_program,total_duration,tempo,_timestamp,include_drums = True):
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
    if include_drums:
        midi_data.instruments.append(drum_track)
    midi_data.write(output_path)

    return output_path

def get_db_connection():
    """Establish a connection to the database (local or cloud based on environment)."""
    # if "database" in st.secrets and "url" in st.secrets["database"]:
    conn = sqlitecloud.connect(st.secrets["database"]["url"])
    # else:  # Local SQLite fallback
    #     conn = sqlite3.connect(FEEDBACK_DB)
    #     conn.row_factory = sqlite3.Row
    return conn

def init_database():
    """Initialize the feedback database and create the table if it doesn't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            emotion TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            feedback TEXT NOT NULL,
            music_file TEXT,
            image_file TEXT,
            date TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_feedback(emotion, timestamp, user_feedback, midi_path=None, image_path=None):
    conn = get_db_connection()
    #cursor = conn.cursor() if isinstance(conn, sqlite3.Connection) else conn
    cursor = conn.cursor()
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    music_file = None
    if midi_path:
        feedback_music_folder = os.path.join(BASE_PATH, "static", "feedback_music")
        os.makedirs(feedback_music_folder, exist_ok=True)
        feedback_midi_path = os.path.join(feedback_music_folder, f"{emotion}_{timestamp}.mid")
        shutil.copy(midi_path, feedback_midi_path)
        music_file = os.path.relpath(feedback_midi_path, BASE_PATH)  # For now; replace with cloud URL later
    
    image_file = None
    if image_path:
        feedback_images_folder = os.path.join(BASE_PATH, "static", "feedback_images")
        os.makedirs(feedback_images_folder, exist_ok=True)
        image_ext = os.path.splitext(image_path)[1]
        feedback_image_path = os.path.join(feedback_images_folder, f"{emotion}_{timestamp}{image_ext}")
        if image_path.startswith(feedback_images_folder):
            image_file = os.path.relpath(image_path, BASE_PATH)
        else:
            shutil.copy(image_path, feedback_image_path)
            image_file = os.path.relpath(feedback_image_path, BASE_PATH)
    
    # Insert into database (compatible with SQLite or SQLAlchemy)
    query = '''
        INSERT INTO feedback (emotion, timestamp, feedback, music_file, image_file, date)
        VALUES (?, ?, ?, ?, ?, ?)
    '''
   
    cursor.execute(query, (emotion, timestamp, user_feedback, music_file, image_file, date))
    conn.commit()
    conn.close()
    
def load_feedback():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM feedback ORDER BY id DESC')
    rows = cursor.fetchall()
    
    # Get column names from cursor.description
    columns = [desc[0] for desc in cursor.description]
    
    # Convert each row to a dictionary
    feedback_list = [dict(zip(columns, row)) for row in rows]
    
    conn.close()
    return feedback_list

def display_feedback():
    feedback_list = load_feedback()
    if feedback_list:
        for entry in feedback_list:
            st.markdown("---")
            col1, col2 = st.columns([1, 2])
            with col1:
                if entry['image_file']:
                    image_path = os.path.join(BASE_PATH, entry['image_file'])
                    if os.path.exists(image_path):
                        st.image(image_path, caption="Uploaded Image", use_container_width=True)
                    else:
                        st.write("Image not available")
                else:
                    st.write("Image not available")
            with col2:
                st.markdown(f"**Emotion:** {entry['emotion'].capitalize()}")
                st.markdown(f"**Feedback:** {entry['feedback']}")
                if entry['music_file']:
                    music_path = os.path.join(BASE_PATH, entry['music_file'])
                    if os.path.exists(music_path):
                        with open(music_path, "rb") as audio_file:
                            audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format="audio/midi")
                        st.download_button(
                            label="Download Music",
                            data=audio_bytes,
                            file_name=os.path.basename(music_path),
                            mime="audio/midi",
                            key=f"download_{entry['timestamp']}"
                        )
                    else:
                        st.write("Music file not available")
                else:
                    st.write("Music file not available")
                st.markdown(f"<small>{entry['date']}</small>", unsafe_allow_html=True)
    else:
        st.write("No feedback yet.")

def generate_music_options(emotion, tab_key_prefix):
    """Display music generation options and return selected parameters with tab-specific keys."""
    options = emotion_options.get(emotion, emotion_options['happy'])  # Fallback to 'happy'

    # Melody instrument
    if 'melody_name' not in st.session_state or st.session_state.melody_name not in [name for name, _ in options['melody_instruments']]:
        st.session_state.melody_name = options['melody_instruments'][0][0]  # Default to first option
    melody_names = [name for name, _ in options['melody_instruments']]
    melody_name = st.selectbox(
        "Choose a melody instrument:",
        options=melody_names,
        index=melody_names.index(st.session_state.melody_name),
        key=f"{tab_key_prefix}_melody_select_{emotion}"
    )
    st.session_state.melody_name = melody_name
    melody_program = next(prog for name, prog in options['melody_instruments'] if name == melody_name)

    # Chord instrument
    if 'chord_name' not in st.session_state or st.session_state.chord_name not in [name for name, _ in options['chord_instruments']]:
        st.session_state.chord_name = options['chord_instruments'][0][0]  # Default to first option
    chord_names = [name for name, _ in options['chord_instruments']]
    chord_name = st.selectbox(
        "Choose a chord instrument:",
        options=chord_names,
        index=chord_names.index(st.session_state.chord_name),
        key=f"{tab_key_prefix}_chord_select_{emotion}"
    )
    st.session_state.chord_name = chord_name
    chord_program = next(prog for name, prog in options['chord_instruments'] if name == chord_name)

    # Duration
    if 'total_duration' not in st.session_state or st.session_state.total_duration not in duration_options:
        st.session_state.total_duration = duration_options[1]  # Default to 30
    total_duration = st.selectbox(
        "Choose the duration (seconds):",
        options=duration_options,
        index=duration_options.index(st.session_state.total_duration),
        key=f"{tab_key_prefix}_duration_select_{emotion}"
    )
    st.session_state.total_duration = total_duration

    # Tempo
    if 'tempo' not in st.session_state or st.session_state.tempo not in options['tempos']:
        st.session_state.tempo = options['tempos'][len(options['tempos']) // 2]  # Middle tempo
    tempo = st.selectbox(
        "Choose the tempo (BPM):",
        options=options['tempos'],
        index=options['tempos'].index(st.session_state.tempo),
        key=f"{tab_key_prefix}_tempo_select_{emotion}"
    )
    st.session_state.tempo = tempo

    # Drums
    if 'include_drums' not in st.session_state:
        st.session_state.include_drums = True
    include_drums = st.checkbox(
        "Include drums?",
        value=st.session_state.include_drums,
        key=f"{tab_key_prefix}_drum_checkbox_{emotion}"
    )
    st.session_state.include_drums = include_drums

    return melody_program, chord_program, total_duration, tempo, include_drums

def handle_music_generation(emotion, melody_program, chord_program, total_duration, tempo, include_drums, tab_key_prefix):
    """Generate music and store it in session state with tab-specific key."""
    generate_key = f"{emotion}_{melody_program}_{chord_program}_{total_duration}_{tempo}"
    if st.button("Generate Music", key=f"{tab_key_prefix}_generate_button"):
       
        with st.spinner("Generating music..."):
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            midi_path = generate_midi(emotion, melody_program, chord_program, total_duration, tempo, timestamp, include_drums)
            st.session_state[f'{tab_key_prefix}_midi_path'] = midi_path
            st.session_state[f'{tab_key_prefix}_timestamp'] = timestamp
            st.session_state[f'{tab_key_prefix}_generate_key'] = generate_key
            st.success("Music generated successfully! 🎵")

def display_music_result(tab_key_prefix):
    """Display generated music if available for the specific tab."""
    midi_key = f'{tab_key_prefix}_midi_path'
    if midi_key in st.session_state:
        midi_path = st.session_state[midi_key]
        timestamp = st.session_state.get(f'{tab_key_prefix}_timestamp', os.path.basename(midi_path).split('_')[1].split('.')[0])
        if tab_key_prefix == "image":
            emotion_label = st.session_state.emotion.capitalize()
        else:
            emotion_label = st.session_state.selected_emotion.capitalize()

        with open(midi_path, "rb") as file:
            st.download_button(
                label=f"Download {emotion_label} Music",
                data=file,
                file_name=os.path.basename(midi_path),
                mime="audio/midi",
                key=f"{tab_key_prefix}_download-midi"
            )

def handle_feedback_submission(emotion, tab_key_prefix, image_path=None):
    """Handle feedback submission with optional image path for the specific tab."""
    with st.form(key=f'{tab_key_prefix}_feedback_form'):
        feedback_text = st.text_area("Share your thoughts about this music!", height=100)
        submit_button = st.form_submit_button(label='Submit Feedback')
        
        if submit_button and feedback_text:
            if image_path and not os.path.exists(image_path):
                st.error("Image file is missing.")
            else:
                feedback_images_folder = os.path.join(BASE_PATH, "static", "feedback_images")
                os.makedirs(feedback_images_folder, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                if image_path:
                    image_ext = os.path.splitext(image_path)[1]
                    feedback_image_path = os.path.join(feedback_images_folder, f"{emotion}_{timestamp}{image_ext}")
                    shutil.copy(image_path, feedback_image_path)
                else:
                    feedback_image_path = None
                midi_key = f'{tab_key_prefix}_midi_path'
                if midi_key in st.session_state:
                    add_feedback(emotion, timestamp, feedback_text, st.session_state[midi_key], feedback_image_path)
                    st.success("Thank you for your feedback! Check it out on the Feedback page.")
                else:
                    st.warning("No music generated yet. Feedback saved without music.")
                    add_feedback(emotion, timestamp, feedback_text, None, feedback_image_path)

def main():

    init_database()
    # Load external CSS
    load_css(os.path.join(BASE_PATH, "styles.css"))

    # Add navigation sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Feedback"])

    # Page logic
    if page == "Home":
        # Initialize emotion system only once
        if 'emotion_system' not in st.session_state:
            st.session_state.emotion_system = init_emotion_system()
            st.success("System ready!")
        emotion_system = st.session_state.emotion_system

        # Optional: Button to force model refresh
        if st.button("Refresh Models"):
            download_models(force_download=True)
            st.cache_resource.clear()
            st.session_state.emotion_system = init_emotion_system()
            st.success("Models refreshed and system reloaded!")

        # Home page content
        st.markdown('<div class="main-title">Emotion-Based Music Generator</div>', unsafe_allow_html=True)
        st.markdown('<div class="subheader">Generate music by uploading an image or selecting an emotion!</div>',
                    unsafe_allow_html=True)

        
        # Tabs for two methods
        tab1, tab2 = st.tabs(["Image Upload", "Emotion Selection"])

        # Tab 1: Image Upload
        with tab1:
            if 'temp_path' in st.session_state and os.path.exists(st.session_state.temp_path):
                st.image(st.session_state.temp_path, caption="Uploaded Image", use_container_width=True)
                emotion = st.session_state.emotion
                st.markdown(f'<p class="emotion-text">Detected Emotion: {emotion.capitalize()}</p>', unsafe_allow_html=True)

                # Music generation options
                melody_program, chord_program, total_duration, tempo, include_drums = generate_music_options(emotion, "image")
                handle_music_generation(emotion, melody_program, chord_program, total_duration, tempo, include_drums, "image")
                display_music_result("image")
                st.markdown('<div class="publish"> Please leave your FEEDBACK to make Developer Improve the App Consistently!</div>',
                    unsafe_allow_html=True)
                handle_feedback_submission(emotion, "image", st.session_state.temp_path)

                if st.button("Upload a new image"):
                    if 'temp_path' in st.session_state and os.path.exists(st.session_state.temp_path):
                        os.remove(st.session_state.temp_path)
                        del st.session_state.temp_path
                    if 'image_midi_path' in st.session_state:
                        del st.session_state.image_midi_path
                    st.rerun()
            else:
                uploaded_file = st.file_uploader("Upload an image to generate music based on its emotion!",
                                                type=["jpg", "jpeg", "png"])
                if uploaded_file is not None:
                    if 'temp_path' in st.session_state and os.path.exists(st.session_state.temp_path):
                        os.remove(st.session_state.temp_path)
                    with st.spinner("Analyzing emotion..."):
                        try:
                            temp_path = os.path.join(MUSIC_FOLDER, secure_filename(uploaded_file.name))
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            result = emotion_system.predict_emotion(temp_path)
                            emotion = result['emotion']
                            st.session_state.emotion = emotion
                            st.session_state.last_file = uploaded_file.name
                            st.session_state.temp_path = temp_path
                            if 'image_midi_path' in st.session_state:
                                del st.session_state.image_midi_path
                        except Exception as e:
                            st.error(f"Oops! Something went wrong: {str(e)}")
                            st.write("Please try uploading a different image.")
                            return
                    st.rerun()

        # Tab 2: Emotion Selection
        with tab2:
            available_emotions = list(emotion_options.keys())
            if 'selected_emotion' not in st.session_state:
                st.session_state.selected_emotion = available_emotions[0]
            emotion = st.selectbox(
                "Select an emotion:",
                options=available_emotions,
                index=available_emotions.index(st.session_state.selected_emotion),
                key="emotion_select"
            )
            st.session_state.selected_emotion = emotion
            #st.session_state.emotion = emotion  # Store for consistency

            # Music generation options
            melody_program, chord_program, total_duration, tempo, include_drums = generate_music_options(emotion, "emotion")
            handle_music_generation(emotion, melody_program, chord_program, total_duration, tempo, include_drums, "emotion")
            display_music_result("emotion")
            st.markdown('<div class="publish"> Please leave your FEEDBACK to make Developer Improve the App Consistently!</div>',
                    unsafe_allow_html=True)
            handle_feedback_submission(emotion, "emotion")

        st.markdown("---")
        st.markdown('<p class="footer">Built with ❤️ using Streamlit</p>', unsafe_allow_html=True)

    elif page == "Feedback":
        st.markdown('<div class="main-title">Community Feedback</div>', unsafe_allow_html=True)
        display_feedback()
        st.markdown("---")
        st.markdown('<p class="footer">Built with ❤️ using Streamlit</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
