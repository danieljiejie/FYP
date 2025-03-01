import MusicAlgorithmFunciton
import pretty_midi
import importlib
import MusicPostProcess
import random



# After modifying the function file, reload it
importlib.reload(MusicAlgorithmFunciton)
importlib.reload(MusicPostProcess)
# Now call your functions

midi_data = pretty_midi.PrettyMIDI()

min_v = 100  # Minimum velocity
max_v = 120  # Maximum velocity
emotion = 'awe'  # Emotion for melody generation
drum_track = pretty_midi.Instrument(program=0,is_drum=True)
if emotion == 'happy':
    # Malody: 1,24,26,73,48,56
    # Chord: 1,24,9
    piano_track = pretty_midi.Instrument(program=56)  # Acoustic Grand Piano
    chord_track = pretty_midi.Instrument(program=0) 
    tempo = 100

elif emotion == 'sad':
    # Malody: 1,40,68,48,19
    # Chord: 0,1,49
    piano_track = pretty_midi.Instrument(program=19)  # Acoustic Grand Piano
    chord_track = pretty_midi.Instrument(program=49) 
    tempo = random.randint(60, 70)

elif emotion == 'angry':
    # Melody: 30,28,1
    # Chord: 29,1,27,28
    piano_track = pretty_midi.Instrument(program=28)  # Acoustic Grand Piano
    chord_track = pretty_midi.Instrument(program=27) 
    tempo = 140

elif emotion == 'fear':
    # Melody: 11,41,103,13,1
    # Chord: 52,50,1,92   
    piano_track = pretty_midi.Instrument(program=1)  # Acoustic Grand Piano
    chord_track = pretty_midi.Instrument(program=92) 
    tempo = 70

elif emotion == 'awe':
    # Melody: 54, 48, 49, 60,1
    # Chord:  49, 19,48,1
    piano_track = pretty_midi.Instrument(program=72)  # Acoustic Grand Piano
    chord_track = pretty_midi.Instrument(program=49) 
    tempo = 65


#Awe : melody: 54, 48, 49,  bass:  56, 60, 61
#tempo = 50 # Tempo in BPM
total_duration = (tempo/60)*20
rhythm = MusicAlgorithmFunciton.generate_rhythm(total_duration,emotion,tempo)

MusicAlgorithmFunciton.generate_melody_with_chordsV2(rhythm, piano_track, chord_track, drum_track, min_v, max_v, emotion,total_duration, tempo)

# audio = MusicAlgorithmFunciton.generate_melody_with_chordsV3(rhythm,min_v, max_v, emotion,\
#                                                              total_duration, tempo,sample_rate=44100,\
#                                                              instrument = 'piano')

print(piano_track.notes)
print(chord_track.notes)
print(drum_track.notes)
# Create a PrettyMIDI object and add the tracks

MusicPostProcess.add_channel_volume(piano_track,127)
MusicPostProcess.set_expression(piano_track,127)
midi_data.instruments.append(piano_track)
midi_data.instruments.append(chord_track)
midi_data.instruments.append(drum_track)

# Apply post-processing
#MusicPostProcess.apply_articulation_postprocessing(midi_data, emotion)

# Save the MIDI file
midi_data.write('Music Output/output.mid')

# if emotion == 'angry':
#     MusicPostProcess.enhance_aggression(midi_data)

#     # Save the MIDI file
#     midi_data.write('Music Output/output_angry.mid')
#sample_rate = 44100
# Save to WAV file
#sf.write('Music Output/output_soundtrack.wav', audio, sample_rate)


