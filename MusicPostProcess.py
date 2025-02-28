import random
import pretty_midi
def apply_articulation_postprocessing(midi_data, emotion):
    """
    Adjusts note durations and timings to simulate different articulations.
    """
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            original_duration = note.end - note.start
            
            # Function to ensure no negative start times
            def safe_adjust(time):
                return max(0, time)

            if emotion == 'angry':
                # Use staccato: shorten the note duration and add slight jitter.
                new_duration = original_duration * 0.7
                jitter = random.uniform(-0.02, 0.02)
                note.start = safe_adjust(note.start + jitter)
                note.end = note.start + new_duration
            
            elif emotion == 'sad':
                # Use legato: slightly lengthen notes and create a small overlap.
                new_duration = original_duration * 1.2
                note.end = note.start + new_duration
                # Optionally, you could adjust subsequent notes to overlap.

            elif emotion == 'fear':
                # Introduce a bit of variability for tension.
                jitter = random.uniform(-0.05, 0.05)
                note.start = safe_adjust(note.start + jitter)
                new_duration = original_duration * random.uniform(0.9, 1.1)
                note.end = note.start + new_duration

            elif emotion == 'awe':
                # A subtle legato effect.
                new_duration = original_duration * 1.1
                note.end = note.start + new_duration

def enhance_aggression(midi_data):
    """
    Modify MIDI file to make the music sound more aggressive and tense.
    - Increases velocity with variation
    - Shortens note durations (staccato effect)
    - Introduces dissonance in harmony
    """
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            # Increase velocity randomly within a heavy range
            note.velocity = min(127, note.velocity + random.randint(10, 30))  # Max is 127
            
            # Shorten note duration for punchy effect (staccato)
            original_duration = note.end - note.start
            note.end = note.start + original_duration * 0.6  # Reduce duration by 40%
            
            # Add slight random timing jitter for human-like aggression
            jitter = random.uniform(-0.02, 0.02)
            note.start = max(0, note.start + jitter)
            note.end = max(note.start, note.end + jitter)
    
    # Introduce harmonic tension by adding a dissonant note to the chord track
    for instrument in midi_data.instruments:
        if instrument.is_drum:  # Skip drum tracks
            continue
        for note in instrument.notes:
            if random.random() < 0.3:  # Add dissonance to 30% of notes
                dissonant_note = pretty_midi.Note(
                    velocity=note.velocity - 10,
                    pitch=note.pitch + random.choice([-1, 1, 6]),  # Minor second or tritone
                    start=note.start,
                    end=note.end
                )
                instrument.notes.append(dissonant_note)


def add_channel_volume(instrument, volume=127, time=0):
    # Create a ControlChange event for channel volume (CC 7)
    volume_cc = pretty_midi.ControlChange(number=7, value=volume, time=time)
    instrument.control_changes.append(volume_cc)

    # Function to set expression
def set_expression(track, value, time=0):
    cc = pretty_midi.ControlChange(number=11, value=value, time=time)
    track.control_changes.append(cc)