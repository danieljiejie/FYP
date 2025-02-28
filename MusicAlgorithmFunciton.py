import random
import pretty_midi
import math
import soundfile as sf
import sounddevice as sd
import numpy as np

def generate_rhythm(total_duration,emotion,tempo):

    # Get next rhythm based on designed probabilities
    def get_next_pattern(current_pattern,emotion):
        if current_pattern is None and emotion == 'happy':
            return random.choice(list(duration_probabilities_happy.keys()))
        elif current_pattern is None and emotion == 'sad':
            return random.choice(list(duration_probabilities_sad.keys()))
        elif current_pattern is None and emotion == 'angry':
            return random.choice(list(duration_probabilities_angry.keys()))
        elif current_pattern is None and emotion == 'fear':
            return random.choice(list(duration_probabilities_fear.keys()))
        elif current_pattern is None and emotion == 'awe':
            return random.choice(list(duration_probabilities_awe.keys()))
        
        if emotion == 'happy':
            probabilities = duration_probabilities_happy[current_pattern]
            
        elif emotion == 'sad':
            probabilities = duration_probabilities_sad[current_pattern]

        elif emotion == 'angry':
            probabilities = duration_probabilities_angry[current_pattern]

        elif emotion == 'fear':
            probabilities = duration_probabilities_fear[current_pattern]
        
        elif emotion == 'awe':
            probabilities = duration_probabilities_awe[current_pattern]

        patterns = list(probabilities.keys())
        weights = list(probabilities.values())
        return random.choices(patterns, weights=weights)[0]
    
    
    # Define rhythm patterns with explicit rests (True indicates the rest)
    rhythm_patterns = {

        "r1": [(1, False), (1, False), (1, False), (1, False)],  # Basic 4/4
        "r2": [(1, False), (0.5, False), (0.5, False), (1, False), (0.5, False),
                (0.5, False)],  # Syncopated
        "r3": [(0.5, False), (0.5, False), (0.5, False), (0.5, False), 
            (0.5, False), (0.5, False), (0.5, False), (0.5, False)],  # Continuous eighth notes
        "r4": [(0.5, False), (0.5, False), (1, False), (1, False), (1, False)],
        "r5": [(1, False), (0.5, False), (0.5, False), (0.5, False), (0.5, False), (1, False)],
        "r6": [(0.5, False), (0.5, False), (0.5, False), (0.5, False), (2, False)],
        "r7": [(0.25, False), (0.25, False), (0.5, False), (1, False), 
            (0.25, False), (0.25, False), (0.5, False), (1, False)],  # Sixteenth note pattern
        "r8": [(1, False), (1, False), (2, False)],  # Whole note ending
        "r9": [(1, False), (1, False), (1, False), (0.5, False), (0.5, False)],
        "r10": [(1, False), (1, False), (1, False), (1, True)],  # Ending with rest
        
        "r11": [(0.75, False), (0.25, False), (0.75, False), (0.25, False), 
        (1, False), (1, False)],  # Dotted rhythm
        "r12": [(1.5, False), (0.5, False), (1.5, False), (0.5, False)],  # Waltz variation
        "r13": [(0.5, True), (0.5, False), (0.5, False), (0.5, True), (1, False), 
        (1, False)],  # Syncopated with rests
        "r14": [(0.25, False), (0.25, False), (0.25, False), (0.25, False), 
        (2, False), (1, True)],  # Quick start, long end
        "r15": [(1, False), (0.25, False), (0.25, False), (0.25, False), 
        (0.25, False), (2, True)],  # Descending pattern

        # Happy Rhythm
        "rh1": [(1,False), (0.75,False),(0.75,False),(0.5,False),(0.5,False),(0.5,False)],
        "rh2": [(1,True),(0.75,False),(0.75,False),(0.5,False),(0.5,False),(0.5,False)],
        "rh3": [(0.5,False),(1.5,False),(1,False),(0.5,False),(0.5,False)],
        "rh4": [(0.5,False),(1,False),(0.5,False),(0.5,False),(0.25, False),(0.75,False),(0.5,False)],
        "rh5": [(0.75,False),(0.25, False),(1,True),(0.75,False),(0.25, False),(1,True)],
        "rh6": [(0.25, False), (0.25, False), (0.25, False), (0.25, False),(0.5,True),
        (0.25, False), (0.25, False), (0.25, False), (0.25, False),(0.5,True),
        (0.25, False), (0.25, False), (0.25, False), (0.25, False)],
        "rh7": [(0.5,False),(0.25, False),(0.25, False),(0.25, True), (0.25, False), 
        (0.5,False),(0.5,False),(0.25, False), (0.25, False), (0.25, False), 
        (0.25, False),(0.5,False)],
        "rh8": [(0.5,False),(0.5,False),(0.25, False), (0.25, False), (0.25, False), 
        (0.25, False),(0.5,True),(0.5,False),(0.5,False),(0.25, False), (0.25, False)],
        "rh9": [(0.75,False),(0.25, False),(0.25,True),(0.25, False), (0.25, False), 
        (0.25, False),(0.5,False),(0.5,False),(1,True)],
        "rh10": [(0.25, False), (0.25, False), (0.25, False), (0.25, False),(0.25, False), 
        (0.25, False), (0.25, False), (0.25, False),(0.5,False),
        (0.25, False), (0.25, False),(0.5, True), (0.25, False), (0.25, False)],
        "rh11": [(0.25, False), (0.25, False), (0.25, False), (0.25, False),(0.25, False), 
        (0.25, False), (0.25, False), (0.25, False),(0.5,False),(0.5,False),(0.5,False),(0.5,False)],
        "rh12": [(0.5,True),(0.5,False),(0.5,False),(0.5,False),(0.5,False),(0.25, False), (0.25, False),(1,True)],
        "rh13": [(0.5,False),(0.25, False), (0.25, False),(0.5,True),(0.5,False),
        (0.5,False),(0.5,False),(0.5,True),(0.5,False)],
        "rh14": [(0.5,True),(0.25, True),(0.25, False), (0.25, False),(0.5,False),
        (0.25, False),(0.25, False),(0.75,False),(1,True)],
        "rh15": [(0.5,False),(0.5,False),(0.5,False),(0.5,False),(0.5,False),
        (0.25, False),(0.5,False),(0.25, False),(0.5,False)],
        "rh16": [(0.25, False),(0.75,False),(0.25, False),(0.75,False),(0.5,False),
        (0.25, False),(0.75,False),(0.25, False),(0.25, False)],
        "rh17": [(1,True),(0.25, False),(0.75,False),(0.25, False),(0.75,False),(1,True)],
        "rh18": [(1,False),(1,False),(1,False),(0.5,False),(0.25, False),(0.25, False)],
        "rh19": [(1,False),(1,False),(0.5,False),(0.25, False),(0.25, False),(0.5,True),(0.5,False)],
        "rh20": [(0.75,False),(0.75,False),(0.5,False),(0.5,False),(0.5,False),
        (0.5,False),(0.25, False),(0.25, False)],
        "rh21": [(0.75,False),(0.75,False),(0.5,False),(0.5,False),(0.5,False),
        (0.5,False),(0.5,False)],
        "rh22": [(0.5,False),(0.25, False),(0.5,False),(0.25, False),(0.5,False),
        (1,False),(0.5,False),(0.5,False)],
        "rh23": [(0.5,True),(0.25, False),(0.25, False),(0.25, False),(0.25, False),
        (0.25, False),(0.25, False),(0.25, False),(0.25, False),(0.25, False),
        (0.25, False),(1,False)],
        "rh24": [(1,True),(0.5,False),(0.5,False),(0.5,False),(1,False),(0.5,False)],
        "rh25": [(0.5,True),(0.5,False),(0.5,False),(0.5,False),(0.5,False),(0.5,False),
        (0.5,False),(0.5,False)],

        # SAD Rhythm
        "rs1": [(1,False),(1,False),(1,False),(1,False)],
        "rs2": [(2,False),(2,False)],
        "rs3": [(0.5,False),(0.5,False),(0.5,False),(0.5,False),(0.5,False),(0.5,False),
               (0.5,False),(0.5,False)],
        "rs4": [(1.5,False),(0.5,False),(1,False),(0.5,False),(0.5,False)],
        "rs5": [(0.75,True),(0.25, False),(0.5,False),(0.25, False),(0.25, False),
                (1,True), (0.5,False),(0.5,False)],
        "rs6": [(1,True),(0.5,False),(0.25, False),(0.25, False),(1,True),
                (0.25,False),(0.75, False)],
        "rs7": [(1,True),(0.5,False),(0.5,False),(0.5,False),(0.5,False),(0.25, False),
                (0.25, False),(0.5,False)],
        "rs8": [(2,False),(1.5,False),(0.5,False)],
        "rs9": [(1,True),(1,False),(0.5,True),(1,False),(0.5,False)],
        "rs10": [(0.5,False),(0.5,False),(0.5,False),(0.25,False),(0.25,False),
                 (0.5,False),(0.5,False),(0.5,False),(0.25,False),(0.25,False)],
        "rs11": [(1.5,False),(0.5,False),(1.5,False),(0.5,False)],
        "rs12": [(1.5,False),(1.5,False),(1,False)],
        "rs13": [(1,True),(1.5,False),(1.5,False),],
        "rs14": [(2,False),(0.5,False),(0.5,False),(0.5,False),(0.5,False)],
        "rs15": [(0.5,False),(0.5,False),(0.75, False),(0.25, False),(0.5,False),(0.5,False),
                 (1,False)],
        "rs16": [(0.5,False),(0.5,False),(0.75,True),(0.25, False),(0.75,True),(0.25, False),
                 (1,False)],
        "rs17": [(2,False),(1.5,False),(0.25,False),(0.25,False)],
        "rs18": [(2,True),(0.5,True),(0.5,False),(0.5,False),(0.5,False)],
        "rs19": [(1,True),(1,False),(1,False),(1,False)],
        "rs20": [(2,False),(1,True),(0.5,False),(0.5,False)],

        # ANGRY Rhythm
        "ra1": [(0.5,False),(0.25, True),(0.5,False),(0.25, True),(0.5,False),(0.5,False),
        (0.25, True),(0.5,False),(0.25, True),(0.5,False)],
        "ra2": [(0.5,False),(0.25, True),(0.5,False),(0.25, True),(0.5,False),
        (0.5,False),(0.5,False),(0.5,False),(0.5,False)],
        "ra3": [(0.5,False),(0.25, False),(0.25, False),(0.5,True),(0.25, False),(0.25, False),
        (0.5,False),(0.5,False),(0.25, False),(0.25, False),(0.5,False)],
        "ra4": [(0.5,False),(0.25, False),(0.25, False),(0.5,True),(0.5,False),
        (0.25,False),(0.25,False),(0.25,False),(0.25,False),(0.25,False),
        (0.25,False),(0.25,False),(0.25,False)],
        "ra5": [(0.5,False),(0.5,False),(0.5,False),(0.5,False),(0.5,False),
        (0.5,False),(0.5,False),(0.5,False),],
        "ra6": [(0.5,False),(0.75, False),(0.25,False),(0.5,False),(0.5,True),(0.5,False),
        (0.5,False),(0.5,False)],
        'ra7': [(0.75, False),(0.25,False),(0.25,False),(0.25,False),(0.5,False),
        (0.25,False),(0.25,False),(0.25,False),(0.25,False),(0.5,False)],
        "ra8": [(2,True),(1,False),(1,False)],
        "ra9": [(0.25,False),(0.25,False),(0.5,True),(0.25,False),(0.25,False),
        (0.5,True),(0.25,False),(0.25,False),(0.5,True),(0.25,False),(0.25,False),(0.5,True)],
        "ra10": [(1,False),(1,False),(1,False),(1,False)],
        "ra11": [(0.25,False),(0.25,False),(0.25,False),(0.25,False),(0.25,False),(0.25,False),
        (0.25,False),(0.25,False),(0.25,False),(0.25,False),(0.5,False),(0.25,False),
        (0.25,False),(0.5,False)],

        # FEAR Rhythm Pattern
        "rf1": [(0.5,False),(0.5,False),(0.5,False),(0.5,False),(0.5,False),
                (0.5,False),(0.5,False),(0.5,False)],
        "rf2": [(1,False),(1,False),(1,False),(1,False)],
        "rf3": [(1,False),(2,False),(1,False)],
        "rf4": [(1,True),(0.5,True),(0.5,False),(0.5,False),(0.5,False),
                (0.5,False),(0.5,False)],
        "rf5": [(1.5,True),(0.5,False),(0.25,False),(0.5,False),(0.5,False),(0.75, False)],
        "rf6": [(0.75, False),(2,False),(0.25,True),(1,True)],
        "rf7": [(1.5,False),(0.5,False),(0.25,False),(0.5,False),(0.5,False),
                (0.75, False)],
        "rf8": [(2,False),(2,False)],
        "rf9": [(2,False),(1,False),(1,False)],
        "rf10": [(1,True),(0.5,True),(0.5,False),(0.25,False),(0.5,False),(0.25,False),
                 (0.5,True),(0.5,False)],
        "rf11": [(0.75, False),(0.25,False),(0.5,True),(0.5,False),(2,False)],
        "rf12": [(0.25,False),(0.25,False),(0.25,False),(0.25,False),
                 (0.25,False),(0.25,False),(0.25,False),(0.25,False),
                 (0.25,False),(0.25,False),(0.25,False),(0.25,False),
                 (0.25,False),(0.25,False),(0.25,False),(0.25,False)],
        "rf13": [(1,False),(0.5,False),(1,False),(0.5,False),(1,False)],
        "rf14": [(0.5,False),(0.5,True),(0.5,False),(0.5,True),
                 (0.5,False),(0.5,True),(0.5,False),(0.5,True)],
        "rf15": [(0.5,False),(0.5,False),(0.25,True),(0.5,False),(0.5,False),(0.25,True),
                 (0.5,False),(0.5,False),(0.5,True)],
        "rf16": [(0.5,False),(0.5,False),(0.5,False),(0.5,False),(1,False),(1,False)],
        "rf17": [(1,False),(1,False),(1,False),(0.5,False),(0.5,False)],
        "rf18": [(0.25,False),(0.25,False),(0.25,True),(0.25,False),(0.25,False),
                 (0.25,False),(0.25,False),(0.25,False),(0.25,False),(0.25,True),
                 (0.25,False),(0.25,False),(0.25,False),(0.25,False),
                 (0.25,False),(0.25,False)],
        "rf19": [(0.25,False),(0.25,False),(0.25,False),(0.25,False),(0.25,False),
                 (0.25,False),(0.25,False),(0.25,False),(0.25,False),(0.25,False),
                 (0.25,True),(0.25,False),(0.25,False),(0.25,False),
                 (0.25,False),(0.25,False)],
        "rf20": [(0.25,False),(0.25,False),(0.25,True),(0.25,False),(0.25,False),
                 (0.25,False),(0.25,False),(0.25,False),(0.25,False),(0.25,False),
                 (0.25,True),(0.25,False),(0.25,False),(0.25,False),
                 (0.25,False),(0.25,True)],

        # AWE Rhythm Pattern
        "raw1": [(1,False),(1,False),(1,False),(1,False)],
        "raw2": [(2,False),(2,False)],
        "raw3": [(0.5,False),(0.5,False),(0.5,False),(0.5,False),
        (0.5,False),(0.5,False),(0.5,False),(0.5,False)],
        "raw4": [(1.5,True),(0.5,False),(0.5,False),(0.5,False),(0.5,False),
        (0.5,False)],
        "raw5": [(1,False),(1,False),(0.5,False),(0.5,False),(0.5,False),(0.5,False)],
        "raw6": [(1,False),(1,False),(1,False),(0.5,False),(0.5,False)],
        "raw7": [(1.5,False),(0.5,False),(1,False),(0.5,False),(0.5,False)],
        "raw8": [(2,True),(1,False),(1,False)],
        "raw9": [(2,False),(0.5,False),(0.5,False),(0.5,False),(0.5,False)],
        "raw10": [(1.5,False),(0.5,False),(2,False)],
        "raw11": [(0.5,False),(0.5,False),(0.5,False),(0.5,False),(1,False),(1,False)],
        "raw12": [(1,False),(0.5,False),(0.5,False),(2,False)],
        "raw13": [(1.5,False),(0.5,False),(0.5,False),(1,False),(0.5,False)],
        "raw14": [(0.5,False),(0.5,False),(0.5,False),(0.5,False),(1,False),
        (0.5,False),(0.5,False)],
        "raw15": [4,False]
                 

    }

    # Enhanced emotional transition probabilities
    duration_probabilities_happy = {

    "r1": {"r1": 0.4, "rh1": 0.3, "rh2": 0.1, "rh3": 0.1, "rh4": 0.1},

    # rh1: Simple, steady rhythm
    "rh1": {"rh1": 0.4, "rh2": 0.3, "rh3": 0.1, "rh4": 0.1, "rh5": 0.1},
    
    # rh2: Similar to rh1 but with an accent
    "rh2": {"rh2": 0.3, "rh1": 0.3, "rh5": 0.2, "rh4": 0.1, "rh3": 0.1},
    
    # rh3: Slightly longer notes, steady
    "rh3": {"rh3": 0.3, "rh4": 0.3, "rh1": 0.1, "rh2": 0.1, "rh5": 0.2},
    
    # rh4: Varied note lengths, similar to rh3
    "rh4": {"rh4": 0.3, "rh3": 0.2, "rh15": 0.2, "rh8": 0.1, "rh5": 0.2},
    
    # rh5: Syncopated rhythm with accents
    "rh5": {"rh5": 0.4, "r1": 0.4, "rh2": 0.1, "r10": 0.1},

    "r10": {"r10": 0.4, "r1": 0.3, "rh2": 0.1, "rh5": 0.1, "rh1": 0.1},
    
    # rh6: Fast, repetitive rhythm
    "rh6": {"rh6": 0.3, "rh7": 0.3, "rh8": 0.2, "rh11": 0.1, "rh10": 0.1},
    
    # rh7: Varied rhythm with accents
    "rh7": {"rh7": 0.4, "rh8": 0.3, "rh6": 0.1, "rh10": 0.1, "rh11": 0.1},
    
    # rh8: Similar to rh7 but more complex
    "rh8": {"rh8": 0.4, "rh7": 0.3, "rh13": 0.1, "rh14": 0.2},
    
    # rh9: Syncopated with longer notes
    "rh9": {"rh9": 0.4, "rh5": 0.3, "rh15": 0.1, "rh16": 0.1, "rh11":0.1},
    
    # rh10: Fast, repetitive with accents
    "rh10": {"rh10": 0.2, "rh6": 0.2, "rh11": 0.4, "rh14": 0.2},
    
    # rh11: Very fast, steady
    "rh11": {"rh11": 0.3, "rh10": 0.2, "rh18": 0.1, "rh19": 0.1, "r3": 0.2},

    "r3": {"r3": 0.4, "rh11": 0.3, "rh10": 0.1, "rh18": 0.1, "rh19": 0.1},
    
    # rh12: Syncopated with longer notes
    "rh12": {"rh12": 0.4, "rh9": 0.2, "rh20": 0.1, "rh21": 0.1, "rh13": 0.2},
    
    # rh13: Varied rhythm with accents
    "rh13": {"rh13": 0.4, "rh14": 0.2, "rh8": 0.1, "rh22": 0.1, "rh12": 0.2},
    
    # rh14: Complex, varied rhythm
    "rh14": {"rh14": 0.2, "rh13": 0.2, "rh23": 0.1, "rh24": 0.1, "rh6": 0.2, "rh19": 0.1, "rh11":0.1},
    
    # rh15: Steady, medium tempo
    "rh15": {"rh15": 0.5, "rh16": 0.3, "rh9": 0.1, "rh25": 0.1},
    
    # rh16: Syncopated, medium tempo
    "rh16": {"rh16": 0.4, "rh15": 0.2, "rh14": 0.2, "rh1": 0.1, "rh13": 0.1},
    
    # rh17: Syncopated with longer notes
    "rh17": {"rh17": 0.2, "rh16": 0.4, "rh18": 0.1, "rh2": 0.1, "rh18": 0.2},
    
    # rh18: Steady, simple rhythm
    "rh18": {"rh18": 0.4, "rh17": 0.2, "rh19": 0.3, "rh3": 0.1},
    
    # rh19: Varied rhythm with accents
    "rh19": {"rh19": 0.4, "rh18": 0.3, "rh20": 0.2, "rh4": 0.1},
    
    # rh20: Fast, repetitive
    "rh20": {"rh20": 0.4, "rh19": 0.2, "rh21": 0.3, "rh5": 0.1},
    
    # rh21: Steady, medium tempo
    "rh21": {"rh21": 0.4, "rh20": 0.3, "rh22": 0.2, "rh6": 0.1},
    
    # rh22: Syncopated, medium tempo
    "rh22": {"rh22": 0.4, "rh21": 0.3, "rh24": 0.2, "rh7": 0.1},
    
    # rh23: Complex, varied rhythm
    "rh23": {"rh23": 0.3, "rh22": 0.3, "rh24": 0.2, "rh8": 0.1, "rh6": 0.1},
    
    # rh24: Syncopated with longer notes
    "rh24": {"rh24": 0.5, "rh23": 0.1, "rh25": 0.1, "r1": 0.3},
    
    # rh25: Steady, simple rhythm
    "rh25": {"rh25": 0.3, "rh24": 0.3, "rh1": 0.1, "r1": 0.3}
    }

    duration_probabilities_sad = {
    # rs1: Slow, steady rhythm
    "rs1": {"rs1": 0.4, "rs2": 0.3, "rs4": 0.1, "rs8": 0.2},
    
    # rs2: Very slow, long notes
    "rs2": {"rs2": 0.5, "rs1": 0.3, "rs8": 0.1, "rs12": 0.1},
    
    # rs3: Fast, repetitive rhythm
    "rs3": {"rs3": 0.5, "rs10": 0.3, "rs15": 0.1, "rs16": 0.1},
    
    # rs4: Varied rhythm with longer notes
    "rs4": {"rs4": 0.4, "rs1": 0.2, "rs5": 0.1, "rs11": 0.2, "rs7": 0.1},
    
    # rs5: Syncopated rhythm with accents
    "rs5": {"rs5": 0.4, "rs6": 0.2, "rs7": 0.2, "rs10": 0.2},
    
    # rs6: Syncopated with longer notes
    "rs6": {"rs6": 0.4, "rs5": 0.2, "rs7": 0.2, "rs10": 0.1, "rs8": 0.1},
    
    # rs7: Varied rhythm with accents
    "rs7": {"rs7": 0.4, "rs6": 0.2, "rs10": 0.1, "rs8": 0.2, "rs5": 0.1},
    
    # rs8: Slow, long notes
    "rs8": {"rs8": 0.3, "rs2": 0.3, "rs12": 0.1, "rs17": 0.1, "rs14":0.1, "rs1":0.1},
    
    # rs9: Syncopated with accents
    "rs9": {"rs9": 0.3, "rs5": 0.2, "rs11": 0.2, "rs12": 0.1, "rs13":0.1, "rs14":0.1},
    
    # rs10: Fast, repetitive rhythm
    "rs10": {"rs10": 0.3, "rs3": 0.3, "rs15": 0.2, "rs16": 0.2},
    
    # rs11: Varied rhythm with longer notes
    "rs11": {"rs11": 0.5, "rs4": 0.2, "rs12": 0.1, "rs14": 0.1,"rs13":0.1,},
    
    # rs12: Slow, long notes
    "rs12": {"rs12": 0.4, "rs2": 0.2, "rs1": 0.2, "rs11": 0.2},
    
    # rs13: Syncopated with longer notes
    "rs13": {"rs13": 0.3, "rs12": 0.2, "rs9": 0.1, "rs14": 0.1, "rs1": 0.3},
    
    # rs14: Varied rhythm with accents
    "rs14": {"rs14": 0.3, "rs3": 0.3, "rs13": 0.1, "rs17": 0.1, "rs1": 0.2},
    
    # rs15: Fast, repetitive rhythm
    "rs15": {"rs15": 0.5, "rs16": 0.3, "rs10": 0.1, "rs16": 0.1},
    
    # rs16: Syncopated with accents
    "rs16": {"rs16": 0.5, "rs15": 0.3, "rs5": 0.1, "rs6": 0.1},
    
    # rs17: Slow, long notes
    "rs17": {"rs17": 0.3, "rs8": 0.3, "rs12": 0.1, "rs18": 0.1, "rs1": 0.2},
    
    # rs18: Syncopated with accents
    "rs18": {"rs18": 0.2, "rs17": 0.2, "rs14": 0.1, "rs13": 0.1, "rs1": 0.2, "rs2":0.1, "rs3": 0.1},
}

    duration_probabilities_angry = {
    # ra1: Fast, syncopated rhythm with accents
    "ra1": {"ra1": 0.5, "ra2": 0.3, "ra11": 0.1, "ra9": 0.1},
    
    # ra2: Similar to ra1 but slightly less syncopated
    "ra2": {"ra2": 0.4, "ra1": 0.3, "ra3": 0.1, "ra6": 0.1, "ra7": 0.1},
    
    # ra3: Varied rhythm with accents
    "ra3": {"ra3": 0.4, "ra4": 0.3, "ra1": 0.1, "ra2": 0.1, "ra7": 0.1},
    
    # ra4: Fast, complex rhythm with accents
    "ra4": {"ra4": 0.4, "ra3": 0.3, "ra7": 0.1, "ra9": 0.1, "ra11": 0.1},
    
    # ra5: Steady, driving rhythm
    "ra5": {"ra5": 0.4, "ra6": 0.2, "ra2": 0.1, "ra7": 0.1, "ra10": 0.1, "ra8": 0.1},
    
    # ra6: Syncopated with longer notes
    "ra6": {"ra6": 0.4, "ra5": 0.3, "ra7": 0.1, "ra10": 0.1, "ra8": 0.1},
    
    # ra7: Fast, repetitive rhythm with accents
    "ra7": {"ra7": 0.4, "ra6": 0.1, "ra4": 0.2, "ra9": 0.1, "ra11": 0.1, "ra3": 0.1},
    
    # ra8: Slow, heavy rhythm with accents
    "ra8": {"ra8": 0.3, "ra10": 0.4, "ra5": 0.3},
    
    # ra9: Very fast, syncopated rhythm with accents
    "ra9": {"ra9": 0.5, "ra4": 0.2, "ra7": 0.1, "ra11": 0.1, "ra3": 0.1},
    
    # ra10: Steady, heavy rhythm
    "ra10": {"ra10": 0.5, "ra8": 0.2, "ra5": 0.2, "ra6": 0.1},
    
    # ra11: Very fast, repetitive rhythm
    "ra11": {"ra11": 0.4, "ra9": 0.2, "ra7": 0.1, "ra4": 0.1, "ra3": 0.1, "ra1": 0.1},
}
    
    duration_probabilities_fear = {
    # rf1: Steady, repetitive rhythm
    "rf1": {"rf1": 0.3, "rf2": 0.3, "rf16": 0.1, "rf17": 0.1, "rf12": 0.2},
    
    # rf2: Slow, steady rhythm
    "rf2": {"rf2": 0.4, "rf1": 0.3, "rf3": 0.1, "rf8": 0.1, "rf4": 0.1},
    
    # rf3: Varied rhythm with longer notes
    "rf3": {"rf3": 0.3, "rf2": 0.3, "rf9": 0.1, "rf8": 0.1, "rf1": 0.1, "rf13": 0.1},
    
    # rf4: Syncopated rhythm with accents
    "rf4": {"rf4": 0.2, "rf5": 0.2, "rf10": 0.1, "rf14": 0.1, "rf6": 0.1, "rf7": 0.1,
    "rf11": 0.1, "rf10": 0.1},
    
    # rf5: Varied rhythm with accents
    "rf5": {"rf5": 0.2, "rf4": 0.2, "rf6": 0.1, "rf7": 0.1, "rf1:": 0.1, "rf11": 0.1,"rf2": 0.1,
    "rf13": 0.1},
    
    # rf6: Syncopated with longer notes
    "rf6": {"rf6": 0.2, "rf5": 0.2, "rf11": 0.1, "rf15": 0.1, "rf4": 0.2, "rf2": 0.2},
    
    # rf7: Varied rhythm with accents
    "rf7": {"rf7": 0.3, "rf5": 0.2, "rf6": 0.1, "rf12": 0.1, "rf4": 0.1, "rf10": 0.1, "rf11": 0.1},
    
    # rf8: Slow, heavy rhythm
    "rf8": {"rf8": 0.3, "rf2": 0.3, "rf3": 0.1, "rf9": 0.2, "rf1": 0.1},
    
    # rf9: Varied rhythm with longer notes
    "rf9": {"rf9": 0.3, "rf3": 0.2, "rf8": 0.2, "rf13": 0.1, "rf1": 0.1, "rf16": 0.1},
    
    # rf10: Syncopated with accents
    "rf10": {"rf10": 0.3, "rf4": 0.2, "rf14": 0.1, "rf15": 0.1, "rf6": 0.1, "rf5": 0.2},
    
    # rf11: Syncopated with longer notes
    "rf11": {"rf11": 0.3, "rf6": 0.3, "rf15": 0.1, "rf20": 0.1, "rf10": 0.1, "rf5": 0.1},
    
    # rf12: Very fast, repetitive rhythm
    "rf12": {"rf12": 0.3, "rf18": 0.3, "rf19": 0.2, "rf20": 0.2},
    
    # rf13: Varied rhythm with accents
    "rf13": {"rf13": 0.4, "rf9": 0.2, "rf16": 0.1, "rf17": 0.1, "rf14": 0.2},
    
    # rf14: Syncopated with accents
    "rf14": {"rf14": 0.4, "rf10": 0.2, "rf15": 0.1, "rf18": 0.1, "rf13": 0.2},
    
    # rf15: Varied rhythm with accents
    "rf15": {"rf15": 0.4, "rf14": 0.3, "rf11": 0.1, "rf20": 0.1, "rf16": 0.1},
    
    # rf16: Steady, repetitive rhythm
    "rf16": {"rf16": 0.3, "rf1": 0.3, "rf17": 0.1, "rf13": 0.1, "rf2": 0.2},
    
    # rf17: Varied rhythm with accents
    "rf17": {"rf17": 0.3, "rf16": 0.3, "rf1": 0.3, "rf2": 0.1},
    
    # rf18: Very fast, repetitive rhythm
    "rf18": {"rf18": 0.5, "rf12": 0.3, "rf19": 0.1, "rf20": 0.1},
    
    # rf19: Very fast, repetitive rhythm with accents
    "rf19": {"rf19": 0.5, "rf18": 0.3, "rf12": 0.1, "rf20": 0.1},
    
    # rf20: Very fast, syncopated rhythm with accents
    "rf20": {"rf20": 0.5, "rf19": 0.3, "rf18": 0.1, "rf15": 0.1},
    }

    duration_probabilities_awe = {
    # raw1: Steady, majestic rhythm
    "raw1": {"raw1": 0.5, "raw2": 0.2, "raw6": 0.1, "raw11": 0.1,"raw5": 0.1},
    
    # raw2: Slow, grand rhythm
    "raw2": {"raw2": 0.5, "raw1": 0.3, "raw8": 0.1, "raw15": 0.1},
    
    # raw3: Fast, repetitive rhythm
    "raw3": {"raw3": 0.5, "raw11": 0.3, "raw14": 0.1, "raw5": 0.1},
    
    # raw4: Syncopated rhythm with accents
    "raw4": {"raw4": 0.5, "raw7": 0.3, "raw10": 0.1, "raw13": 0.1},
    
    # raw5: Varied rhythm with accents
    "raw5": {"raw5": 0.5, "raw1": 0.3, "raw6": 0.1, "raw11": 0.1},
    
    # raw6: Varied rhythm with longer notes
    "raw6": {"raw6": 0.5, "raw5": 0.3, "raw12": 0.1, "raw13": 0.1},
    
    # raw7: Syncopated with longer notes
    "raw7": {"raw7": 0.5, "raw4": 0.3, "raw10": 0.1, "raw13": 0.1},
    
    # raw8: Slow, grand rhythm with accents
    "raw8": {"raw8": 0.5, "raw2": 0.3, "raw15": 0.1, "raw10": 0.1},
    
    # raw9: Varied rhythm with longer notes
    "raw9": {"raw9": 0.5, "raw12": 0.3, "raw14": 0.1, "raw11": 0.1},
    
    # raw10: Syncopated with longer notes
    "raw10": {"raw10": 0.5, "raw7": 0.3, "raw4": 0.1, "raw13": 0.1},
    
    # raw11: Fast, repetitive rhythm
    "raw11": {"raw11": 0.5, "raw3": 0.3, "raw14": 0.1, "raw5": 0.1},
    
    # raw12: Varied rhythm with accents
    "raw12": {"raw12": 0.5, "raw6": 0.3, "raw9": 0.1, "raw13": 0.1},
    
    # raw13: Syncopated with longer notes
    "raw13": {"raw13": 0.5, "raw7": 0.3, "raw10": 0.1, "raw12": 0.1},
    
    # raw14: Fast, repetitive rhythm
    "raw14": {"raw14": 0.5, "raw3": 0.3, "raw11": 0.1, "raw9": 0.1},
    
    # raw15: Very slow, grand rhythm
    "raw15": {"raw15": 0.5, "raw2": 0.3, "raw8": 0.1, "raw10": 0.1},
}


    

   # Calculate the tempo multiplier to convert beats to seconds
    #tempo_multiplier = 60 / tempo

    # Initialize rhythm generation
    rhythm = []
    current_duration = 0
    current_pattern = None
    while current_duration < total_duration:
        pattern_key = get_next_pattern(current_pattern,emotion)
        pattern = rhythm_patterns[pattern_key]
        for duration,is_rest in pattern:
            rhythm.extend([(duration,is_rest)])
            current_duration += duration
        current_pattern = pattern_key

    final_rhythm = []
    current_sum = 0
    for duration, is_rest in rhythm:
        if current_sum + duration  <= total_duration:
            final_rhythm.append((duration, is_rest))
            current_sum += duration
        else:
            remaining = total_duration - current_sum
            if remaining > 0:
                final_rhythm.append((remaining, is_rest))
            break

    #print("Final rhythm is ",final_rhythm)

    return final_rhythm


def generate_chord_progression(start_pattern):
    """
    Generate a chord progression by linking predefined chord patterns.
    """
    # Predefined chord patterns
    chord_patterns = {
        'pattern_1': ['C', 'G', 'Am', 'Em','F','G','C'],  # C -> G -> Am -> F
        'pattern_2': ['F', 'G', 'C'],        # F -> G -> C
        'pattern_sad1': ['Am', 'F', 'C', 'G'],  # Am -> F -> C -> G
        'pattern_4': ['C', 'Am', 'F', 'G'],  # C -> Am -> F -> G
        'pattern_5': ['F','G','Em','Am'],
        'pattern_sad2': ['Am', 'G', 'F', 'E' ],
        'pattern_7': ['F', 'G', 'Am'],
        'pattern_sad3': ['Dm', 'E'],
        'pattern_fear1': ['Cdim', 'Ddim', 'Edim', 'Fdim'],
        'pattern_fear2': ['Cm','Gdim','Abm','Fm'],
        'pattern_fear3': ['Fm','Bm','Ebm','Am'],
        'pattern_fear4': ['Em','F','G','Am'],
        'pattern_fear5': ['Em','C#m','Gm','Bm'],
        'pattern_fear6': ['Fdim','Gdim','Adim','Bdim','Cdim'],
        'pattern_fear7': ['Bdim','Cm','Dm','Ebm'],
        'pattern_8': ['Dm', 'E', 'F', 'G'],
        'pattern_9': ['C','Bm','Am','G','F','Em','F','G'],
        
        # Add more patterns as needed
    }

    # Transition probabilities between patterns
    pattern_transitions = {
        'pattern_1': {'pattern_1': 0.6, 'pattern_2': 0.2, 'pattern_5':0.2},  # After pattern_1, go to pattern_2 or pattern_3
        'pattern_2': {'pattern_2': 0.5, 'pattern_7': 0.2, 'pattern_5': 0.3},  # After pattern_2, go to pattern_1 or pattern_4
        'pattern_sad1': {'pattern_sad1': 0.7, 'pattern_2': 0.2,'pattern_sad3':0.1},  # After pattern_3, go to pattern_2 or pattern_4
        'pattern_4': {'pattern_4': 0.7, 'pattern_1': 0.2, 'pattern_5':0.1},
        'pattern_5': {'pattern_5': 0.6, 'pattern_2': 0.1, 'pattern_7': 0.1, 'pattern_sad3':0.2},
        'pattern_7': {'pattern_2':0.1, 'pattern_5': 0.2, 'pattern_7':0.5,
                      'pattern_8': 0.1, 'pattern_sad3': 0.1},
        'pattern_8': {'pattern_5':0.4, 'pattern_9': 0.2, 'pattern_7':0.4},
        'pattern_9': {'pattern_9': 0.5, 'pattern_1': 0.2, 'pattern_5': 0.3},
        'pattern_sad2': {'pattern_sad2': 0.7, 'pattern_5':0.3},
        'pattern_sad3': {'pattern_5': 0.8, 'pattern_7': 0.2},
        'pattern_fear1':{'pattern_fear1': 0.5, 'pattern_fear6':0.5},
        'pattern_fear2':{'pattern_fear2':0.3, 'pattern_fear3':0.3, 'pattern_fear7':0.4},
        'pattern_fear3':{'pattern_fear4':0.3, 'pattern_fear5':0.3,'pattern_fear2':0.2,'pattern_fear3':0.2},
        'pattern_fear4':{'pattern_fear4':0.5,'pattern_fear5':0.2,'pattern_fear3':0.3},
        'pattern_fear5':{'pattern_fear5':0.3, 'pattern_fear4':0.2,'pattern_fear7':0.2,'pattern_fear6':0.3},
        'pattern_fear6':{'pattern_fear6':0.4,'pattern_fear1':0.4,'pattern_fear7':0.2},
        'pattern_fear7':{'pattern_fear1':0.2, 'pattern_fear2':0.3, 'pattern_fear6':0.2,'pattern_fear7':0.2,'pattern_fear4':0.1}
        # Add more transitions as needed
    }

    progression = []
    current_pattern = start_pattern

   
    # Add the chords from the current pattern to the progression
    progression.extend(chord_patterns[current_pattern])


    # Choose the next pattern based on transition probabilities
    next_pattern_options = list(pattern_transitions[current_pattern].keys())
    next_pattern_probabilities = list(pattern_transitions[current_pattern].values())
    current_pattern = random.choices(next_pattern_options, next_pattern_probabilities)[0]

    # Trim the progression to the desired length
    return progression

def get_next_note_angryV2(current_note, current_chord,previous_notes,
                          markov_order,phrase_position,key_shift = 0):
    """
    Generate the next note, emphasizing chord tones.
    """
    # Handle None for `previous_notes`
    if previous_notes is None:
        previous_notes = []

    # Clamp `phrase_position` to [0, 1]
    phrase_position = max(0, min(phrase_position, 1))

    # Define transition probabilities for higher-order Markov chains
    markov_transitions = {
        (60, 62, 64): {64: 0.3, 67: 0.3, 65: 0.4},  # C -> D -> E -> E or F
        (60, 60, 60): {60: 0.8, 67: 0.2},  # G -> A -> E -> F or D
        (72, 71, 69): {72: 0.4, 67: 0.6},  # A -> E -> F -> G or A
        (64, 64, 64): {64: 0.7, 69: 0.3},  # C -> D -> E -> F or G
        (67, 67, 67): {67: 0.8, 72: 0.2},
        (67, 67, 72): {74: 0.6, 76: 0.4},
        (64, 64, 67): {67: 0.7, 69: 0.3},
        (67, 71, 72): {72: 0.6, 74: 0.4},
        (60, 64, 67): {67: 0.7, 62: 0.3},
        (72, 72, 72): {67: 0.6, 72: 0.4},
        (60, 67, 67): {67: 0.5, 72: 0.5},
        # (67, 69, 72): {74: 0.7, 76: 0.3},
        # (57, 60, 62): {64: 0.8, 60: 0.2},
        # (62, 60, 60): {57: 0.8, 62: 0.2}
        
        # Add more patterns as needed
    }
    # Define transition probabilities for single-note transitions
    default_transitions = {
        # 53: {57: 0.3, 60: 0.5, 62: 0.2},  # F3 -> A3/C4/D4
        # 55: {57: 0.2, 60: 0.6, 62: 0.2},  # G3 -> A3/C4/D4
        # 57: {60: 0.4, 59: 0.4, 64: 0.2},  # A3 -> C4/D4/E4/F4
        # 59: {60: 0.7, 57: 0.3},  # B3 -> C4/D4/E4
        60: {60: 0.5, 62: 0.1, 64: 0.1, 67: 0.3},  # C4 -> D4/E4/G4/A4
        62: {64: 0.6, 67: 0.4},  # D4 -> E4/G4/A4
        64: {64: 0.6, 67: 0.3, 62: 0.1},  # E4 -> G4/A4/C5
        65: {64: 0.3, 67: 0.5, 60: 0.2},  # F4 -> G4/A4/C5
        67: {67: 0.6, 67: 0.2, 72: 0.1, 65: 0.1},  # G4 -> A4/C5/E4
        69: {72: 0.5, 69: 0.3, 67: 0.2},  # A4 -> C5/G4/F4
        71: {72: 0.5, 69: 0.3, 67: 0.2},  # B4 -> C5/A4
        72: {72: 0.5, 71: 0.3, 76: 0.2},
        74: {76: 0.6, 72: 0.4},
        76: {76: 0.5, 72: 0.4, 69: 0.1}  
    }
    def apply_rules(candidate_notes, current_note, previous_notes, chord_notes):
        """
        Apply musical rules to adjust the probabilities of candidate notes.
        """
        adjusted_notes = []
        adjusted_probabilities = []

        for note in candidate_notes:
            probability = 1.0  # Default probability

            # Rule 1: Resolve leading tones (B -> C)
            if current_note == 71 and note != 72:  # B4 should resolve to C5
                probability *= 0.5  #  discourage non-resolutions

            # Rule 2: Avoid large leaps (e.g., > 5 semitones)
            if abs(note - current_note) > 5:
                probability *= 0.2  # Discourage large leaps

            # Rule 3: Emphasize chord tones
            if (chord_notes and note + key_shift in chord_notes) or \
            (chord_notes and note - 12 + key_shift in chord_notes) or \
            (chord_notes and note - 24 + key_shift in chord_notes) :
                probability *= 2.0  # Strongly encourage chord tones

            # Rule 4: Resolve to tonic at the end of a phrase
            if phrase_position > 0.9 and abs(note - 60) % 12 in [1, 6]:  # Resolve to C4
                probability *= 2.0  # Strongly encourage tonic resolution

            # Rule 5 (customize the range of note (C3-C6)) 
            min_note = 48
            max_note = 84
            if note < min_note or note > max_note:
                probability *= 0.2

            # Rule 6 (Avoid the note repeat too many times)
            repeat_count = 0
            for prev_note in reversed(previous_notes + [current_note]):
                if prev_note == note:
                    repeat_count += 1
                else:
                    break
            if repeat_count >= 2:
                probability *= 0.8  # If repeat more than 2 times, reduce the probability

            interval = note - current_note
            # Rule 8 (Phrase Start)
            if phrase_position < 0.2:
                if abs(interval) >= 7:  # Large leap
                        probability *= 2.0

            if interval in [6, 7, 8, 9, 10, 11, -6, -7, -8, -9, -10, -11]:
                probability *= 1.3

            if previous_notes and phrase_position < 0.2 and len(previous_notes) >= 1:
                prev_note = previous_notes[-1]
                if abs(note - prev_note) <= 2:  # Stepwise motion
                    probability *= 0.8
                elif note in chord_notes and prev_note in chord_notes:
                    probability *= 1.0


            
            adjusted_notes.append(note)
            adjusted_probabilities.append(probability)

     
        # Normalize probabilities
        total_prob = sum(adjusted_probabilities)
        if total_prob > 0:
            adjusted_probabilities = [p / total_prob for p in adjusted_probabilities]
        else:
            adjusted_probabilities = [1.0 / len(adjusted_notes)] * len(adjusted_notes)

        return adjusted_notes, adjusted_probabilities

    # Get the current chord notes
    chord_notes = get_chord_notes(current_chord)

    # If the current note is not in the transition probabilities, default to C4
    # if current_note not in default_transitions:
    #     current_note = 60  # C4

    # Get possible next notes and their probabilities
    if len(previous_notes) >= markov_order:
        last_pattern = tuple(previous_notes[-markov_order:])
        if last_pattern in markov_transitions:
            candidate_notes = list(markov_transitions[last_pattern].keys())
            candidate_probabilities = list(markov_transitions[last_pattern].values())
        else:
            # Fallback to lower-order Markov chain or single-note transitions
            if len(previous_notes) >= markov_order - 1:
                last_pattern = tuple(previous_notes[-(markov_order - 1):])
                if last_pattern in markov_transitions:
                    candidate_notes = list(markov_transitions[last_pattern].keys())
                    candidate_probabilities = list(markov_transitions[last_pattern].values())
                else:
                    # Fallback to default single-note transitions
                    candidate_notes = list(default_transitions.get(current_note, {}).keys())
                    candidate_probabilities = list(default_transitions.get(current_note, {}).values())
            else:
                # Fallback to default single-note transitions
                candidate_notes = list(default_transitions.get(current_note, {}).keys())
                candidate_probabilities = list(default_transitions.get(current_note, {}).values())
    else:
        # Fallback to default single-note transitions
        candidate_notes = list(default_transitions.get(current_note, {}).keys())
        candidate_probabilities = list(default_transitions.get(current_note, {}).values())

    # Apply rule-based adjustments
    candidate_notes, candidate_probabilities = apply_rules(candidate_notes, current_note, previous_notes, chord_notes)

    # Choose the next note based on adjusted probabilities
    if candidate_notes:
        next_note = random.choices(candidate_notes, candidate_probabilities)[0]
    else:
        next_note = 60  # Default to C4 if no candidates are available

    return next_note

def get_next_note_happyV2(current_note, current_chord,previous_notes,
                          markov_order,phrase_position,key_shift=0):
    """
    Generate the next note, emphasizing chord tones.
    """
    # Handle None for `previous_notes`
    if previous_notes is None:
        previous_notes = []

    # Clamp `phrase_position` to [0, 1]
    phrase_position = max(0, min(phrase_position, 1))

    # Define transition probabilities for higher-order Markov chains
    markov_transitions = {
        (60, 62, 64): {67: 0.7, 69: 0.3},  # C -> D -> E -> E or F
        (67, 72, 74): {76: 0.9, 72: 0.1},  # G -> A -> E -> F or D
        (72, 71, 69): {69: 0.6, 67: 0.4},  # A -> E -> F -> G or A
        (64, 65, 67): {67: 0.5, 72: 0.5},  # C -> D -> E -> F or G
        (64, 67, 69): {69: 0.6, 72: 0.4},
        (64, 62, 60): {60: 0.7, 67: 0.3},
        (60, 64, 67): {72: 0.8, 69: 0.2},
        (60, 67, 72): {72: 0.8, 74: 0.2},
        (62, 64, 67): {67: 0.7, 72: 0.3},
        (67, 65, 64): {62: 0.6, 65: 0.4},
        (60, 67, 67): {69: 0.7, 72: 0.3}
        # Add more patterns as needed
    }
    # Define transition probabilities for single-note transitions
    default_transitions = {
        # 53: {57: 0.3, 60: 0.5, 62: 0.2},  # F3 -> A3/C4/D4
        # 55: {57: 0.2, 60: 0.6, 62: 0.2},  # G3 -> A3/C4/D4
        # 57: {60: 0.4, 62: 0.3, 64: 0.2, 65: 0.1},  # A3 -> C4/D4/E4/F4
        # 59: {60: 0.6, 62: 0.3, 64: 0.1},  # B3 -> C4/D4/E4
        60: {62: 0.3, 64: 0.3, 67: 0.3, 72: 0.1},  # C4 -> D4/E4/G4/A4
        62: {64: 0.6, 60: 0.3, 67: 0.1},  # D4 -> E4/G4/A4
        64: {65: 0.3, 67: 0.3, 62: 0.3, 72: 0.1},  # E4 -> G4/A4/C5
        65: {64: 0.4, 67: 0.4, 60: 0.2},  # F4 -> G4/A4/C5
        67: {65: 0.3, 72: 0.4, 67: 0.2, 69: 0.1},  # G4 -> A4/C5/E4
        69: {72: 0.5, 71: 0.3, 67: 0.2},  # A4 -> C5/G4/F4
        71: {72: 0.5, 69: 0.3, 67: 0.2},  # B4 -> C5/A4
        72: {74: 0.4, 71: 0.3, 67: 0.3},
        74: {76: 0.6, 72: 0.4},
        76: {74: 0.4, 72: 0.4, 69: 0.2}  
    }
    def apply_rules(candidate_notes, current_note, previous_notes, chord_notes):
        """
        Apply musical rules to adjust the probabilities of candidate notes.
        """
        adjusted_notes = []
        adjusted_probabilities = []

        for note in candidate_notes:
            #print(note)
            probability = 1.0  # Default probability

            # Rule 1: Resolve leading tones (B -> C)
            if current_note == 71 and note != 72:  # B4 should resolve to C5
                probability *= 0.2  # Strongly discourage non-resolutions

            # Rule 2: Avoid large leaps (e.g., > 5 semitones)
            if abs(note - current_note) > 12:
                probability *= 0.5  # Discourage large leaps

            # Rule 3: Emphasize chord tones
            if (chord_notes and note + key_shift in chord_notes) or \
            (chord_notes and note - 12 + key_shift in chord_notes) or \
            (chord_notes and note - 24 + key_shift in chord_notes) :
                probability *= 2.0  # Strongly encourage chord tones

            # Rule 4: Resolve to tonic at the end of a phrase
            if phrase_position > 0.85 and note % 12 == 60 % 12:  # Resolve to C4
                probability *= 1.5  # Strongly encourage tonic resolution

            # Rule 5 (customize the range of note (C3-C6)) 
            min_note = 48
            max_note = 84
            if note < min_note or note > max_note:
                probability *= 0.2

            # Rule 6 (Avoid the note repeat too many times)
            repeat_count = 0
            for prev_note in reversed(previous_notes + [current_note]):
                if prev_note == note:
                    repeat_count += 1
                else:
                    break
            if repeat_count >= 2:
                probability *= 0.8  # If repeat more than 2 times, reduce the probability

            # Rule 7 (Emphasize the stepwise motion)
            interval = note - current_note
            # Rule 8 (Phrase Start)
            if phrase_position < 0.2:
                if interval in [4, 5, 7] and interval > 0:  # Large leap
                        probability *= 1.5

            if interval in [2, 3, 4, 5, 7]:
                probability *= 1.3

            if previous_notes and phrase_position < 0.2 and len(previous_notes) >= 1:
                prev_note = previous_notes[-1]
                if abs(note - prev_note) <= 2:  # Stepwise motion
                    probability *= 1.3
                elif note in chord_notes and prev_note in chord_notes:
                    probability *= 1.5

            adjusted_notes.append(note)
            adjusted_probabilities.append(probability)



        # Normalize probabilities
        total_prob = sum(adjusted_probabilities)
        if total_prob > 0:
            adjusted_probabilities = [p / total_prob for p in adjusted_probabilities]
        else:
            adjusted_probabilities = [1.0 / len(adjusted_notes)] * len(adjusted_notes)

        return adjusted_notes, adjusted_probabilities

    # Get the current chord notes
    chord_notes = get_chord_notesV2(current_chord)

    # If the current note is not in the transition probabilities, default to C4
    # if current_note not in default_transitions:
    #     current_note = 60  # C4

    # Get possible next notes and their probabilities
    if len(previous_notes) >= markov_order:
        last_pattern = tuple(previous_notes[-markov_order:])
        if last_pattern in markov_transitions:
            candidate_notes = list(markov_transitions[last_pattern].keys())
            candidate_probabilities = list(markov_transitions[last_pattern].values())
        else:
            # Fallback to lower-order Markov chain or single-note transitions
            if len(previous_notes) >= markov_order - 1:
                last_pattern = tuple(previous_notes[-(markov_order - 1):])
                if last_pattern in markov_transitions:
                    candidate_notes = list(markov_transitions[last_pattern].keys())
                    candidate_probabilities = list(markov_transitions[last_pattern].values())
                else:
                    # Fallback to default single-note transitions
                    candidate_notes = list(default_transitions.get(current_note, {}).keys())
                    candidate_probabilities = list(default_transitions.get(current_note, {}).values())
            else:
                # Fallback to default single-note transitions
                candidate_notes = list(default_transitions.get(current_note, {}).keys())
                candidate_probabilities = list(default_transitions.get(current_note, {}).values())
    else:
        # Fallback to default single-note transitions
        candidate_notes = list(default_transitions.get(current_note, {}).keys())
        candidate_probabilities = list(default_transitions.get(current_note, {}).values())

    # Apply rule-based adjustments
    candidate_notes, candidate_probabilities = apply_rules(candidate_notes, current_note, previous_notes, chord_notes)

    # Choose the next note based on adjusted probabilities
    if candidate_notes:
        next_note = random.choices(candidate_notes, candidate_probabilities)[0]
    else:       
        if chord_notes:
            candidate_notes = chord_notes + 12
            candidate_probabilities = [1.0 / len(chord_notes)] * len(chord_notes)
        else:
            candidate_notes = [current_note]
            candidate_probabilities = [1.0]

    #print(next_note)
    return next_note

def get_next_note_happyV3(current_note, current_chord, previous_notes, markov_order, phrase_position):
    """
    Generate the next note, emphasizing chord tones.
    """
    # Handle None for `previous_notes`
    if previous_notes is None:
        previous_notes = []

    # Clamp `phrase_position` to [0, 1]
    phrase_position = max(0, min(phrase_position, 1))

    # Define transition probabilities for higher-order Markov chains
    markov_transitions = {
        (60, 62, 64): {67: 0.7, 69: 0.3},
        (67, 72, 74): {76: 0.9, 72: 0.1},
        (72, 71, 69): {69: 0.6, 67: 0.4},
        (64, 65, 67): {67: 0.5, 72: 0.5},
        (64, 67, 69): {69: 0.6, 72: 0.4},
        (64, 62, 60): {60: 0.7, 67: 0.3},
        (60, 64, 67): {72: 0.8, 69: 0.2},
        (60, 67, 72): {72: 0.8, 74: 0.2},
        (62, 64, 67): {67: 0.7, 72: 0.3},
        (67, 65, 64): {62: 0.6, 65: 0.4},
        (60, 67, 67): {69: 0.7, 72: 0.3}
    }

    # Define transition probabilities for single-note transitions
    default_transitions = {
        60: {62: 0.3, 64: 0.3, 67: 0.3, 72: 0.1},  # C4 -> D4/E4/G4/A4
        62: {64: 0.6, 60: 0.3, 67: 0.1},  # D4 -> E4/G4/A4
        64: {65: 0.3, 67: 0.3, 62: 0.3, 72: 0.1},  # E4 -> G4/A4/C5
        65: {64: 0.4, 67: 0.4, 60: 0.2},  # F4 -> G4/A4/C5
        67: {65: 0.3, 72: 0.4, 67: 0.2, 69: 0.1},  # G4 -> A4/C5/E4
        69: {72: 0.5, 71: 0.3, 67: 0.2},  # A4 -> C5/G4/F4
        71: {72: 0.5, 69: 0.3, 67: 0.2},  # B4 -> C5/A4
        72: {74: 0.4, 71: 0.3, 67: 0.3},
        74: {76: 0.6, 72: 0.4},
        76: {74: 0.4, 72: 0.4, 69: 0.2}
    }

    def apply_rules(candidate_notes, current_note, previous_notes, chord_notes):
        """
        Apply musical rules to adjust the probabilities of candidate notes.
        """
        # If no candidate notes, return a default (e.g., chord tones or current_note)
        if not candidate_notes:
            if chord_notes:
                return chord_notes, [1.0 / len(chord_notes)] * len(chord_notes)
            else:
                return [current_note], [1.0]  # Default to current note

        adjusted_notes = []
        adjusted_probabilities = []

        for note in candidate_notes:
            probability = 1.0  # Default probability

            # Rule 1: Resolve leading tones (B -> C)
            if current_note == 71 and note != 72:
                probability *= 0.1

            # Rule 2: Avoid large leaps (e.g., > 12 semitones)
            if abs(note - current_note) > 12:
                probability *= 0.5

            # Rule 3: Emphasize chord tones
            if (chord_notes and note in chord_notes) or \
               (chord_notes and note - 12 in chord_notes):
                probability *= 2.0

            # Rule 4: Resolve to tonic at the end of a phrase
            if phrase_position > 0.75 and note == 60:
                probability *= 2.0

            adjusted_notes.append(note)
            adjusted_probabilities.append(probability)

        # Normalize probabilities
        total_prob = sum(adjusted_probabilities)
        if total_prob > 0:
            adjusted_probabilities = [p / total_prob for p in adjusted_probabilities]
        else:
            # If all probabilities are zero, assign equal probabilities
            adjusted_probabilities = [1.0 / len(adjusted_notes)] * len(adjusted_notes)

        return adjusted_notes, adjusted_probabilities

    # Get the current chord notes
    chord_notes = get_chord_notes(current_chord)

    # Get possible next notes and their probabilities
    if len(previous_notes) >= markov_order:
        last_pattern = tuple(previous_notes[-markov_order:])
        if last_pattern in markov_transitions:
            candidate_notes = list(markov_transitions[last_pattern].keys())
            candidate_probabilities = list(markov_transitions[last_pattern].values())
        else:
            # Fallback to lower-order Markov chain or single-note transitions
            if len(previous_notes) >= markov_order - 1:
                last_pattern = tuple(previous_notes[-(markov_order - 1):])
                if last_pattern in markov_transitions:
                    candidate_notes = list(markov_transitions[last_pattern].keys())
                    candidate_probabilities = list(markov_transitions[last_pattern].values())
                else:
                    candidate_notes = list(default_transitions.get(current_note, {}).keys())
                    candidate_probabilities = list(default_transitions.get(current_note, {}).values())
            else:
                candidate_notes = list(default_transitions.get(current_note, {}).keys())
                candidate_probabilities = list(default_transitions.get(current_note, {}).values())
    else:
        candidate_notes = list(default_transitions.get(current_note, {}).keys())
        candidate_probabilities = list(default_transitions.get(current_note, {}).values())

    # If still no candidates, use chord notes or default to current_note
    if not candidate_notes:
        if chord_notes:
            candidate_notes = chord_notes
            candidate_probabilities = [1.0 / len(chord_notes)] * len(chord_notes)
        else:
            candidate_notes = [current_note]
            candidate_probabilities = [1.0]

    # Apply rule-based adjustments
    candidate_notes, candidate_probabilities = apply_rules(candidate_notes, current_note, previous_notes, chord_notes)

    # Choose the next note based on adjusted probabilities
    next_note = random.choices(candidate_notes, candidate_probabilities)[0]
    return next_note

def get_next_note_sadV2(current_note, current_chord,previous_notes,
                          markov_order,phrase_position,key_shift = 0):
    """
    Generate the next note, emphasizing chord tones.
    """
    # Handle None for `previous_notes`
    if previous_notes is None:
        previous_notes = []

    # Clamp `phrase_position` to [0, 1]
    phrase_position = max(0, min(phrase_position, 1))

    # Define transition probabilities for higher-order Markov chains
    markov_transitions = {
        (60, 62, 64): {64: 0.5, 67: 0.3, 62: 0.2},  # C -> D -> E -> E or F
        (67, 72, 74): {76: 0.8, 72: 0.2},  # G -> A -> E -> F or D
        (72, 71, 69): {69: 0.8, 67: 0.2},  # A -> E -> F -> G or A
        (64, 65, 67): {67: 0.5, 65: 0.5},  # C -> D -> E -> F or G
        (64, 67, 69): {69: 0.8, 67: 0.2},
        (64, 62, 60): {60: 0.6, 59: 0.4},
        (60, 64, 67): {69: 0.8, 67: 0.2},
        (60, 67, 72): {71: 0.6, 74: 0.4},
        (62, 64, 67): {69: 0.7, 64: 0.3},
        (67, 65, 64): {64: 0.6, 62: 0.4},
        (60, 67, 67): {65: 0.5, 69: 0.5},
        (67, 69, 72): {74: 0.7, 76: 0.3},
        (57, 60, 62): {64: 0.8, 60: 0.2},
        (62, 60, 60): {57: 0.8, 62: 0.2}
        
        # Add more patterns as needed
    }
    # Define transition probabilities for single-note transitions
    default_transitions = {
        # 53: {57: 0.3, 60: 0.5, 62: 0.2},  # F3 -> A3/C4/D4
        # 55: {57: 0.2, 60: 0.6, 62: 0.2},  # G3 -> A3/C4/D4
        57: {60: 0.4, 59: 0.4, 64: 0.2},  # A3 -> C4/D4/E4/F4
        59: {60: 0.7, 57: 0.3},  # B3 -> C4/D4/E4
        60: {62: 0.4, 64: 0.3, 67: 0.3},  # C4 -> D4/E4/G4/A4
        62: {64: 0.6, 60: 0.4},  # D4 -> E4/G4/A4
        64: {65: 0.2, 67: 0.4, 62: 0.3, 72: 0.1},  # E4 -> G4/A4/C5
        65: {64: 0.5, 67: 0.3, 60: 0.2},  # F4 -> G4/A4/C5
        67: {65: 0.3, 72: 0.2, 67: 0.1, 69: 0.4},  # G4 -> A4/C5/E4
        69: {72: 0.5, 71: 0.3, 64: 0.2},  # A4 -> C5/G4/F4
        71: {72: 0.5, 69: 0.3, 67: 0.2},  # B4 -> C5/A4
        72: {74: 0.4, 71: 0.3, 67: 0.3},
        74: {76: 0.6, 72: 0.4},
        76: {74: 0.4, 72: 0.4, 69: 0.2}  
    }
    def apply_rules(candidate_notes, current_note, previous_notes, chord_notes):
        """
        Apply musical rules to adjust the probabilities of candidate notes.
        """
        adjusted_notes = []
        adjusted_probabilities = []

        for note in candidate_notes:
            probability = 1.0  # Default probability

            # Rule 1: Resolve leading tones (B -> C)
            if current_note == 71 and note != 72:  # B4 should resolve to C5
                probability *= 0.1  # Strongly discourage non-resolutions

            # Rule 2: Avoid large leaps (e.g., > 5 semitones)
            if abs(note - current_note) > 5:
                probability *= 0.2  # Discourage large leaps

            # Rule 3: Emphasize chord tones
            if (chord_notes and note + key_shift in chord_notes) or \
            (chord_notes and note - 12 + key_shift in chord_notes) or \
            (chord_notes and note - 24 + key_shift in chord_notes) :
                probability *= 2.0  # Strongly encourage chord tones

            # Rule 4: Resolve to tonic at the end of a phrase
            if phrase_position > 0.85 and note in chord_notes and (note % 12 == (60 + 3) % 12 or note % 12 == (60 - 9) % 12):  # Resolve to C4
                probability *= 1.3  # Strongly encourage tonic resolution

            # Rule 5 (customize the range of note (C3-C6)) 
            min_note = 48
            max_note = 84
            if note < min_note or note > max_note:
                probability *= 0.2

            # Rule 6 (Avoid the note repeat too many times)
            repeat_count = 0
            for prev_note in reversed(previous_notes + [current_note]):
                if prev_note == note:
                    repeat_count += 1
                else:
                    break
            if repeat_count >= 2:
                probability *= 0.8  # If repeat more than 2 times, reduce the probability

            # Rule 7 (Emphasize the stepwise motion)
            interval = note - current_note

            if phrase_position < 0.2:
                if interval in [-2, -3, -4]:  # Large leap
                        probability *= 1.3

            if interval in [-2, -3, -4, 3, 4]:
                probability *= 1.3

            if previous_notes and phrase_position < 0.2 and len(previous_notes) >= 1:
                prev_note = previous_notes[-1]
                if abs(note - prev_note) <= 2:  # Stepwise motion
                    probability *= 1.3
                elif note in chord_notes and prev_note in chord_notes:
                    probability *= 1.1

            adjusted_notes.append(note)
            adjusted_probabilities.append(probability)

        # Normalize probabilities
        total_prob = sum(adjusted_probabilities)
        if total_prob > 0:
            adjusted_probabilities = [p / total_prob for p in adjusted_probabilities]
        else:
            adjusted_probabilities = [1.0 / len(adjusted_notes)] * len(adjusted_notes)

        return adjusted_notes, adjusted_probabilities

    # Get the current chord notes
    chord_notes = get_chord_notesV2(current_chord)

    # If the current note is not in the transition probabilities, default to C4
    # if current_note not in default_transitions:
    #     current_note = 60  # C4

    # Get possible next notes and their probabilities
    if len(previous_notes) >= markov_order:
        last_pattern = tuple(previous_notes[-markov_order:])
        if last_pattern in markov_transitions:
            candidate_notes = list(markov_transitions[last_pattern].keys())
            candidate_probabilities = list(markov_transitions[last_pattern].values())
        else:
            # Fallback to lower-order Markov chain or single-note transitions
            if len(previous_notes) >= markov_order - 1:
                last_pattern = tuple(previous_notes[-(markov_order - 1):])
                if last_pattern in markov_transitions:
                    candidate_notes = list(markov_transitions[last_pattern].keys())
                    candidate_probabilities = list(markov_transitions[last_pattern].values())
                else:
                    # Fallback to default single-note transitions
                    candidate_notes = list(default_transitions.get(current_note, {}).keys())
                    candidate_probabilities = list(default_transitions.get(current_note, {}).values())
            else:
                # Fallback to default single-note transitions
                candidate_notes = list(default_transitions.get(current_note, {}).keys())
                candidate_probabilities = list(default_transitions.get(current_note, {}).values())
    else:
        # Fallback to default single-note transitions
        candidate_notes = list(default_transitions.get(current_note, {}).keys())
        candidate_probabilities = list(default_transitions.get(current_note, {}).values())

    # Apply rule-based adjustments
    candidate_notes, candidate_probabilities = apply_rules(candidate_notes, current_note, previous_notes, chord_notes)

    # Choose the next note based on adjusted probabilities
    if candidate_notes:
        next_note = random.choices(candidate_notes, candidate_probabilities)[0]
    else:
        next_note = 60  # Default to C4 if no candidates are available

    return next_note

def get_next_note_fearV2(current_note, current_chord,previous_notes,
                          markov_order,phrase_position,key_shift = 0):
    """
    Generate the next note, emphasizing chord tones.
    """
    # Handle None for `previous_notes`
    if previous_notes is None:
        previous_notes = []

    # Clamp `phrase_position` to [0, 1]
    phrase_position = max(0, min(phrase_position, 1))

    # Define transition probabilities for higher-order Markov chains
    markov_transitions = {
        (60, 62, 64): {67: 0.5, 66: 0.3, 65: 0.2},  # C -> D -> E -> G (chord tone) or F# (tritone)
        (67, 72, 74): {76: 0.6, 75: 0.2, 72: 0.2},  # G -> C -> D -> E or D# (chromatic)
        (72, 71, 69): {69: 0.5, 67: 0.3, 66: 0.2},  # C -> B -> A -> A or G or F#
        (64, 65, 67): {67: 0.4, 66: 0.3, 69: 0.3},  # E -> F -> G -> G or F# or A
        (64, 67, 69): {69: 0.5, 68: 0.3, 72: 0.2},  # E -> G -> A -> A or G# or C
        (64, 62, 60): {60: 0.6, 58: 0.2, 55: 0.2},  # E -> D -> C -> C or Bb or G (low register)
        (60, 64, 67): {72: 0.6, 71: 0.2, 69: 0.2},  # C -> E -> G -> C or B or A
        (60, 67, 72): {72: 0.5, 71: 0.3, 74: 0.2},  # C -> G -> C -> C or B or D
        (62, 64, 67): {67: 0.5, 66: 0.3, 69: 0.2},  # D -> E -> G -> G or F# or A
        (67, 65, 64): {62: 0.5, 60: 0.3, 58: 0.2},  # G -> F -> E -> D or C or Bb
        (60, 67, 67): {69: 0.6, 67: 0.2, 66: 0.2},  # C -> G -> G -> A or G or F#
        (60, 61, 62): {63: 0.6, 64: 0.2, 65: 0.2},  # C -> C# -> D -> D# or E or F (chromatic)
        (64, 63, 62): {61: 0.6, 60: 0.2, 59: 0.2},  # E -> D# -> D -> C# or C or B (chromatic descent)
        
    }
    # Define transition probabilities for single-note transitions
    default_transitions = {

        60: {63: 0.3, 61: 0.3, 66: 0.4},  # C4 -> D4, C#4 (chromatic), F#4 (tritone), Bb3 (low)
        61: {64: 0.5, 60: 0.2, 68: 0.3},
        62: {65: 0.4, 63: 0.3, 60: 0.2, 66: 0.1},  # D4 -> E4, D#4 (chromatic), C4, F#4 (tritone)
        63: {63: 0.2, 66: 0.6, 70: 0.2},
        64: {67: 0.3, 63: 0.3, 67: 0.2, 62: 0.2},  # E4 -> F4, D#4 (chromatic), G4, D4
        65: {64: 0.4, 66: 0.3, 68: 0.3},  # F4 -> E4, F#4 (tritone), G4, C4
        66: {70: 0.2, 69: 0.5, 67: 0.2, 72: 0.1},
        67: {65: 0.3, 66: 0.3, 72: 0.2, 69: 0.2},  # G4 -> F4, F#4 (tritone), C5, A4
        68: {70: 0.4, 74: 0.3, 67: 0.2, 69: 0.1},
        69: {72: 0.5, 71: 0.3, 67: 0.2},  # A4 -> C5, B4, G4
        70: {73: 0.4, 65: 0.3, 64: 0.3},
        71: {74: 0.6, 68: 0.2, 67: 0.2},  # B4 -> C5, A4, G4
        72: {75: 0.4, 71: 0.3, 66: 0.3},  # C5 -> D5, B4, A4
        73: {76: 0.3, 67: 0.4, 72: 0.3},
        74: {75: 0.4, 73: 0.1, 68: 0.5},  # D5 -> E5, C5
        75: {74: 0.4, 76: 0.2, 66: 0.4},
        76: {74: 0.4, 75: 0.4, 67: 0.2},  # E5 -> D5, C5, A4
    }
    def apply_rules(candidate_notes, current_note, previous_notes, chord_notes):
        """
        Apply musical rules to adjust the probabilities of candidate notes.
        """
        adjusted_notes = []
        adjusted_probabilities = []

        for note in candidate_notes:
            probability = 1.0  # Default probability

            # Rule 1: Resolve leading tones (B -> C)
            if current_note == 71 and note != 72:  # B4 should resolve to C5
                probability *= 0.6  # Strongly discourage non-resolutions

            # Rule 2: Avoid large leaps (e.g., > 5 semitones)
            if abs(note - current_note) > 12:
                probability *= 0.5  # Discourage large leaps

            # Rule 3: Emphasize chord tones
            if (chord_notes and note + key_shift in chord_notes) or \
            (chord_notes and note - 12 + key_shift in chord_notes) or \
            (chord_notes and note - 24 + key_shift in chord_notes) :
                probability *= 2.0  # Strongly encourage chord tones

            # Rule 4: Resolve to tonic at the end of a phrase
            if phrase_position > 0.85 and abs(note - 60) % 12 in [1, 6, 11]:  # Resolve to C4
                probability *= 1.3  # Strongly encourage tonic resolution

            # Rule 5 (customize the range of note (C3-C6)) 
            min_note = 48
            max_note = 84
            if note < min_note or note > max_note:
                probability *= 0.2

            # Rule 6 (Avoid the note repeat too many times)
            repeat_count = 0
            for prev_note in reversed(previous_notes + [current_note]):
                if prev_note == note:
                    repeat_count += 1
                else:
                    break
            if repeat_count >= 2:
                probability *= 0.8  # If repeat more than 2 times, reduce the probability

            # Rule 7 (Emphasize the stepwise motion)
            interval = note - current_note

            if phrase_position < 0.2:
                if interval in [1, 6, 11]:  # Large leap
                        probability *= 1.3

            if interval in [1, 6, 11, -1, -6, -11]:
                probability *= 1.3

            if previous_notes and phrase_position < 0.2 and len(previous_notes) >= 1:
                prev_note = previous_notes[-1]
                if abs(note - prev_note) <= 2:  # Stepwise motion
                    probability *= 0.8
                elif note in chord_notes and prev_note in chord_notes:
                    probability *= 1.1

            adjusted_notes.append(note)
            adjusted_probabilities.append(probability)

        # Normalize probabilities
        total_prob = sum(adjusted_probabilities)
        if total_prob > 0:
            adjusted_probabilities = [p / total_prob for p in adjusted_probabilities]
        else:
            adjusted_probabilities = [1.0 / len(adjusted_notes)] * len(adjusted_notes)

        return adjusted_notes, adjusted_probabilities

    # Get the current chord notes
    chord_notes = get_chord_notesV2(current_chord)

    # If the current note is not in the transition probabilities, default to C4
    # if current_note not in default_transitions:
    #     current_note = 60  # C4

    # Get possible next notes and their probabilities
    if len(previous_notes) >= markov_order:
        last_pattern = tuple(previous_notes[-markov_order:])
        if last_pattern in markov_transitions:
            candidate_notes = list(markov_transitions[last_pattern].keys())
            candidate_probabilities = list(markov_transitions[last_pattern].values())
        else:
            # Fallback to lower-order Markov chain or single-note transitions
            if len(previous_notes) >= markov_order - 1:
                last_pattern = tuple(previous_notes[-(markov_order - 1):])
                if last_pattern in markov_transitions:
                    candidate_notes = list(markov_transitions[last_pattern].keys())
                    candidate_probabilities = list(markov_transitions[last_pattern].values())
                else:
                    # Fallback to default single-note transitions
                    candidate_notes = list(default_transitions.get(current_note, {}).keys())
                    candidate_probabilities = list(default_transitions.get(current_note, {}).values())
            else:
                # Fallback to default single-note transitions
                candidate_notes = list(default_transitions.get(current_note, {}).keys())
                candidate_probabilities = list(default_transitions.get(current_note, {}).values())
    else:
        # Fallback to default single-note transitions
        candidate_notes = list(default_transitions.get(current_note, {}).keys())
        candidate_probabilities = list(default_transitions.get(current_note, {}).values())

    # Apply rule-based adjustments
    candidate_notes, candidate_probabilities = apply_rules(candidate_notes, current_note, previous_notes, chord_notes)

    # Choose the next note based on adjusted probabilities
    if candidate_notes:
        next_note = random.choices(candidate_notes, candidate_probabilities)[0]
    else:
        next_note = 60  # Default to C4 if no candidates are available

    

    return next_note 

def get_next_note_aweV2(current_note, current_chord,previous_notes,
                          markov_order,phrase_position,key_shift = 0):
    """
    Generate the next note, emphasizing chord tones.
    """
    # Handle None for `previous_notes`
    if previous_notes is None:
        previous_notes = []

    # Clamp `phrase_position` to [0, 1]
    phrase_position = max(0, min(phrase_position, 1))

    # Define transition probabilities for higher-order Markov chains
    markov_transitions = {
        (60, 62, 64): {67: 0.6, 72: 0.3, 64: 0.1},  # C -> D -> E -> G (chord tone) or F# (tritone)
        (67, 72, 74): {76: 0.6, 74: 0.2, 72: 0.2},  # G -> C -> D -> E or D# (chromatic)
        (72, 71, 69): {69: 0.7, 67: 0.3},  # C -> B -> A -> A or G or F#
        (64, 65, 67): {67: 0.6, 72: 0.2, 69: 0.2},  # E -> F -> G -> G or F# or A
        (64, 67, 69): {69: 0.6, 67: 0.2, 72: 0.2},  # E -> G -> A -> A or G# or C
        (64, 62, 60): {60: 0.6, 64: 0.2, 55: 0.2},  # E -> D -> C -> C or Bb or G (low register)
        (60, 64, 67): {72: 0.3, 67: 0.4, 69: 0.3},  # C -> E -> G -> C or B or A
        (60, 67, 72): {72: 0.5, 71: 0.3, 67: 0.2},  # C -> G -> C -> C or B or D
        (62, 64, 67): {67: 0.5, 62: 0.2, 69: 0.3},  # D -> E -> G -> G or F# or A
        (67, 67, 72): {74: 0.5, 76: 0.3, 71: 0.2},  # G -> F -> E -> D or C or Bb
        (60, 67, 67): {67: 0.6, 69: 0.2, 65: 0.2},  # C -> G -> G -> A or G or F#
        (60, 62, 64): {62: 0.3, 64: 0.3, 67: 0.4},  # C -> C# -> D -> D# or E or F (chromatic)
        (64, 62, 62): {60: 0.9, 55: 0.1},  # E -> D# -> D -> C# or C or B (chromatic descent)
        (57, 60, 62): {62: 0.4, 64: 0.4, 60: 0.2},
        
    }
    # Define transition probabilities for single-note transitions
    default_transitions = {
        55: {57: 0.3, 60: 0.5, 62: 0.2},
        57: {59: 0.3, 57: 0.3, 60: 0.3, 62: 0.1},
        59: {60: 0.5, 57: 0.3, 55: 0.1, 62: 0.1},
        60: {60: 0.3, 67: 0.3, 64: 0.2, 62: 0.2},  # C4 -> D4, C#4 (chromatic), F#4 (tritone), Bb3 (low)
        #61: {64: 0.5, 60: 0.2, 68: 0.3},
        62: {62: 0.3, 64: 0.3, 60: 0.2, 67: 0.2},  # D4 -> E4, D#4 (chromatic), C4, F#4 (tritone)
        #63: {63: 0.2, 66: 0.6, 70: 0.2},
        64: {64: 0.3, 62: 0.3, 67: 0.2, 69: 0.2},  # E4 -> F4, D#4 (chromatic), G4, D4
        65: {64: 0.4, 67: 0.3, 72: 0.3},  # F4 -> E4, F#4 (tritone), G4, C4
        #66: {70: 0.2, 69: 0.5, 67: 0.2, 72: 0.1},
        67: {65: 0.2, 67: 0.3, 72: 0.2, 69: 0.2, 60 : 0.1},  # G4 -> F4, F#4 (tritone), C5, A4
        #68: {70: 0.4, 74: 0.3, 67: 0.2, 69: 0.1},
        69: {72: 0.4, 71: 0.3, 69: 0.2, 67: 0.1},  # A4 -> C5, B4, G4
        #70: {73: 0.4, 65: 0.3, 64: 0.3},
        71: {72: 0.5, 69: 0.2, 67: 0.2, 74: 0.1},  # B4 -> C5, A4, G4
        72: {72: 0.3, 71: 0.3, 67: 0.2, 74: 0.2},  # C5 -> D5, B4, A4
        #73: {76: 0.3, 67: 0.4, 72: 0.3},
        74: {76: 0.4, 72: 0.3, 64: 0.3},  # D5 -> E5, C5
        #75: {74: 0.4, 76: 0.2, 66: 0.4},
        76: {76: 0.4, 74: 0.4, 72: 0.2},  # E5 -> D5, C5, A4
    }
    def apply_rules(candidate_notes, current_note, previous_notes, chord_notes):
        """
        Apply musical rules to adjust the probabilities of candidate notes.
        """
        adjusted_notes = []
        adjusted_probabilities = []

        for note in candidate_notes:
            probability = 1.0  # Default probability

            # Rule 1: Resolve leading tones (B -> C)
            if current_note == 71 and note != 72:  # B4 should resolve to C5
                probability *= 0.1  # Strongly discourage non-resolutions

            # Rule 2: Avoid large leaps (e.g., > 5 semitones)
            if abs(note - current_note) > 12:
                probability *= 0.5  # Discourage large leaps

            # Rule 3: Emphasize chord tones
            if (chord_notes and note + key_shift in chord_notes) or \
            (chord_notes and note - 12 + key_shift in chord_notes) or \
            (chord_notes and note - 24 + key_shift in chord_notes) :
                probability *= 2.0  # Strongly encourage chord tones

            # Rule 4: Resolve to tonic at the end of a phrase
            if phrase_position > 0.85 and (note % 12 == 60 % 12 or note % 12 == (67 + key_shift) % 12 or note in chord_notes):  # Resolve to C4
                probability *= 1.3  # Strongly encourage tonic resolution

            # Rule 5 (customize the range of note (C3-C6)) 
            min_note = 48
            max_note = 84
            if note < min_note or note > max_note:
                probability *= 0.2

            # Rule 6 (Avoid the note repeat too many times)
            repeat_count = 0
            for prev_note in reversed(previous_notes + [current_note]):
                if prev_note == note:
                    repeat_count += 1
                else:
                    break
            if repeat_count >= 2:
                probability *= 0.8  # If repeat more than 2 times, reduce the probability

            # Rule 7 (Emphasize the stepwise motion)
            interval = note - current_note

            if phrase_position < 0.2:
                if abs(interval) in [7, 12]:  # Large leap
                        probability *= 1.3

            if interval in [5, 7, 12, -5, -7, -12]:
                probability *= 1.3

            if previous_notes and phrase_position < 0.2 and len(previous_notes) >= 1:
                prev_note = previous_notes[-1]
                if abs(note - prev_note) <= 2:  # Stepwise motion
                    probability *= 0.8
                elif note in chord_notes and prev_note in chord_notes:
                    probability *= 1.5

            adjusted_notes.append(note)
            adjusted_probabilities.append(probability)

        # Normalize probabilities
        total_prob = sum(adjusted_probabilities)
        if total_prob > 0:
            adjusted_probabilities = [p / total_prob for p in adjusted_probabilities]
        else:
            adjusted_probabilities = [1.0 / len(adjusted_notes)] * len(adjusted_notes)

        return adjusted_notes, adjusted_probabilities

    # Get the current chord notes
    chord_notes = get_chord_notesV2(current_chord)

    # If the current note is not in the transition probabilities, default to C4
    # if current_note not in default_transitions:
    #     current_note = 60  # C4

    # Get possible next notes and their probabilities
    if len(previous_notes) >= markov_order:
        last_pattern = tuple(previous_notes[-markov_order:])
        if last_pattern in markov_transitions:
            candidate_notes = list(markov_transitions[last_pattern].keys())
            candidate_probabilities = list(markov_transitions[last_pattern].values())
        else:
            # Fallback to lower-order Markov chain or single-note transitions
            if len(previous_notes) >= markov_order - 1:
                last_pattern = tuple(previous_notes[-(markov_order - 1):])
                if last_pattern in markov_transitions:
                    candidate_notes = list(markov_transitions[last_pattern].keys())
                    candidate_probabilities = list(markov_transitions[last_pattern].values())
                else:
                    # Fallback to default single-note transitions
                    candidate_notes = list(default_transitions.get(current_note, {}).keys())
                    candidate_probabilities = list(default_transitions.get(current_note, {}).values())
            else:
                # Fallback to default single-note transitions
                candidate_notes = list(default_transitions.get(current_note, {}).keys())
                candidate_probabilities = list(default_transitions.get(current_note, {}).values())
    else:
        # Fallback to default single-note transitions
        candidate_notes = list(default_transitions.get(current_note, {}).keys())
        candidate_probabilities = list(default_transitions.get(current_note, {}).values())

    # Apply rule-based adjustments
    candidate_notes, candidate_probabilities = apply_rules(candidate_notes, current_note, previous_notes, chord_notes)

    # Choose the next note based on adjusted probabilities
    if candidate_notes:
        next_note = random.choices(candidate_notes, candidate_probabilities)[0]
    else:
        next_note = 60  # Default to C4 if no candidates are available

    return next_note

# Helper function to get chord notes
def get_chord_notes(chord):
    """Get the notes in a chord."""
    if chord == 'C':
        return [48,55,60]  # C4, E4, G4
    elif chord == 'D':
        return [50,57,62]  # D4, F4, A4
    elif chord == 'A':
        return [57,61,64]  # A4, C5, E5
    elif chord == 'Bm':
        return [47,54,59]  # B4, D5, F5
    elif chord == 'F#m':
        return [54,57,61]
    elif chord == 'F#':
        return [54,58,61]
    elif chord == 'G':
        return [43,50,55]  # G4, B4, D5
    elif chord == 'Am':
        return [57,60,64]  # A4, C5, E5
    elif chord == 'F':
        return [53,57,60]  # F4, A4, C5
    elif chord == 'Em':
        return [52,55,59]
    elif chord == 'Dm':
        return [50,53,57]
    elif chord == 'E':
        return [52,56,59]
    elif chord == 'Eb':
        return [51,55,58]
    
    elif chord == 'Cdim':
        return [48,51,54]
    elif chord == 'Ddim':
        return [50,53,56]
    elif chord == 'Edim':
        return [52,55,58]
    elif chord == 'Fdim':
        return [41,44,47]
    elif chord == 'Gdim':
        return [43,46,49]
    elif chord == 'Adim':
        return [45,48,51]
    elif chord == 'Bdim':
        return [47,50,53]
    elif chord == 'Cm':
        return [48,51,55]
    elif chord == 'Fm':
        return [41,44,48]
    elif chord == 'Bm':
        return [47,50,54]
    elif chord == 'Ebm':
        return [39,42,46]
    elif chord == 'Bbm':
        return [46,49,53]
    elif chord == 'Gm':
        return [43,46,50]
    elif chord == 'C#m':
        return [49,52,56]
    elif chord == 'Abm':
        return [44,47,51]
    return [60, 64, 67]  # Default to C major

def parse_chord(chord):
    """
    Parse the chord string into root note and chord type.
    
    Args:
        chord (str): Chord name (e.g., 'C', 'Cm', 'C#dim', 'Bbaug').
    
    Returns:
        tuple: (root, chord_type) where root is the root note and chord_type is
               'major', 'minor', 'dim', or 'aug'.
    """
    if len(chord) > 1 and chord[1] in ['#', 'b']:
        root = chord[:2]  # e.g., 'C#', 'Bb'
        type_str = chord[2:]
    else:
        root = chord[0]  # e.g., 'C'
        type_str = chord[1:]
    
    if type_str == '':
        chord_type = 'major'
    elif type_str == 'm':
        chord_type = 'minor'
    elif type_str == 'dim':
        chord_type = 'dim'
    elif type_str == 'aug':
        chord_type = 'aug'
    else:
        chord_type = 'major'  # default to major
    
    return root, chord_type

def get_chord_notesV2(chord):
    """
    Get the MIDI note numbers for the given chord.
    
    Args:
        chord (str): Chord name (e.g., 'C', 'Cm', 'C#dim', 'Bbaug').
    
    Returns:
        list: List of MIDI note numbers representing the chord.
    """
    # Dictionary mapping root notes to semitone offsets from 'C'
    note_to_offset = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4,
        'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9,
        'A#': 10, 'Bb': 10, 'B': 11
    }
    
    try:
        # Parse the chord to get root and type
        root, chord_type = parse_chord(chord)
        
        # Check if root note is valid
        if root not in note_to_offset:
            return [60, 64, 67]  # default to C major (C4, E4, G4)
        
        # Calculate root MIDI note in octave 3 (C3 = 48)
        offset = note_to_offset[root]
        root_midi = 48 + offset
        
        # Define intervals based on chord type
        if chord_type == 'major':
            intervals = [0, 4, 7]  # root, major third, perfect fifth
        elif chord_type == 'minor':
            intervals = [0, 3, 7]  # root, minor third, perfect fifth
        elif chord_type == 'dim':
            intervals = [0, 3, 6]  # root, minor third, diminished fifth
        elif chord_type == 'aug':
            intervals = [0, 4, 8]  # root, major third, augmented fifth
        else:
            intervals = [0, 4, 7]  # default to major
        
        # Calculate MIDI notes
        notes = [root_midi + interval for interval in intervals]
        return notes
    except:
        return [60, 64, 67]  # default to C major in case of error
# Mapping of root notes to semitone indices (handles both sharps and flats)
note_to_index = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5,
    'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
}

def transpose_chord(chord, semitones):
    """
    Transpose a single chord by a number of semitones.
    
    Args:
        chord (str): The chord to transpose (e.g., 'Cm', 'Bb', 'C#dim').
        semitones (int): Number of semitones to transpose.
    
    Returns:
        str: The transposed chord.
    """
    chromatic_scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Determine if the root is 2 characters (e.g., 'C#', 'Db') or 1 character
    if len(chord) >= 2 and chord[:2] in note_to_index:
        root = chord[:2]
        modifier = chord[2:]
    else:
        root = chord[0]
        modifier = chord[1:]
    
    if root not in note_to_index:
        raise ValueError(f"Invalid root note: {root}")
    
    root_index = note_to_index[root]
    new_index = (root_index + semitones) % 12
    new_root = chromatic_scale[new_index]
    return new_root + modifier

def apply_key_shift(chord_progression, key_shift):
    """
    Apply a key shift (transposition) to an entire chord progression.
    
    Args:
        chord_progression (list): List of chord names.
        key_shift (int): Number of semitones to transpose.
    
    Returns:
        list: Transposed chord progression.
    """
    return [transpose_chord(chord, key_shift) for chord in chord_progression]

def get_next_pattern(current_pattern):
    """
    Given the current pattern, choose and return the next pattern
    based on predefined transition probabilities.
    
    Args:
        current_pattern (str): The current chord pattern key.
        
    Returns:
        str: The next chord pattern key.
        
    Raises:
        ValueError: If the current pattern is not recognized.
    """
    pattern_transitions = {
        'pattern_1': {'pattern_1': 0.6, 'pattern_2': 0.2, 'pattern_5':0.2},  # After pattern_1, go to pattern_2 or pattern_3
        'pattern_2': {'pattern_2': 0.5, 'pattern_7': 0.2, 'pattern_5': 0.3},  # After pattern_2, go to pattern_1 or pattern_4
        'pattern_sad1': {'pattern_sad1': 0.7, 'pattern_2': 0.2,'pattern_sad3':0.1},  # After pattern_3, go to pattern_2 or pattern_4
        'pattern_4': {'pattern_4': 0.7, 'pattern_1': 0.2, 'pattern_5':0.1},
        'pattern_5': {'pattern_5': 0.6, 'pattern_2': 0.1, 'pattern_7': 0.1, 'pattern_sad3':0.2},
        'pattern_7': {'pattern_2':0.1, 'pattern_5': 0.2, 'pattern_7':0.5,
                      'pattern_8': 0.1, 'pattern_sad3': 0.1},
        'pattern_8': {'pattern_5':0.4, 'pattern_9': 0.2, 'pattern_7':0.4},
        'pattern_9': {'pattern_9': 0.5, 'pattern_1': 0.2, 'pattern_5': 0.3},
        'pattern_sad2': {'pattern_sad2': 0.7, 'pattern_5':0.3},
        'pattern_sad3': {'pattern_5': 0.8, 'pattern_7': 0.2},
        'pattern_fear1':{'pattern_fear1': 0.5, 'pattern_fear6':0.5},
        'pattern_fear2':{'pattern_fear2':0.3, 'pattern_fear3':0.3, 'pattern_fear7':0.4},
        'pattern_fear3':{'pattern_fear4':0.3, 'pattern_fear5':0.3,'pattern_fear2':0.2,'pattern_fear3':0.2},
        'pattern_fear4':{'pattern_fear4':0.5,'pattern_fear5':0.2,'pattern_fear3':0.3},
        'pattern_fear5':{'pattern_fear5':0.3, 'pattern_fear4':0.2,'pattern_fear7':0.2,'pattern_fear6':0.3},
        'pattern_fear6':{'pattern_fear6':0.4,'pattern_fear1':0.4,'pattern_fear7':0.2},
        'pattern_fear7':{'pattern_fear1':0.2, 'pattern_fear2':0.3, 'pattern_fear6':0.2,'pattern_fear7':0.2,'pattern_fear4':0.1}
        # Add more transitions as needed
    }
    
    if current_pattern not in pattern_transitions:
        raise ValueError(f"Invalid pattern: {current_pattern}")
    
    next_pattern_options = list(pattern_transitions[current_pattern].keys())
    next_pattern_probabilities = list(pattern_transitions[current_pattern].values())
    next_pattern = random.choices(next_pattern_options, weights=next_pattern_probabilities)[0]
    return next_pattern

# Example usage combining both functionalities:
def generate_chord_progression_with_keyshift(start_pattern, key_shift=0):
    """
    Generate a chord progression that chains multiple patterns,
    applies a key shift to the entire progression, and returns the
    final progression along with the last used pattern.
    
    Args:
        start_pattern (str): The starting chord pattern key.
        key_shift (int): Semitones to transpose the chords.
        num_patterns (int): How many patterns to chain.
    
    Returns:
        tuple: (transposed_progression, final_pattern)
    """
    chord_patterns = {
        'pattern_1': ['C', 'G', 'Am', 'Em', 'F', 'G', 'C'],
        'pattern_2': ['F', 'G', 'C'],
        'pattern_sad1': ['Am', 'F', 'C', 'G'],
        'pattern_4': ['C', 'Am', 'F', 'G'],
        'pattern_5': ['F', 'G', 'Em', 'Am'],
        'pattern_sad2': ['Am', 'G', 'F', 'E'],
        'pattern_7': ['F', 'G', 'Am'],
        'pattern_sad3': ['Dm', 'E'],
        'pattern_fear1': ['Cdim', 'Ddim', 'Edim', 'Fdim'],
        'pattern_fear2': ['Cm', 'Gdim', 'Abm', 'Fm'],
        'pattern_fear3': ['Fm', 'Bm', 'Ebm', 'Am'],
        'pattern_fear4': ['Em', 'F', 'G', 'Am'],
        'pattern_fear5': ['Em', 'C#m', 'Gm', 'Bm'],
        'pattern_fear6': ['Fdim', 'Gdim', 'Adim', 'Bdim', 'Cdim'],
        'pattern_fear7': ['Bdim', 'Cm', 'Dm', 'Ebm'],
        'pattern_8': ['Dm', 'E', 'F', 'G'],
        'pattern_9': ['C', 'Bm', 'Am', 'G', 'F', 'Em', 'F', 'G'],
    }
    
    progression = []
    current_pattern = start_pattern

    if current_pattern not in chord_patterns:
        raise ValueError(f"Invalid pattern: {current_pattern}")
    # Extend progression with chords from the current pattern
    progression.extend(chord_patterns[current_pattern])
   
    
    # Apply the key shift (transposition) to the entire progression
    transposed_progression = apply_key_shift(progression, key_shift)
    return transposed_progression


def generate_waveform(t, freq, wave_type='sine'):
    if wave_type == 'sine':
        return np.sin(2 * np.pi * freq * t)
    elif wave_type == 'sawtooth':
        return 2 * (t * freq - np.floor(t * freq + 0.5))
    elif wave_type == 'square':
        return np.sign(np.sin(2 * np.pi * freq * t))
    else:
        return np.sin(2 * np.pi * freq * t)
    
def frequency_from_midi(midi_note):
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * 2 ** ((midi_note - 69) / 12.0)


def generate_melody_with_chordsV2(rhythm, piano_track, chord_track, drum_track, min_v, max_v, emotion,total_duration, tempo=120):
    #ticks_per_beat = 480  # Resolution: 480 ticks per quarter note
    #ticks_per_whole_note = 4 * ticks_per_beat  # 4 beats per whole not
    #current_time_ticks = 0  # Current time in ticks

    # Generate a chord progression
    chord_pattern = 'pattern_1'
    if emotion == 'happy':
        current_note = random.choice([60,64,67])
        chord_pattern = random.choice(['pattern_1','pattern_4'])
    elif emotion == 'sad':
        current_note = random.choice([57,60,64,69])
        chord_pattern = random.choice(['pattern_sad1','pattern_sad2'])
    elif emotion == 'angry':
        current_note = random.choice([60,67,72])
        chord_pattern = random.choice(['pattern_1','pattern_4'])
    elif emotion == 'fear':
        current_note = random.choice([60,64,69])
        chord_pattern = random.choice(['pattern_fear1','pattern_fear2'])
    elif emotion == 'awe':
        current_note = random.choice([60,64,67,69,72])
        chord_pattern = random.choice(['pattern_7','pattern_5'])

    key_shift = random.randint(-4, 7)
    #key_shift = 0
    # Generate a chord progression
    #chord_progression = generate_chord_progression(start_pattern= chord_pattern)
    chord_progression = generate_chord_progression_with_keyshift(start_pattern=chord_pattern, key_shift=key_shift)
    chord_index = 0
    current_chord = chord_progression[chord_index]
    previous_notes = []  # Track previous notes for pattern recognition
    current_time_seconds = 0  # Current time in seconds
    bar_duration_seconds = 4 * (60 / tempo)  # Duration of a bar (4 beats)
    beat_duration_seconds = 60 / tempo
    current_beat = 0

    drum_patterns = {
        "happy": [
            (36, [0, 2]),  # Kick on beats 1 and 3
            (38, [1, 3]),  # Snare on beats 2 and 4
            (42, [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])  # Hi-hat on all beats
        ],
        "sad": [
            (36, [0]),  # Kick on beat 1
            (38, [2]),  # Snare on beat 3
            (42, [0, 2]),  # Soft hi-hat on beats 1 and 3
            (46, [1, 3])  # Open hi-hat on beats 2 and 4
        ],
        "angry": [
            (36, [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]),  # Double bass drum
            (38, [1, 3]),  # Snare on beats 2 and 4
            (42, [0, 1, 2, 3]),  # Hi-hat on all beats
            (49, [0])  # Crash on beat 1
        ],
        "fear": [
            (36, [0, 1.5, 2.75]),  # Irregular kick
            (38, [2, 3.5]),  # Snare on beat 3 and before 4
            (42, [0, 1, 2, 3]),  # Constant hi-hat
            (56, [2.25, 3.75])  # Cowbell for suspense
        ],
        "awe": [
            (36, [0, 2]),  # Kick on beats 1 and 3
            (38, [2.5]),  # Snare just after beat 3
            (42, [1, 3]),  # Hi-hat on beats 2 and 4
            (81, [1.5, 3.5])  # Triangle for cinematic feel
        ]
    }

    drum_pattern = drum_patterns.get(emotion, drum_patterns["happy"])

    total_beats_in_pattern = 4  # The length of one cycle
    for beat, is_rest in rhythm:
        duration = beat * beat_duration_seconds
        phrase_position = current_time_seconds / total_duration
        if phrase_position > 1.0:
            phrase_position = 1.0
        #print(current_beat)
        #if not is_rest:
            # Generate the next melody note based on emotion
        # 🎵 **Melody**
        note_function = {
            "happy": get_next_note_happyV2,
            "sad": get_next_note_sadV2,
            "angry": get_next_note_angryV2,
            "fear": get_next_note_fearV2,
            "awe": get_next_note_aweV2
        }.get(emotion, get_next_note_happyV2)
        

        # Melody Pattern
        current_note = note_function(current_note, current_chord, previous_notes, markov_order=3, phrase_position=phrase_position,key_shift=key_shift)
        shift_note = current_note + key_shift
        if not is_rest:
            # Add melody note
           
            velocity = random.randint(min_v, max_v)
            melody_note = pretty_midi.Note(
                velocity=velocity,
                pitch=shift_note,
                start= current_time_seconds,  # Convert ticks to seconds
                end=current_time_seconds + duration  # Convert ticks to seconds
            )
            piano_track.notes.append(melody_note)
        # print(current_time_seconds)
        # print(bar_duration_seconds)
        # print(is_multiple_float(current_time_seconds,bar_duration_seconds))
        # Check if it's the first beat of a bar to play the chord

        # Drum Pattern
        for drum_note, beats in drum_pattern:
            for drum_beat in beats:
                adjusted_beat = drum_beat % total_beats_in_pattern  # Make beats repeat
                if math.isclose(current_beat % total_beats_in_pattern, adjusted_beat, abs_tol=0.05):
                    drum_note_midi = pretty_midi.Note(
                        velocity=80,
                        pitch=drum_note,
                        start=current_time_seconds,
                        end=current_time_seconds + beat_duration_seconds
                    )
                    drum_track.notes.append(drum_note_midi)

        # Chord Pattern: Change chords at the start of each bar
        if math.isclose(current_beat % 2, 0, abs_tol=0.05):
            # Add chord notes for the current chord
            chord_notes = get_chord_notesV2(current_chord)
            for pitch in chord_notes:
                chord_note = pretty_midi.Note(
                    velocity=80,
                    pitch=pitch,
                    start=current_time_seconds,
                    end=current_time_seconds + bar_duration_seconds
                )
                chord_track.notes.append(chord_note)

            # Move to the next chord in the progression
            chord_index += 1

            # If we've reached the end of the current progression, update the chord pattern and regenerate the progression
            if chord_index >= len(chord_progression):
                chord_pattern = get_next_pattern(chord_pattern)
                print('Change Chord Pattern !!!!!!')
                chord_progression = generate_chord_progression_with_keyshift(start_pattern=chord_pattern, key_shift=key_shift)
                chord_index = 0

            current_chord = chord_progression[chord_index]

            # Update previous notes
            previous_notes.append(current_note)
            if len(previous_notes) > 4:  # Keep only the last 4 notes
                previous_notes.pop(0)

        current_beat += beat

        # Update current time and beat
        current_time_seconds+=duration

def generate_melody_with_chordsV3(rhythm, min_v, max_v, emotion,total_duration, tempo=120,sample_rate=44100,instrument = 'piano'):
    #ticks_per_beat = 480  # Resolution: 480 ticks per quarter note
    #ticks_per_whole_note = 4 * ticks_per_beat  # 4 beats per whole not
    #current_time_ticks = 0  # Current time in ticks

    # Generate a chord progression
    chord_pattern = 'pattern_1'
    if emotion == 'happy':
        current_note = random.choice([60,64,67])
        chord_pattern = random.choice(['pattern_1','pattern_4'])
    elif emotion == 'sad':
        current_note = random.choice([57,60,64,69])
        chord_pattern = random.choice(['pattern_sad1','pattern_sad2'])
    elif emotion == 'angry':
        current_note = random.choice([60,67,72])
        chord_pattern = random.choice(['pattern_1','pattern_4'])
    elif emotion == 'fear':
        current_note = random.choice([60,64,69])
        chord_pattern = random.choice(['pattern_fear1','pattern_fear2'])
    elif emotion == 'awe':
        current_note = random.choice([60,64,67,69,72])
        chord_pattern = random.choice(['pattern_7','pattern_5'])

    key_shift = random.randint(-4, 7)
    #key_shift = 0
    # Generate a chord progression
    #chord_progression = generate_chord_progression(start_pattern= chord_pattern)
    chord_progression = generate_chord_progression_with_keyshift(start_pattern=chord_pattern, key_shift=key_shift)
    chord_index = 0
    current_chord = chord_progression[chord_index]
    previous_notes = []  # Track previous notes for pattern recognition
    current_time_seconds = 0  # Current time in seconds
    bar_duration_seconds = 4 * (60 / tempo)  # Duration of a bar (4 beats)
    beat_duration_seconds = 60 / tempo
    current_beat = 0

    # Initialize audio array
    total_samples = int(total_duration * sample_rate)
    audio = np.zeros(total_samples, dtype=np.float32)

    # Instrument profiles (simulating MIDI programs)
    instrument_profiles = {
    'piano': {'wave': 'sine', 'harmonics': [1.0, 0.5, 0.2], 'decay': 5.0, 'amp': 1.0},  # Bright, quick decay
    'violin': {'wave': 'sine', 'harmonics': [1.0, 0.3], 'decay': 0.5, 'amp': 0.8, 'vibrato': 0.1},  # Sustained, vibrato
    'flute': {'wave': 'sine', 'harmonics': [1.0], 'decay': 0.2, 'amp': 0.7, 'vibrato': 0.02},  # Smooth, pure tone
    'guitar': {'wave': 'sawtooth', 'harmonics': [1.0, 0.7, 0.4], 'decay': 4.0, 'amp': 0.9}  # Sharp, plucked
}
    profile = instrument_profiles.get(instrument.lower(), instrument_profiles['piano'])

    drum_patterns = {
        "happy": [
            (36, [0, 2]),  # Kick on beats 1 and 3
            (38, [1, 3]),  # Snare on beats 2 and 4
            (42, [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])  # Hi-hat on all beats
        ],
        "sad": [
            (36, [0]),  # Kick on beat 1
            (38, [2]),  # Snare on beat 3
            (42, [0, 2]),  # Soft hi-hat on beats 1 and 3
            (46, [1, 3])  # Open hi-hat on beats 2 and 4
        ],
        "angry": [
            (36, [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]),  # Double bass drum
            (38, [1, 3]),  # Snare on beats 2 and 4
            (42, [0, 1, 2, 3]),  # Hi-hat on all beats
            (49, [0])  # Crash on beat 1
        ],
        "fear": [
            (36, [0, 1.5, 2.75]),  # Irregular kick
            (38, [2, 3.5]),  # Snare on beat 3 and before 4
            (42, [0, 1, 2, 3]),  # Constant hi-hat
            (56, [2.25, 3.75])  # Cowbell for suspense
        ],
        "awe": [
            (36, [0, 2]),  # Kick on beats 1 and 3
            (38, [2.5]),  # Snare just after beat 3
            (42, [1, 3]),  # Hi-hat on beats 2 and 4
            (81, [1.5, 3.5])  # Triangle for cinematic feel
        ]
    }

    drum_pattern = drum_patterns.get(emotion, drum_patterns["happy"])

    total_beats_in_pattern = 4  # The length of one cycle
    for beat, is_rest in rhythm:
        duration = beat * beat_duration_seconds
        phrase_position = current_time_seconds / total_duration
        if phrase_position > 1.0:
            phrase_position = 1.0
        #print(current_beat)
        #if not is_rest:
            # Generate the next melody note based on emotion
        # 🎵 **Melody**
        note_function = {
            "happy": get_next_note_happyV2,
            "sad": get_next_note_sadV2,
            "angry": get_next_note_angryV2,
            "fear": get_next_note_fearV2,
            "awe": get_next_note_aweV2
        }.get(emotion, get_next_note_happyV2)
        

        # Melody Pattern
        current_note = note_function(current_note, current_chord, previous_notes, markov_order=3, phrase_position=phrase_position,key_shift=key_shift)
        shift_note = current_note + key_shift
        if not is_rest:
            # Generate melody waveform with instrument profile
            freq = frequency_from_midi(shift_note)
            velocity = random.randint(min_v, max_v) / 127.0
            start_sample = int(current_time_seconds * sample_rate)
            end_sample = start_sample + int(duration * sample_rate)
            t = np.linspace(0, duration, end_sample - start_sample, False)
            waveform = 0.0
            for i, harmonic in enumerate(profile['harmonics']):
                waveform += harmonic * generate_waveform(t, freq * (i + 1), profile.get('wave', 'sine'))
            if 'vibrato' in profile:
                waveform *= (1 + profile['vibrato'] * np.sin(2 * np.pi * 5 * t))
            waveform *= velocity * profile['amp'] * np.exp(-t * profile['decay'])

        
        # Drum pattern
        for drum_note, beats in drum_pattern:
            for drum_beat in beats:
                adjusted_beat = drum_beat % total_beats_in_pattern
                if math.isclose(current_beat % total_beats_in_pattern, adjusted_beat, abs_tol=0.05):
                    # Generate drum sound (simple percussive burst)
                    drum_freq = 50 if drum_note == 36 else 100 if drum_note == 38 else 200  # Basic freqs for kick, snare, hi-hat
                    drum_duration = 0.1  # Short duration for drums
                    start_sample = int(current_time_seconds * sample_rate)
                    end_sample = start_sample + int(drum_duration * sample_rate)
                    t = np.linspace(0, drum_duration, end_sample - start_sample, False)
                    drum_waveform = 0.5 * np.sin(2 * np.pi * drum_freq * t) * np.exp(-t * 20)  # Fast decay
                    audio[start_sample:end_sample] += drum_waveform

        # Chord Pattern: Change chords at the start of each bar
        if math.isclose(current_beat % 2, 0, abs_tol=0.05):
            # Add chord notes for the current chord
            chord_notes = get_chord_notesV2(current_chord)
            for pitch in chord_notes:
                # Generate chord waveform
                freq = frequency_from_midi(pitch)
                start_sample = int(current_time_seconds * sample_rate)
                end_sample = start_sample + int(bar_duration_seconds * sample_rate)
                t = np.linspace(0, bar_duration_seconds, end_sample - start_sample, False)
                waveform = 0.3 * np.sin(2 * np.pi * freq * t) * np.exp(-t * 1)  # Softer, longer decay
                audio[start_sample:end_sample] += waveform

            # Move to the next chord in the progression
            chord_index += 1

            # If we've reached the end of the current progression, update the chord pattern and regenerate the progression
            if chord_index >= len(chord_progression):
                chord_pattern = get_next_pattern(chord_pattern)
                print('Change Chord Pattern !!!!!!')
                chord_progression = generate_chord_progression_with_keyshift(start_pattern=chord_pattern, key_shift=key_shift)
                chord_index = 0

            current_chord = chord_progression[chord_index]

            # Update previous notes
            previous_notes.append(current_note)
            if len(previous_notes) > 4:  # Keep only the last 4 notes
                previous_notes.pop(0)

        current_beat += beat
        # Update current time and beat
        current_time_seconds+=duration
    # Normalize audio to prevent clipping
    audio = audio / np.max(np.abs(audio), initial=1.0)

    return audio