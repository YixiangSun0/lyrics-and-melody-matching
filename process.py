import numpy as np
import pretty_midi as pyd
import os
import torch


def convert_to_midi(path_to_note, path_to_midi, velocity=120):
    with open(path_to_note, 'r') as note_file:
        note_list = []
        for line in note_file:

            one_note = [float(ele) for ele in line.strip('\n').split('\t')]
            notes = pyd.Note(start=one_note[0], end=one_note[1], pitch=int(
                one_note[2]), velocity=velocity)
            note_list.append(notes)
        midi_c_chord = pyd.PrettyMIDI()
        midi_program = pyd.instrument_name_to_program('Acoustic_Grand_Piano')
        midi_note = pyd.Instrument(program=midi_program)
        for note in note_list:
            midi_note.notes.append(note)
        midi_c_chord.instruments.append(midi_note)
        midi_c_chord.write(path_to_midi)


def pair_lyric_and_note(id_num):
    PREFIX = "/Users/yixiangsun/Desktop/cb_lyrics_midi"
    path_to_midi = os.path.join(PREFIX, id_num, "midi.lab")
    path_to_lyric = os.path.join(PREFIX, id_num, "lyric.lab")
    lyrics_list = []
    notes_list = []
    with open(path_to_lyric, 'r', encoding="UTF-16") as lyric_file:
        for line in lyric_file:
            lyrics_list.append(line.strip('\n').split('\t'))
    with open(path_to_midi, 'r') as midi_file:
        for line in midi_file:
            notes_list.append(line.strip('\n').split('\t'))

    torch.save((notes_list, lyrics_list), os.path.join("torch_file", id_num + ".pth"))


if __name__ == "__main__":
    if not os.path.exists("torch_file"):
        os.mkdir("torch_file")
    if not os.path.exists("midi"):
        os.mkdir("midi")
    cnt = 0
    for root, dirs, files in os.walk("./cb_lyrics_midi/", topdown=False):
        for dir_name in dirs:
            print(dir_name)
            pair_lyric_and_note(dir_name)
            convert_to_midi("./cb_lyrics_midi/" + dir_name + "/midi.lab", "midi/" + dir_name + ".midi")

       
       

