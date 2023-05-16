from pydub import AudioSegment
import os
import mido


def split_mp3(audio_path: str):
    if not os.path.isdir("tmp"):
        os.mkdir("tmp")
    # загрузка аудиофайла
    audio_file = AudioSegment.from_file(audio_path, format="mp3")
    # задание длительности частей в миллисекундах
    part_length = 2000
    # разделение аудиофайла на части
    parts = [audio_file[i:i + part_length] for i in range(0, len(audio_file), part_length)]
    # сохранение каждой части в отдельный файл
    for i, part in enumerate(parts):
        part.export(f"tmp/part_{i}.mp3", format="mp3")


def split_midi(audio_path: str):
    if not os.path.isdir("tmp"):
        os.mkdir("tmp")

    # загрузка MIDI файла
    midi_file = mido.MidiFile(audio_path)

    # определение длительности сегментов в миллисекундах
    segment_length = 2000

    # создание списка сегментов
    segments = []
    current_segment = []
    current_tick = 0

    # обход всех сообщений в треках
    for track in midi_file.tracks:
        for msg in track:
            # проверка, достигнут ли конец текущего сегмента
            if msg.time + current_tick >= segment_length:
                # добавление текущего сегмента в список и создание нового
                segments.append(current_segment)
                current_segment = [msg]
                current_tick = msg.time
            else:
                # добавление сообщения в текущий сегмент и увеличение текущего времени
                current_segment.append(msg)
                current_tick += msg.time

    # добавление последнего сегмента в список
    segments.append(current_segment)

    # сохранение каждого сегмента в отдельный файл
    for i, segment in enumerate(segments):
        # создание нового MIDI файла и добавление трека с текущим сегментом
        new_midi = mido.MidiFile()
        new_track = mido.MidiTrack()
        new_midi.tracks.append(new_track)
        for msg in segment:
            new_track.append(msg)
        # сохранение файла
        new_midi.save(f'tmp/segment_{i}.mid')


if __name__ == "__main__":
    split_midi("test_audio/chop.mid")
