from pydub import AudioSegment
import os


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


if __name__ == "__main__":
    split_mp3("test_audio/chop.mp3")
