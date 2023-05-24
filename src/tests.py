import unittest
import os
from split_audio import split_mp3
from clean_dirs import clean_dir, remove_dir
from estimation import calc_measure


class SplitAudio(unittest.TestCase):
    def test_splitting(self):
        audio = "test_audio/vo_sne.mp3"
        step = 5000
        split_mp3(audio, step)
        self.assertEqual(os.path.exists("tmp/"), True)
        self.assertGreater(len(os.listdir("tmp/")), 0)


class CleanDirs(unittest.TestCase):
    def test_cleaning(self):
        os.mkdir("temp/")
        fp = open('temp/file.txt', 'x')
        fp.close()
        dir = "temp/"
        clean_dir(dir)
        self.assertEqual(len(os.listdir(dir)), 0)
        os.rmdir(dir)

    def test_removing(self):
        os.mkdir("temp/")
        dir = "temp/"
        remove_dir(dir)
        self.assertEqual(os.path.exists("temp/"), False)


class CalcMeasure(unittest.TestCase):
    def test_calculating(self):
        downbeats = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        measure = calc_measure(downbeats)
        self.assertEqual(measure, 4)


if __name__ == '__main__':
    unittest.main()
