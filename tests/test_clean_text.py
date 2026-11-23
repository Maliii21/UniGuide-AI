import unittest

from src.clean_text import clean_text


class CleanTextTests(unittest.TestCase):
    def test_clean_preserves_structure_and_unicode(self):
        text = "  Heading\r\n\r\n\r\n  •\t75%  \r\n 1-  March 2022\r\n  Cre\u0300me — student@aiu.edu.eg "
        self.assertEqual(clean_text(clean_text(text)), clean_text(text))
        self.assertEqual(clean_text(" \t\n "), "")
        self.assertEqual(clean_text(text), "Heading\n\n• 75%\n1- March 2022\nCrème — student@aiu.edu.eg")
