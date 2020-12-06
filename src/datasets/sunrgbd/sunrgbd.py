# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""


class SUNRBDBase:
    SPLITS = ['train', 'test']

    # number of classes without void
    N_CLASSES = 37

    CLASS_NAMES_ENGLISH = ['void', 'wall', 'floor', 'cabinet', 'bed', 'chair',
                           'sofa', 'table', 'door', 'window', 'bookshelf',
                           'picture', 'counter', 'blinds', 'desk', 'shelves',
                           'curtain', 'dresser', 'pillow', 'mirror',
                           'floor mat', 'clothes', 'ceiling', 'books',
                           'fridge', 'tv', 'paper', 'towel', 'shower curtain',
                           'box', 'whiteboard', 'person', 'night stand',
                           'toilet', 'sink', 'lamp', 'bathtub', 'bag']

    CLASS_NAMES_GERMAN = ['Void', 'Wand', 'Boden', 'Schrank', 'Bett', 'Stuhl',
                          'Sofa', 'Tisch', 'Tür', 'Fenster', 'Bücherregal',
                          'Bild', 'Tresen', 'Jalousien', 'Schreibtisch',
                          'Regal', 'Vorhang', 'Kommode', 'Kissen', 'Spiegel',
                          'Bodenmatte', 'Kleidung', 'Zimmerdecke', 'Bücher',
                          'Kühlschrank', 'Fernseher', 'Papier', 'Handtuch',
                          'Duschvorhang', 'Kiste', 'Whiteboard', 'Person',
                          'Nachttisch', 'Toilette', 'Waschbecken', 'Lampe',
                          'Badewanne', 'Tasche']

    CLASS_COLORS = [(0, 0, 0), (119, 119, 119), (244, 243, 131),
                    (137, 28, 157), (150, 255, 255), (54, 114, 113),
                    (0, 0, 176), (255, 69, 0), (87, 112, 255), (0, 163, 33),
                    (255, 150, 255), (255, 180, 10), (101, 70, 86),
                    (38, 230, 0), (255, 120, 70), (117, 41, 121),
                    (150, 255, 0), (132, 0, 255), (24, 209, 255),
                    (191, 130, 35), (219, 200, 109), (154, 62, 86),
                    (255, 190, 190), (255, 0, 255), (192, 79, 212),
                    (152, 163, 55), (230, 230, 230), (53, 130, 64),
                    (155, 249, 152), (87, 64, 34), (214, 209, 175),
                    (170, 0, 59), (255, 0, 0), (193, 195, 234), (70, 72, 115),
                    (255, 255, 0), (52, 57, 131), (12, 83, 45)]
