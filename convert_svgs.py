from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF

pairs = [
    ('visiting_card_front.svg', 'visiting_card_front.pdf'),
    ('visiting_card_back.svg', 'visiting_card_back.pdf'),
]

for src, dst in pairs:
    print('Converting', src, '→', dst)
    drawing = svg2rlg(src)
    renderPDF.drawToFile(drawing, dst)
print('Done')
