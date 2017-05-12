dot -Tpdf model_1.dot -o model_1.pdf
dot -Tpdf model_2.dot -o model_2.pdf
dot -Tpdf model_3.dot -o model_3.pdf

dot -Tpdf qrnn.dot -o qrnn.pdf
dot -Tpdf sentrnn.dot -o sentrnn.pdf

pdfcrop model_1.pdf
pdfcrop model_2.pdf
pdfcrop model_3.pdf
pdfcrop qrnn.pdf
pdfcrop sentrnn.pdf
