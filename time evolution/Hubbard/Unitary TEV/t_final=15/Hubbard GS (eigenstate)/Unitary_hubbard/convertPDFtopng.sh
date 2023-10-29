mkdir -p images
for i in *.pdf; do
    # Extracts the number from the filename
    number=$(echo $i | sed 's/\.pdf//')

    # Uses ImageMagick to convert the PDF to an image
    convert -density 150 "${i}" "images/frame${number}.png"
done
