import matplotlib.pyplot as plt
from pdf2image import convert_from_path


# Manually specify the PDF filenames
pdf_files = [
    '0.pdf',
    '4560.pdf',
    '8560.pdf',
    '16560.pdf',
    '20560.pdf',
    '24560.pdf',
    '28560.pdf',
    '36560.pdf',
    '40560.pdf',
]

# Number of rows and columns in the grid
rows = 3
cols = 3

# Create a new figure
fig, axs = plt.subplots(rows, cols, figsize=(15, 11))

for i in range(rows):
    for j in range(cols):
        # Get the index of the current PDF file
        idx = i * cols + j
        
        # Check if the index is out of bounds
        if idx >= len(pdf_files):
            break
        
        # Convert the PDF to an image
        images = convert_from_path(pdf_files[idx])
        
        # Display the image in the subplot
        axs[i, j].imshow(images[0], cmap='gray', aspect='auto')
        if i == rows - 1:
            axs[i, j].set_xlabel('x')
        if j == 0:
            axs[i, j].set_ylabel('Probability density')

        # Clear outer axis stuff:
        axs[i, j].spines['top'].set_color('none')
        axs[i, j].spines['bottom'].set_color('none')
        axs[i, j].spines['left'].set_color('none')
        axs[i, j].spines['right'].set_color('none')

        # Remove ticks
        axs[i, j].xaxis.set_ticks([])
        axs[i, j].yaxis.set_ticks([])

        # Remove tick labels (this may be redundant as removing ticks often removes their labels as well)
        axs[i, j].xaxis.set_ticklabels([])
        axs[i, j].yaxis.set_ticklabels([])

# Adjust the layout
plt.tight_layout()

# Save the figure as a PDF
plt.savefig("combined.pdf", dpi=300)

# Show the figure
plt.show()
