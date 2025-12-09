# this python code fixes the arbitary
# Image 1, Image 2 naming conventions

data = [
    "4980-Balcony.jpg",
    "4980-City.jpg",
    "4980-City5.jpg",
    "4980-Football.jpg",
    "4980-IATL.jpg",
    "4980-Linquistic.jpg",
    "4980-Park.jpg",
    "4980-Seamans2.jpg",
    "4980-eng.jpg",
    "4980-law.jpg",
    "4980-Biz.jpg",
    "4980-City2.jpg",
    "4980-City6.jpg",
    "4980-Forest.jpg",
    "4980-IMU.jpg",
    "4980-MF-Outer.jpg",
    "4980-River.jpg",
    "4980-Staples.jpg",
    "4980-epp.jpg",
    "4980-library.jpg",
    "4980-Bridge.jpg",
    "4980-City4.jpg",
    "4980-Downtown3.jpg",
    "4980-GradHotel.jpg",
    "4980-IMU2.jpg",
    "4980-Mall.jpg",
    "4980-Seamans.jpg",
    "4980-Westside.jpg",
    "4980-iowabook.jpg",
    "4980-med.jpg"
]

# Read the input TSV file
input_file = "U5.tsv"
output_file = "U5.tsv"

with open(input_file, 'r') as f:
    content = f.read()

# Replace Image_XX with corresponding image names
for i in range(len(data)):
    image_num = f"{i+1:02d}"  # Format as 01, 02, etc.
    placeholder = f"Image_{image_num}"
    content = content.replace(placeholder, data[i])

# Write the output file
with open(output_file, 'w') as f:
    f.write(content)

print(f"Conversion complete! Output written to {output_file}")