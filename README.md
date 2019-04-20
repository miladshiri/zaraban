# Zaraban
Zaraban is a biomedical tools that provide high-level APIs for analyzing echocardiograms. 
This python package consists of various methods including:
- Seckle detection using machine learning
- Speckle Tracking Echocardiography (STE)
- Movement visualization and saving
- ...

## Installation

After downloading the package, you should install all packages in requirements.txt by the command below:

    pip install -r requirements.txt


## How to use the package!

Here are some examples for reading, tracking and saving frames with Zaraban.

### Read frames:

    frames = tools.read_frames(path, size=(200, 200), pattern="im ({}).bmp")
    
path: path of the parent folder of frames
size (optional): desire output frame size. Default=(200, 200)
pattern (optional): filename pattern. If it is not determined, files will be read alphabetically. 
    

      
