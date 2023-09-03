# table-tennis-ball-bounce-detection
Aim to the project is to detect bounce event in Table Tennis match using one camera angle. 
## Basic Idea:
Bounce event consist of series of frames so the main idea is to accumalate ball coordinates in `9` (as per the experimentations) frames and then train a model over these coordinates to learn the pattern basically learn how to tragectory looks like when a bounce event occurs. 
