# known_faces

Put example photos of each person you want the system to recognize in this folder.

## Guidelines

- You can either create one subfolder per person, e.g. `known_faces/Barack_Obama/` with several images, or place images in the root and name them like `barack_obama.jpg`, `joe_biden.png`.
- Use clear, frontal photos where the face is visible. Multiple images per person (different angles, lighting) improve accuracy.
- Supported formats: `.jpg`, `.jpeg`, `.png`.

## After adding images, run

```powershell
C:\Users\miche\miniconda3\condabin\conda.bat run -n facerec python encode_faces.py
```
