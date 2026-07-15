# Photo fixtures

Drop **real** photographs here — the ones the bot actually exists for: old
black-and-white pictures, ideally with faces and of varying quality (sharp and
blurry, portrait and group, scratched and clean).

The images themselves stay out of git (see `.gitignore`); only this README is
tracked. Other people's photographs do not belong in the repository.

What they are used for:

* **Face detection.** A detector finds nothing in a synthetic image, so without
  real photos those tests skip.
* **Quality review (phase 5).** The `original | current bot | new` contact sheet
  is built from this directory, and it is what decides the stage order, the face
  restoration model, and the DDColor variant.

Five to ten images is plenty — preferably ones the current bot handles badly.
